#!/usr/bin/env python3
"""
Valet Operations Management System with Damage Detection
- Uses Roboflow Hosted API via requests (hardcoded key/model here)
- Falls back to local YOLOv8 (if installed)
- Enforces exactly 4 photos per check-in
- Shows Minor/Moderate/Severe based on TOTAL detections across all 4 photos:
    <3 = minor, 3–6 = moderate, >6 = severe
- Clicking the severity badge opens a modal gallery with the 4 annotated images
- Allows up to 64 MB uploads and downscales images before sending to Roboflow
"""

import os, io, json, datetime, qrcode, cv2
from functools import wraps
from flask import Flask, render_template_string, request, redirect, jsonify, session, send_from_directory, url_for
from apscheduler.schedulers.background import BackgroundScheduler
from PIL import Image, ImageDraw, ImageFont
import requests

# ===================== CONFIG =====================
STATIC_DIR = "static"
UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
QRCODE_FOLDER = os.path.join(STATIC_DIR, "qrcodes")

# Roboflow Hosted API (hardcoded per your request)
RF_API_KEY  = "v7rqUvArsg97ISvd3PEj"                 # <-- your Private API Key
RF_MODEL_ID = "car-damage-assessment-8mb45-aigqn/1"   # <-- Universe model/version id
RF_ENABLED  = True

# YOLO local (optional fallback)
YOLO_ENABLED = False
yolo_model = None
try:
    from ultralytics import YOLO
    MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
    yolo_model = YOLO(MODEL_PATH)
    YOLO_ENABLED = True
    print(f"✅ YOLOv8 model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️  YOLOv8 disabled: {e}")
    print("   To enable, run: pip3 install ultralytics opencv-python torch")

# Data files
DATA_FILE = "data.json"
RUNNER_FILE = "runners.json"
SHIFTS_FILE = "shifts.json"
ADMIN_FILE = "admins.json"
ANNOUNCEMENT_FILE = "announcement.json"
CALENDAR_FILE = "calendar_events.json"

# ===================== UTILITIES =====================
def annotate_image(src_path: str, boxes: list, dst_path: str):
    """Draw red boxes + labels on the image and save to dst_path."""
    im = Image.open(src_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for b in boxes:
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        label = f"{b.get('label','damage')} {float(b.get('score',0.0)):.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        try:
            tw = draw.textlength(label, font=font)
        except Exception:
            tw = len(label) * 7
        draw.rectangle([x1, max(0, y1-18), x1+tw+6, y1], fill="red")
        draw.text((x1+3, max(0, y1-16)), label, fill="white", font=font)

    im.save(dst_path)

def prepare_image_for_api(path, max_side=1600, quality=85):
    """
    Load image, downscale so the longest side <= max_side, and JPEG-encode to bytes.
    Returns a (filename, fileobj, mimetype) triple suitable for requests 'files'.
    """
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        im = im.resize((int(w*scale), int(h*scale)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return ("image.jpg", buf, "image/jpeg")

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def generate_ticket_id():
    data = load_json(DATA_FILE)
    return str(int(data[-1]["ticketID"]) + 1).zfill(4) if data else "0001"

def generate_qr_code(ticket_id):
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(f"CHECKOUT:{ticket_id}")
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    qr_path = os.path.join(QRCODE_FOLDER, f"qr_{ticket_id}.png")
    img.save(qr_path)
    return qr_path
# ===================== DETECTION =====================
def detect_with_roboflow(img_path):
    """
    Hosted API path first; otherwise fallback to YOLO if available.
    Returns: {is_car, damage, location[], severity, boxes[], version}
    """
    # --- helpers for post-processing ---
    def _area(b): return max(0, (b["x2"] - b["x1"])) * max(0, (b["y2"] - b["y1"]))
    def _iou(a, b):
        x1 = max(a["x1"], b["x1"]); y1 = max(a["y1"], b["y1"])
        x2 = min(a["x2"], b["x2"]); y2 = min(a["y2"], b["y2"])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        ua = _area(a) + _area(b) - inter + 1e-9
        return inter / ua

    # thresholds (tune if needed)
    HARD_MIN_AREA = 800      # pixels; try 500–2000 depending on image size
    NMS_IOU       = 0.45     # suppress near-duplicates

    if RF_ENABLED:
        try:
            if not os.path.isfile(img_path):
                raise RuntimeError(f"Cannot read image: {img_path}")

            url = f"https://detect.roboflow.com/{RF_MODEL_ID}"
            params = {
                "api_key": RF_API_KEY,
                "confidence": 0.4,   # raised from 0.2 to reduce noise
                "overlap": 0.5
            }

            # Downscale and send compressed JPEG bytes to avoid 413 and speed up
            filename, fileobj, mimetype = prepare_image_for_api(img_path)
            r = requests.post(
                url, params=params,
                files={"file": (filename, fileobj, mimetype)},
                timeout=120
            )

            if r.status_code != 200:
                print(f"[RF ERROR] {r.status_code} -> {r.text}")
                raise RuntimeError(f"Roboflow error {r.status_code}")

            res = r.json()
            preds = res.get("predictions", [])

            # image size for rough location bucketing
            w = res.get("image", {}).get("width")
            h = res.get("image", {}).get("height")
            if not (w and h):
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]

            boxes, locations = [], set()
            for p in preds:
                x, y, ww, hh = p["x"], p["y"], p["width"], p["height"]
                x1, y1 = int(x - ww/2), int(y - hh/2)
                x2, y2 = int(x + ww/2), int(y + hh/2)
                label = str(p.get("class", "damage")).lower()
                conf  = float(p.get("confidence", 0.0))

                if h and w:
                    # rough location
                    if y < h/3: locations.add("front")
                    elif y > 2*h/3: locations.add("rear")
                    if x < w/3 or x > 2*w/3: locations.add("side")

                boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "label": label, "score": round(conf, 2)
                })

            # --- POST-PROCESS: tiny box filter + NMS ---
            boxes = [b for b in boxes if _area(b) >= HARD_MIN_AREA]
            boxes = sorted(boxes, key=lambda b: b["score"], reverse=True)
            keep = []
            for b in boxes:
                if all(_iou(b, k) < NMS_IOU for k in keep):
                    keep.append(b)
            boxes = keep

            damage_found = len(boxes) > 0
            is_car = bool(damage_found)  # treat any detection as car context

            # per-image severity (final severity is computed at ticket level)
            if damage_found:
                n = len(boxes)
                severity = "severe" if n >= 4 else "moderate" if n >= 2 else "minor"
            else:
                severity = "none"

            return {
                "is_car": is_car,
                "damage": damage_found,
                "location": sorted(list(locations)),
                "severity": severity,
                "boxes": boxes,
                "version": f"roboflow:{RF_MODEL_ID}"
            }

        except Exception as e:
            print(f"❌ Roboflow inference error: {e}")

    # YOLO fallback
    if not YOLO_ENABLED or yolo_model is None:
        print(f"⚠️  Skipping detection for {img_path} - no Roboflow and YOLOv8 not available")
        return {
            "is_car": True, "damage": False, "severity": "none",
            "location": [], "boxes": [], "version": "disabled"
        }

    try:
        results = yolo_model(img_path, conf=0.25, verbose=False)
        damage_keywords = ['scratch','dent','crack','broken','damage','rust','collision','shatter']
        car_keywords    = ['car','vehicle','automobile','sedan','suv','truck']

        is_car = False
        locations = set()
        boxes = []

        for result in results:
            img_height, img_width = result.orig_shape
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0]); cls_id = int(box.cls[0])
                label = result.names[cls_id].lower()
                if any(kw in label for kw in car_keywords): is_car = True
                if any(kw in label for kw in damage_keywords):
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    if cy < img_height/3: locations.add("front")
                    elif cy > 2*img_height/3: locations.add("rear")
                    if cx < img_width/3 or cx > 2*img_width/3: locations.add("side")
                    boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "label": label, "score": round(conf, 2)
                    })

        # --- POST-PROCESS: tiny box filter + NMS ---
        boxes = [b for b in boxes if _area(b) >= HARD_MIN_AREA]
        boxes = sorted(boxes, key=lambda b: b["score"], reverse=True)
        keep = []
        for b in boxes:
            if all(_iou(b, k) < NMS_IOU for k in keep):
                keep.append(b)
        boxes = keep

        damage_found = len(boxes) > 0
        if damage_found and not is_car:
            is_car = True

        if damage_found:
            n = len(boxes)
            severity = "severe" if n >= 4 else "moderate" if n >= 2 else "minor"
        else:
            severity = "none"

        return {
            "is_car": is_car, "damage": damage_found,
            "location": sorted(list(locations)), "severity": severity,
            "boxes": boxes, "version": f"yolov8:{os.getenv('YOLO_MODEL_PATH','yolov8n.pt')}"
        }

    except Exception as e:
        print(f"❌ YOLOv8 error: {e}")
        return {
            "is_car": True, "damage": False, "severity": "none",
            "location": [], "boxes": [], "version": "error"
        }

# ===================== OPTIONAL: VONAGE SMS =====================
SMS_ENABLED = False
try:
    import vonage
    VONAGE_API_KEY = os.getenv("VONAGE_API_KEY", "")
    VONAGE_API_SECRET = os.getenv("VONAGE_API_SECRET", "")
    VONAGE_PHONE_NUMBER = os.getenv("VONAGE_PHONE_NUMBER", "")
    if VONAGE_API_KEY and VONAGE_API_SECRET:
        vonage_client = vonage.Client(key=VONAGE_API_KEY, secret=VONAGE_API_SECRET)
        sms = vonage.Sms(vonage_client)
        SMS_ENABLED = True
        print("✅ Vonage SMS enabled")
except Exception as e:
    print(f"⚠️  Vonage SMS disabled: {e}")

def send_sms(to, message):
    if not SMS_ENABLED:
        print(f"📱 SMS (disabled): {to} - {message[:50]}...")
        return False
    try:
        res = sms.send_message({"from": VONAGE_PHONE_NUMBER, "to": to, "text": message})
        return res["messages"][0]["status"] == "0"
    except Exception as e:
        print(f"SMS error: {e}")
        return False

# ===================== FLASK APP =====================
app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY", "valet_secret_2024")

# 64 MB total per request (fixes 413 for 4 large photos)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

# Ensure folders exist
for folder in [STATIC_DIR, UPLOAD_FOLDER, QRCODE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config.update(UPLOAD_FOLDER=UPLOAD_FOLDER, QRCODE_FOLDER=QRCODE_FOLDER)

# Init data files
for fpath, default in [
    (DATA_FILE, []),
    (RUNNER_FILE, []),
    (SHIFTS_FILE, []),
    (ADMIN_FILE, [{"username": "admin", "password": "valet123"}]),
    (ANNOUNCEMENT_FILE, {"message": "Welcome to Valet Operations!"}),
    (CALENDAR_FILE, []),
]:
    if not os.path.exists(fpath):
        with open(fpath, "w") as f:
            json.dump(default, f, indent=2)

# --------------------- Auth helpers ---------------------
def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return func(*args, **kwargs)
    return wrapper

# ===================== ROUTES =====================
@app.route("/", methods=["GET"])
def index():
    data = load_json(DATA_FILE)
    runners = load_json(RUNNER_FILE)
    keys_in_system = len([t for t in data if t.get("status") == "Checked-In"])
    message = session.pop("message", None)
    message_type = session.pop("message_type", None)

    # Pagination
    page = int(request.args.get("page", 1))
    per_page = 15
    total = len(data)
    total_pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    end = start + per_page
    paged_data = data[start:end]

    return render_template_string(MAIN_PAGE_HTML,
        data=paged_data,
        runners=runners,
        keys_in_system=keys_in_system,
        page=page,
        total_pages=total_pages,
        message=message,
        message_type=message_type
    )

@app.route("/checkin", methods=["POST"])
def checkin():
    data = load_json(DATA_FILE)
    ticket_id = generate_ticket_id()
    now = datetime.datetime.now().strftime("%b %d, %Y %I:%M %p")

    new_ticket = {
        "ticketID": ticket_id,
        "licensePlate": request.form["licensePlate"].upper(),
        "customerName": request.form["customerName"],
        "customerPhone": request.form["customerPhone"].strip(),
        "carMake": request.form["carMake"],
        "carColor": request.form["carColor"],
        "notes": request.form.get("notes", ""),
        "status": "Checked-In",
        "checkInTime": now,
        "checkOutTime": None,
        "assignedRunner": None,
        "damageImages": [],
        "damageAnnotated": [],
        "damageSummary": {"damage": False, "severity": "none", "location": [], "modelVersion": "", "totalDetections": 0},
        "damageDetections": []
    }

    # Gather all uploaded files from supported fields
    uploaded_files = []
    for field in ("damageImages", "capturedImages"):
        if field in request.files:
            for f in request.files.getlist(field):
                if f and f.filename:
                    uploaded_files.append(f)

    # Enforce EXACTLY 4 photos
    if len(uploaded_files) != 4:
        session["message"] = f"❌ Please upload exactly 4 photos (you provided {len(uploaded_files)})."
        session["message_type"] = "warning"
        return redirect("/")

    total_detections = 0
    union_locations = set()
    model_version_used = ""

    # Save originals, run detection, save annotated copies
    for idx, img in enumerate(uploaded_files, start=1):
        safe_name = f"{ticket_id}_{idx}_{img.filename}"
        img_path = os.path.join(UPLOAD_FOLDER, safe_name)
        img.save(img_path)
        web_path = f"/static/uploads/{safe_name}"
        new_ticket["damageImages"].append(web_path)

        det = detect_with_roboflow(img_path) or {}
        boxes = det.get("boxes", [])
        total_detections += len(boxes)
        union_locations.update(det.get("location", []))
        model_version_used = det.get("version", model_version_used)

        ann_name = f"{ticket_id}_{idx}_annotated.jpg"
        ann_path = os.path.join(UPLOAD_FOLDER, ann_name)
        if boxes:
            annotate_image(img_path, boxes, ann_path)
            web_ann = f"/static/uploads/{os.path.basename(ann_path)}"
        else:
            # keep original if no boxes, for consistent gallery
            web_ann = web_path
        new_ticket["damageAnnotated"].append(web_ann)

        new_ticket["damageDetections"].append({
            "imagePath": web_path,
            "annotatedPath": web_ann,
            "damage": bool(boxes),
            "severity": "n/a",
            "location": det.get("location", []),
            "boxes": boxes,
        })

    # Overall severity by TOTAL detections (your rule)
    if total_detections > 6:
        overall_severity = "severe"
    elif 3 <= total_detections <= 6:
        overall_severity = "moderate"
    else:
        overall_severity = "minor"

    any_damage = total_detections > 0

    new_ticket["damageSummary"].update({
        "damage": any_damage,
        "severity": overall_severity,
        "location": sorted(union_locations),
        "modelVersion": model_version_used,
        "totalDetections": total_detections
    })

    data.append(new_ticket)
    save_json(DATA_FILE, data)
    generate_qr_code(ticket_id)

    # SMS (best-effort)
    try:
        qr_url = url_for('static', filename=f"qrcodes/qr_{ticket_id}.png", _external=True)
        msg = f"Hi {new_ticket['customerName']}! Vehicle checked in. Ticket #{ticket_id}. QR: {qr_url}"
        ok = send_sms(new_ticket["customerPhone"], msg)
        session["message"] = f"✅ Vehicle #{ticket_id} checked in" + (" (SMS sent)" if ok else "")
    except:
        session["message"] = f"✅ Vehicle #{ticket_id} checked in"

    session["message_type"] = "success"
    return redirect("/")

@app.route("/checkout/<ticket_id>", methods=["POST"])
def checkout(ticket_id):
    data = load_json(DATA_FILE)
    now = datetime.datetime.now().strftime("%b %d, %Y %I:%M %p")
    for t in data:
        if t.get("ticketID") == ticket_id and t.get("status") == "Checked-In":
            t["status"] = "Checked-Out"
            t["checkOutTime"] = now
            save_json(DATA_FILE, data)
            send_sms(t.get("customerPhone", ""), f"Thank you! Ticket #{ticket_id} checked out.")
            session["message"] = f"✅ Ticket #{ticket_id} checked out"
            session["message_type"] = "success"
            return redirect("/")
    session["message"] = f"❌ Ticket #{ticket_id} not found"
    session["message_type"] = "warning"
    return redirect("/")

@app.route("/checkout_manual", methods=["POST"])
def checkout_manual():
    return checkout(request.form.get("ticket_id"))

@app.route("/delete/<ticket_id>", methods=["POST"])
def delete_ticket(ticket_id):
    data = load_json(DATA_FILE)
    data = [t for t in data if t.get("ticketID") != ticket_id]
    save_json(DATA_FILE, data)
    session["message"] = f"🗑️ Ticket #{ticket_id} deleted"
    session["message_type"] = "warning"
    return redirect("/")

@app.route("/qrcode/<ticket_id>")
def view_qrcode(ticket_id):
    qr_path = os.path.join(QRCODE_FOLDER, f"qr_{ticket_id}.png")
    if not os.path.exists(qr_path):
        generate_qr_code(ticket_id)
    return f'''<html><body style="text-align:center;padding:50px;">
    <h1>Ticket #{ticket_id}</h1>
    <img src="/static/qrcodes/qr_{ticket_id}.png" style="border:4px solid #000;padding:20px;">
    <br><br><button onclick="window.print()">Print</button>
    <button onclick="window.close()">Close</button>
    </body></html>'''

@app.route("/assign_runner/<ticket_id>", methods=["POST"])
def assign_runner(ticket_id):
    data = load_json(DATA_FILE)
    runner = request.form.get("runnerName")
    for t in data:
        if t.get("ticketID") == ticket_id:
            t["assignedRunner"] = runner
            break
    save_json(DATA_FILE, data)
    session["message"] = f"✅ Runner {runner} assigned to #{ticket_id}"
    return redirect("/")

@app.route("/vehicle_ready/<ticket_id>", methods=["POST"])
def vehicle_ready(ticket_id):
    data = load_json(DATA_FILE)
    for t in data:
        if t.get("ticketID") == ticket_id:
            send_sms(t.get("customerPhone", ""), f"Your vehicle (#{ticket_id}) is ready!")
            session["message"] = "📱 SMS sent"
            break
    return redirect("/")

@app.route("/runner_clockin")
def runner_clockin_page():
    runners = load_json(RUNNER_FILE)
    return render_template_string(RUNNER_PAGE_HTML, runners=runners)

@app.route("/clockin", methods=["POST"])
def clockin():
    runners = load_json(RUNNER_FILE)
    name = request.form.get("runnerName", "").strip()
    if name and not any(r.get("name", "").lower() == name.lower() and "clockOutTime" not in r for r in runners):
        runners.append({"name": name, "clockInTime": datetime.datetime.now().strftime("%b %d, %Y %I:%M %p")})
        save_json(RUNNER_FILE, runners)
        session["message"] = f"✅ {name} clocked in"
    return redirect("/runner_clockin")

@app.route("/clockout", methods=["POST"])
def clockout():
    runners = load_json(RUNNER_FILE)
    name = request.form.get("runnerName", "").strip()
    now = datetime.datetime.now()
    for r in runners:
        if r.get("name", "").lower() == name.lower() and "clockOutTime" not in r:
            r["clockOutTime"] = now.strftime("%b %d, %Y %I:%M %p")
            cin = datetime.datetime.strptime(r["clockInTime"], "%b %d, %Y %I:%M %p")
            secs = (now - cin).total_seconds()
            r["duration"] = f"{int(secs//3600)}h {int((secs%3600)//60)}m"
            break
    save_json(RUNNER_FILE, runners)
    return redirect("/runner_clockin")

@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        admins = load_json(ADMIN_FILE)
        u, p = request.form.get("username"), request.form.get("password")
        if any(a.get("username") == u and a.get("password") == p for a in admins):
            session["admin_logged_in"] = True
            return redirect("/admin")
        return "<h3 style='color:red;text-align:center;'>Invalid credentials</h3>" + LOGIN_HTML
    return LOGIN_HTML

@app.route("/admin_logout")
@login_required
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect("/")

@app.route("/admin")
@login_required
def admin_dashboard():
    return ADMIN_HTML

@app.route("/admin_announcement", methods=["POST"])
@login_required
def admin_announcement():
    msg = request.form.get("message", "").strip()
    if msg:
        save_json(ANNOUNCEMENT_FILE, {"message": f"Admin: {msg}"})
    return redirect("/admin")

@app.route("/announcement_page")
def announcement_page():
    return render_template_string(ANNOUNCEMENT_HTML)

@app.route("/announcement")
def get_announcement():
    return jsonify(load_json(ANNOUNCEMENT_FILE))

@app.route("/shift_portal")
def shift_portal():
    return SHIFT_PORTAL_HTML

@app.route("/shifts")
def get_shifts():
    return jsonify(load_json(SHIFTS_FILE))

@app.route("/add_shift", methods=["POST"])
def add_shift():
    data = load_json(SHIFTS_FILE)
    payload = request.get_json(force=True)
    new_id = (max([s.get("id", 0) for s in data]) + 1) if data else 1
    data.append({"id": new_id, "day": payload.get("day", "").strip(), "time": payload.get("time", "").strip(), "assigned_to": None})
    save_json(SHIFTS_FILE, data)
    return jsonify({"ok": True})

@app.route("/pick_shift", methods=["POST"])
def pick_shift():
    data = load_json(SHIFTS_FILE)
    payload = request.get_json(force=True)
    sid = int(payload.get("id"))
    user = payload.get("user", "").strip()
    for s in data:
        if s.get("id") == sid and not s.get("assigned_to"):
            s["assigned_to"] = user
            save_json(SHIFTS_FILE, data)
            return jsonify({"ok": True})
    return jsonify({"error": "Not available"}), 400

@app.route("/drop_shift", methods=["POST"])
def drop_shift():
    data = load_json(SHIFTS_FILE)
    payload = request.get_json(force=True)
    sid = int(payload.get("id"))
    user = payload.get("user", "").strip()
    for s in data:
        if s.get("id") == sid and s.get("assigned_to", "").lower() == user.lower():
            s["assigned_to"] = None
            save_json(SHIFTS_FILE, data)
            return jsonify({"ok": True})
    return jsonify({"error": "Cannot drop"}), 400

@app.route("/delete_shift", methods=["POST"])
def delete_shift():
    data = load_json(SHIFTS_FILE)
    sid = int(request.get_json(force=True).get("id"))
    data = [s for s in data if s.get("id") != sid]
    save_json(SHIFTS_FILE, data)
    return jsonify({"ok": True})

@app.route("/calendar")
def calendar_view():
    return CALENDAR_HTML

@app.route("/<path:filename>")
def serve_file(filename):
    return send_from_directory(".", filename)

# ===================== PROFESSIONAL HTML TEMPLATES =====================

LOGIN_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Login - Valet Management System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-container {
            background: white;
            padding: 50px 60px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            width: 100%;
            max-width: 420px;
        }
        h2 {
            color: #2c3e50;
            margin-bottom: 30px;
            text-align: center;
            font-size: 28px;
            font-weight: 600;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
            font-size: 14px;
        }
        input[type="text"], input[type="password"] {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 15px;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus, input[type="password"]:focus {
            outline: none;
            border-color: #667eea;
        }
        button[type="submit"] {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 10px;
        }
        button[type="submit"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        .back-btn {
            width: 100%;
            padding: 12px;
            background: white;
            color: #667eea;
            border: 2px solid #667eea;
            border-radius: 6px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 15px;
            transition: all 0.3s;
        }
        .back-btn:hover {
            background: #667eea;
            color: white;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <h2>Administrator Login</h2>
        <form method="POST">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" required autofocus>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit">Sign In</button>
        </form>
        <button class="back-btn" onclick="location.href='/'">Back to Dashboard</button>
    </div>
</body>
</html>'''

ADMIN_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Panel - Valet Management System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            padding: 40px 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        h2 {
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 32px;
            font-weight: 600;
        }
        h3 {
            color: #34495e;
            margin: 30px 0 20px 0;
            font-size: 20px;
            font-weight: 600;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }
        textarea {
            width: 100%;
            padding: 14px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 15px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #3498db;
        }
        button {
            padding: 12px 28px;
            border: none;
            border-radius: 6px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin: 10px 5px 0 0;
        }
        button[type="submit"] {
            background: #3498db;
            color: white;
        }
        button[type="submit"]:hover {
            background: #2980b9;
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
        }
        .btn-secondary {
            background: #95a5a6;
            color: white;
        }
        .btn-secondary:hover {
            background: #7f8c8d;
        }
        .btn-logout {
            background: #e74c3c;
            color: white;
        }
        .btn-logout:hover {
            background: #c0392b;
        }
        hr {
            margin: 30px 0;
            border: none;
            border-top: 1px solid #e0e0e0;
        }
        .button-group {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Administration Panel</h2>
        
        <h3>Post System Announcement</h3>
        <form method="POST" action="/admin_announcement">
            <textarea name="message" rows="4" placeholder="Enter announcement message..." required></textarea>
            <div class="button-group">
                <button type="submit">Publish Announcement</button>
            </div>
        </form>
        
        <hr>
        
        <div class="button-group">
            <button class="btn-secondary" onclick="location.href='/'">Return to Dashboard</button>
            <button class="btn-logout" onclick="location.href='/admin_logout'">Logout</button>
        </div>
    </div>
</body>
</html>'''

ANNOUNCEMENT_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Announcements - Valet Management System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            padding: 40px 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        h2 {
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 32px;
            font-weight: 600;
        }
        #msg {
            font-size: 18px;
            line-height: 1.6;
            color: #34495e;
            background: #ecf0f1;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
            min-height: 60px;
        }
        button {
            padding: 12px 28px;
            background: #95a5a6;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 25px;
            transition: all 0.3s;
        }
        button:hover {
            background: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>System Announcements</h2>
        <div id="msg">Loading announcement...</div>
        <button onclick="location.href='/'">Back to Dashboard</button>
    </div>
    <script>
        fetch("/announcement")
            .then(r => r.json())
            .then(d => {
                document.getElementById("msg").innerText = d.message || "No announcements at this time.";
            })
            .catch(() => {
                document.getElementById("msg").innerText = "Unable to load announcements.";
            });
    </script>
</body>
</html>'''

RUNNER_PAGE_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Runner Management - Valet Operations</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            padding: 40px 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 32px;
            font-weight: 600;
        }
        .subtitle {
            color: #7f8c8d;
            margin-bottom: 35px;
            font-size: 16px;
        }
        .form-section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
            border: 1px solid #e0e0e0;
        }
        input[type="text"] {
            width: 100%;
            padding: 14px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 15px;
            margin-bottom: 15px;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #8e44ad;
        }
        button {
            padding: 12px 28px;
            border: none;
            border-radius: 6px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-right: 10px;
        }
        .btn-primary {
            background: #8e44ad;
            color: white;
        }
        .btn-primary:hover {
            background: #732d91;
            box-shadow: 0 4px 12px rgba(142, 68, 173, 0.3);
        }
        .btn-warning {
            background: #e67e22;
            color: white;
            padding: 8px 18px;
            font-size: 14px;
        }
        .btn-warning:hover {
            background: #d35400;
        }
        .btn-back {
            background: #e74c3c;
            color: white;
            margin-top: 20px;
        }
        .btn-back:hover {
            background: #c0392b;
        }
        h3 {
            color: #34495e;
            margin: 30px 0 20px 0;
            font-size: 20px;
            font-weight: 600;
            padding-bottom: 10px;
            border-bottom: 2px solid #8e44ad;
        }
        .runner-item {
            background: #ffffff;
            padding: 18px;
            margin: 12px 0;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
            border-left: 4px solid #8e44ad;
            transition: box-shadow 0.3s;
        }
        .runner-item:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .runner-name {
            font-weight: 600;
            color: #2c3e50;
            font-size: 16px;
            margin-bottom: 8px;
        }
        .runner-details {
            color: #7f8c8d;
            font-size: 14px;
        }
        .no-data {
            text-align: center;
            color: #95a5a6;
            padding: 40px;
            font-size: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Runner Management</h1>
        <p class="subtitle">Clock in/out and track runner activity</p>
        
        <div class="form-section">
            <form action="/clockin" method="post">
                <input type="text" name="runnerName" placeholder="Enter runner name" required autofocus>
                <button type="submit" class="btn-primary">Clock In</button>
            </form>
        </div>
        
        <h3>Activity Log</h3>
        {% if runners %}
            {% for r in runners %}
            <div class="runner-item">
                <div class="runner-name">{{ r.name }}</div>
                {% if r.get('clockOutTime') %}
                    <div class="runner-details">
                        Clock In: {{ r.clockInTime }} | Clock Out: {{ r.clockOutTime }} | Duration: {{ r.get('duration', 'N/A') }}
                    </div>
                {% else %}
                    <div class="runner-details">
                        Active since: {{ r.clockInTime }}
                        <form action="/clockout" method="post" style="display:inline; margin-left:15px;">
                            <input type="hidden" name="runnerName" value="{{ r.name }}">
                            <button type="submit" class="btn-warning">Clock Out</button>
                        </form>
                    </div>
                {% endif %}
            </div>
            {% endfor %}
        {% else %}
            <div class="no-data">No runner activity recorded yet</div>
        {% endif %}
        
        <button class="btn-back" onclick="location.href='/'">Back to Dashboard</button>
    </div>
</body>
</html>'''

SHIFT_PORTAL_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shift Management - Valet Operations</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            padding: 40px 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h2 {
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 32px;
            font-weight: 600;
        }
        .add-shift-section {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-bottom: 30px;
        }
        h3 {
            color: #34495e;
            margin-bottom: 20px;
            font-size: 20px;
            font-weight: 600;
        }
        select {
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 15px;
            margin-right: 10px;
            min-width: 160px;
            background: white;
            cursor: pointer;
            transition: border-color 0.3s;
        }
        select:focus {
            outline: none;
            border-color: #3498db;
        }
        button {
            padding: 12px 28px;
            border: none;
            border-radius: 6px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-add {
            background: #3498db;
            color: white;
        }
        .btn-add:hover {
            background: #2980b9;
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
        }
        .btn-pick {
            background: #27ae60;
            color: white;
            padding: 8px 18px;
            font-size: 14px;
        }
        .btn-pick:hover {
            background: #229954;
        }
        .btn-drop {
            background: #e67e22;
            color: white;
            padding: 8px 18px;
            font-size: 14px;
        }
        .btn-drop:hover {
            background: #d35400;
        }
        .btn-delete {
            background: #e74c3c;
            color: white;
            padding: 8px 18px;
            font-size: 14px;
        }
        .btn-delete:hover {
            background: #c0392b;
        }
        .btn-back {
            background: #2c3e50;
            color: white;
            margin-top: 25px;
        }
        .btn-back:hover {
            background: #1a252f;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border-radius: 12px;
            overflow: hidden;
        }
        th, td {
            padding: 16px;
            text-align: left;
        }
        th {
            background: #b30000;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 13px;
            letter-spacing: 0.5px;
        }
        td {
            border-bottom: 1px solid #e0e0e0;
        }
        tr:last-child td {
            border-bottom: none;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .shift-id {
            font-weight: 600;
            color: #2c3e50;
        }
        .no-shifts {
            text-align: center;
            padding: 40px;
            color: #95a5a6;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Shift Management</h2>
        
        <div class="add-shift-section">
            <h3>Create New Shift</h3>
            <select id="daySelect">
                <option value="">Select Day</option>
                <option>Monday</option>
                <option>Tuesday</option>
                <option>Wednesday</option>
                <option>Thursday</option>
                <option>Friday</option>
                <option>Saturday</option>
                <option>Sunday</option>
            </select>
            <select id="timeSelect">
                <option value="">Select Time</option>
                <option>6am-2pm</option>
                <option>7am-3pm</option>
                <option>8am-4pm</option>
                <option>9am-5pm</option>
                <option>2pm-10pm</option>
            </select>
            <button onclick="addShift()" class="btn-add">Add Shift</button>
        </div>
        
        <div id="shiftTable"></div>
        
        <button class="btn-back" onclick="location.href='/'">Back to Dashboard</button>
    </div>
    
    <script>
    async function loadShifts(){
        const res = await fetch("/shifts");
        const data = await res.json();
        let html = "<table><thead><tr><th>Shift ID</th><th>Day</th><th>Time Slot</th><th>Assigned To</th><th>Actions</th></tr></thead><tbody>";
        
        if(data.length === 0){
            html += "<tr><td colspan='5' class='no-shifts'>No shifts scheduled</td></tr>";
        } else {
            data.forEach(s => {
                html += `<tr>
                    <td class="shift-id">#${s.id}</td>
                    <td>${s.day || ''}</td>
                    <td>${s.time || ''}</td>
                    <td>${s.assigned_to || '<span style="color:#95a5a6;">Available</span>'}</td>
                    <td>
                        ${s.assigned_to
                            ? `<button class="btn-drop" onclick="drop(${s.id}, '${s.assigned_to}')">Unassign</button>`
                            : `<button class="btn-pick" onclick="pick(${s.id})">Assign Shift</button>`
                        }
                        <button class="btn-delete" onclick="del(${s.id})">Delete</button>
                    </td>
                </tr>`;
            });
        }
        
        html += "</tbody></table>";
        document.getElementById("shiftTable").innerHTML = html;
    }
    
    async function addShift(){
        const day = document.getElementById("daySelect").value;
        const time = document.getElementById("timeSelect").value;
        if(!day || !time){
            alert("Please select both day and time");
            return;
        }
        await fetch("/add_shift", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({day, time})
        });
        loadShifts();
    }
    
    async function pick(id){
        const user = prompt("Enter your name:");
        if(!user) return;
        await fetch("/pick_shift", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({id, user})
        });
        loadShifts();
    }
    
    async function drop(id, user){
        const u = prompt(`Enter name to confirm (must match: ${user}):`);
        if(!u || u.toLowerCase() != user.toLowerCase()) return;
        await fetch("/drop_shift", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({id, user: u})
        });
        loadShifts();
    }
    
    async function del(id){
        if(!confirm("Are you sure you want to delete this shift?")) return;
        await fetch("/delete_shift", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({id})
        });
        loadShifts();
    }
    
    loadShifts();
    </script>
</body>
</html>'''

CALENDAR_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shift Calendar - Valet Operations</title>
    <link href="https://cdn.jsdelivr.net/npm/fullcalendar@6.1.10/index.global.min.css" rel="stylesheet"/>
    <script src="https://cdn.jsdelivr.net/npm/fullcalendar@6.1.10/index.global.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            padding: 40px 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h2 {
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 32px;
            font-weight: 600;
            text-align: center;
        }
        #calendar {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        button {
            padding: 12px 28px;
            background: #2c3e50;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 25px;
            transition: all 0.3s;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        button:hover {
            background: #1a252f;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Shift Calendar</h2>
        <div id="calendar"></div>
        <button onclick="location.href='/'">Back to Dashboard</button>
    </div>
    
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const cal = new FullCalendar.Calendar(document.getElementById('calendar'), {
            initialView: 'dayGridMonth',
            headerToolbar: {
                left: 'prev,next today',
                center: 'title',
                right: 'dayGridMonth,timeGridWeek'
            },
            events: []
        });
        cal.render();
    });
    </script>
</body>
</html>'''

MAIN_PAGE_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Valet Operations System - Moffitt Cancer Center</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            color: #2c3e50;
        }
        
        .header {
            background: linear-gradient(135deg, #b30000 0%, #8b0000 100%);
            color: white;
            padding: 30px 40px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            position: relative;
        }
        
        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 6px;
        }
        
        .header h2 {
            font-size: 16px;
            font-weight: 400;
            opacity: 0.95;
        }
        
        .admin-link {
            background: rgba(255,255,255,0.15);
            color: white;
            padding: 10px 20px;
            border-radius: 6px;
            text-decoration: none;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.3s;
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        .admin-link:hover {
            background: rgba(255,255,255,0.25);
        }
        
        .container {
            max-width: 1400px;
            margin: 30px auto;
            padding: 0 20px;
        }
        
        .alert {
            padding: 16px 20px;
            margin-bottom: 25px;
            border-radius: 8px;
            font-weight: 500;
            border-left: 4px solid;
        }
        
        .alert.success {
            background: #d4edda;
            color: #155724;
            border-color: #28a745;
        }
        
        .alert.warning {
            background: #fff3cd;
            color: #856404;
            border-color: #ffc107;
        }
        
        .section {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-bottom: 25px;
        }
        
        .nav-buttons {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 25px;
        }
        
        .nav-buttons button {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-purple { background: #8e44ad; color: white; }
        .btn-purple:hover { background: #732d91; box-shadow: 0 4px 12px rgba(142,68,173,0.3); }
        
        .btn-blue { background: #3498db; color: white; }
        .btn-blue:hover { background: #2980b9; box-shadow: 0 4px 12px rgba(52,152,219,0.3); }
        
        .btn-orange { background: #f39c12; color: white; }
        .btn-orange:hover { background: #e67e22; box-shadow: 0 4px 12px rgba(243,156,18,0.3); }
        
        .btn-teal { background: #16a085; color: white; }
        .btn-teal:hover { background: #138d75; box-shadow: 0 4px 12px rgba(22,160,133,0.3); }
        
        h3 {
            color: #2c3e50;
            font-size: 22px;
            font-weight: 600;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid #b30000;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 15px;
        }
        
        input[type="text"],
        input[type="tel"],
        input[type="file"] {
            padding: 12px 14px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus,
        input[type="tel"]:focus {
            outline: none;
            border-color: #27ae60;
        }
        
        label {
            display: block;
            margin: 15px 0 8px 0;
            font-weight: 600;
            font-size: 14px;
            color: #34495e;
        }
        
        button[type="submit"] {
            padding: 12px 32px;
            border: none;
            border-radius: 6px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 15px;
        }
        
        .btn-submit {
            background: #27ae60;
            color: white;
        }
        
        .btn-submit:hover {
            background: #229954;
            box-shadow: 0 4px 12px rgba(39,174,96,0.3);
        }
        
        .checkout-form {
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }
        
        .checkout-form input {
            flex: 0 0 250px;
        }
        
        .btn-checkout {
            background: #3498db;
            color: white;
            padding: 12px 28px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-checkout:hover {
            background: #2980b9;
        }
        
        .key-counter {
            background: linear-gradient(135deg, #b30000 0%, #8b0000 100%);
            color: white;
            padding: 18px 28px;
            border-radius: 8px;
            font-weight: 600;
            display: inline-block;
            margin: 20px 0;
            font-size: 18px;
            box-shadow: 0 4px 12px rgba(179,0,0,0.3);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
        }
        
        th, td {
            padding: 14px 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        
        th {
            background: #2c3e50;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
        }
        
        tr:hover {
            background: #f8f9fa;
        }
        
        .ticket-id {
            background: #b30000;
            color: white;
            font-weight: 700;
            text-align: center;
            padding: 8px;
            border-radius: 4px;
        }
        
        .status-active {
            color: #27ae60;
            font-weight: 600;
        }
        
        .status-complete {
            color: #95a5a6;
            font-weight: 600;
        }
        
        .damage-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        
        .damage-none {
            background: #d4edda;
            color: #155724;
        }
        
        .damage-minor {
            background: #fff3cd;
            color: #856404;
        }
        
        .damage-moderate {
            background: #ffc107;
            color: #000;
        }
        
        .damage-severe {
            background: #f8d7da;
            color: #721c24;
        }
        
        .action-buttons {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }
        
        .action-buttons button,
        .action-buttons select {
            padding: 6px 12px;
            border: none;
            border-radius: 4px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
        }
        
        .action-buttons select {
            border: 2px solid #e0e0e0;
            background: white;
        }
        
        .btn-assign {
            background: #8e44ad;
            color: white;
        }
        
        .btn-ready {
            background: #f39c12;
            color: white;
        }
        
        .btn-checkout-action {
            background: #3498db;
            color: white;
        }
        
        .btn-delete {
            background: #e74c3c;
            color: white;
        }
        
        .btn-qr {
            background: #16a085;
            color: white;
            text-decoration: none;
            display: inline-block;
            padding: 6px 12px;
            border-radius: 4px;
        }
        
        .pagination {
            text-align: center;
            margin-top: 30px;
            padding-top: 25px;
            border-top: 1px solid #e0e0e0;
        }
        
        .pagination button {
            background: #b30000;
            color: white;
            border: none;
            padding: 10px 18px;
            margin: 3px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .pagination button:hover {
            background: #8b0000;
        }
        
        .pagination .current {
            color: #b30000;
            font-weight: 700;
            margin: 0 8px;
        }
        
        .modal {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.85);
            z-index: 9999;
            overflow-y: auto;
        }
        
        .modal-content {
            max-width: 1200px;
            margin: 40px auto;
            background: white;
            border-radius: 12px;
            padding: 30px;
            position: relative;
        }
        
        .modal-close {
            position: absolute;
            right: 20px;
            top: 20px;
            background: #e74c3c;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
        }
        
        .modal-close:hover {
            background: #c0392b;
        }
        
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .gallery-item {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .gallery-item img {
            width: 100%;
            display: block;
            max-height: 450px;
            object-fit: contain;
            background: #000;
        }
        
        .no-data {
            text-align: center;
            padding: 60px 20px;
            color: #95a5a6;
            font-size: 16px;
        }
        
        hr {
            margin: 30px 0;
            border: none;
            border-top: 1px solid #e0e0e0;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div>
                <h1>Valet Operations System</h1>
                <h2>Moffitt Cancer Center — Red Ramp Valet</h2>
            </div>
            <a href="/admin_login" class="admin-link">Administrator</a>
        </div>
    </div>
    
    <div class="container">
        {% if message %}
        <div class="alert {{message_type}}">{{message}}</div>
        {% endif %}
        
        <div class="nav-buttons">
            <button class="btn-purple" onclick="location.href='/runner_clockin'">Runner Management</button>
            <button class="btn-blue" onclick="location.href='/shift_portal'">Shift Management</button>
            <button class="btn-orange" onclick="location.href='/calendar'">Shift Calendar</button>
            <button class="btn-teal" onclick="location.href='/announcement_page'">Announcements</button>
        </div>
        
        <div class="section">
            <h3>Vehicle Check-In</h3>
            <form action="/checkin" method="post" enctype="multipart/form-data">
                <div class="form-row">
                    <input type="text" name="licensePlate" placeholder="License Plate" required>
                    <input type="text" name="customerName" placeholder="Customer Name" required>
                    <input type="tel" name="customerPhone" placeholder="+12345678901" required>
                </div>
                <div class="form-row">
                    <input type="text" name="carMake" placeholder="Car Make" required>
                    <input type="text" name="carColor" placeholder="Car Color" required>
                    <input type="text" name="notes" placeholder="Notes (optional)">
                </div>
                <label>Upload damage assessment photos (exactly 4 required):</label>
                <input type="file" name="damageImages" accept="image/*" multiple>
                <button type="submit" class="btn-submit">Complete Check-In</button>
            </form>
        </div>
        
        <div class="section">
            <h3>Vehicle Checkout</h3>
            <form action="/checkout_manual" method="post" class="checkout-form">
                <div>
                    <label>Ticket Number</label>
                    <input type="text" name="ticket_id" placeholder="Enter ticket number" required>
                </div>
                <button type="submit" class="btn-checkout">Process Checkout</button>
            </form>
        </div>
        
        <hr>
        
        <div class="section">
            <h3>Active Tickets</h3>
            <div style="text-align:center;">
                <div class="key-counter">Keys in System: {{keys_in_system}}</div>
            </div>
            
            {% if data %}
            <table>
                <thead>
                    <tr>
                        <th>Ticket</th>
                        <th>License</th>
                        <th>Customer</th>
                        <th>Phone</th>
                        <th>Vehicle</th>
                        <th>Color</th>
                        <th>Status</th>
                        <th>Check-In Time</th>
                        <th>Runner</th>
                        <th>Damage</th>
                        <th>QR Code</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {% for t in data %}
                    <tr>
                        <td><span class="ticket-id">{{t.ticketID}}</span></td>
                        <td>{{t.licensePlate}}</td>
                        <td>{{t.customerName}}</td>
                        <td>{{t.customerPhone}}</td>
                        <td>{{t.carMake}}</td>
                        <td>{{t.carColor}}</td>
                        <td>
                            <span class="{% if t.status=='Checked-In' %}status-active{% else %}status-complete{% endif %}">
                                {{t.status}}
                            </span>
                        </td>
                        <td>{{t.checkInTime}}</td>
                        <td>{{t.assignedRunner or 'Unassigned'}}</td>
                        
                        <td>
                            {% set ds = t.get('damageSummary', {}) %}
                            {% if ds.get('damage') %}
                                <span
                                    class="damage-badge damage-{{ds.get('severity','none')}}"
                                    title="AI Model: {{ds.get('modelVersion','')}} | Detections: {{ds.get('totalDetections', 0)}}"
                                    style="cursor:pointer;"
                                    onclick="openGallery('{{ t.ticketID }}')"
                                >
                                    {{ds.get('severity','none')|capitalize}}
                                    {% if ds.get('location') %}<br>{{', '.join(ds.get('location'))}}{% endif %}
                                </span>
                                <script type="application/json" id="ann-{{ t.ticketID }}">
                                    {{ (t.get('damageAnnotated', []) or []) | tojson }}
                                </script>
                            {% else %}
                                <span class="damage-badge damage-none">None Detected</span>
                            {% endif %}
                        </td>
                        
                        <td>
                            {% if t.status == 'Checked-In' %}
                                <a href="/qrcode/{{t.ticketID}}" target="_blank" class="btn-qr">View QR</a>
                            {% else %}
                                —
                            {% endif %}
                        </td>
                        
                        <td>
                            <div class="action-buttons">
                                {% if t.status == 'Checked-In' %}
                                    <form action="/assign_runner/{{t.ticketID}}" method="post" style="display:inline;">
                                        <select name="runnerName" required>
                                            <option value="">Assign Runner</option>
                                            {% for r in runners %}
                                                <option value="{{r.name}}">{{r.name}}</option>
                                            {% endfor %}
                                        </select>
                                        <button type="submit" class="btn-assign">Assign</button>
                                    </form>
                                    <form action="/vehicle_ready/{{t.ticketID}}" method="post" style="display:inline;">
                                        <button type="submit" class="btn-ready">Ready</button>
                                    </form>
                                    <form action="/checkout/{{t.ticketID}}" method="post" style="display:inline;">
                                        <button type="submit" class="btn-checkout-action">Checkout</button>
                                    </form>
                                {% endif %}
                                <form action="/delete/{{t.ticketID}}" method="post" style="display:inline;">
                                    <button type="submit" class="btn-delete" onclick="return confirm('Delete this ticket?')">Delete</button>
                                </form>
                            </div>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            
            <div id="galleryModal" class="modal">
                <div class="modal-content">
                    <button class="modal-close" onclick="closeGallery()">Close</button>
                    <h3>Damage Assessment Photos (AI-Annotated)</h3>
                    <div id="galleryGrid" class="gallery-grid"></div>
                </div>
            </div>
            
            <script>
            function openGallery(ticketId){
                const el = document.getElementById('ann-' + ticketId);
                if(!el) return;
                let imgs = [];
                try { imgs = JSON.parse(el.textContent || '[]'); } catch(e){ imgs = []; }
                const grid = document.getElementById('galleryGrid');
                grid.innerHTML = '';
                if(!imgs || !imgs.length){
                    grid.innerHTML = '<p class="no-data">No annotated images available for this ticket.</p>';
                } else {
                    imgs.forEach(src=>{
                        const card = document.createElement('div');
                        card.className = 'gallery-item';
                        card.innerHTML = `<img src="${src}" alt="Damage photo">`;
                        grid.appendChild(card);
                    });
                }
                document.getElementById('galleryModal').style.display='block';
            }
            function closeGallery(){ document.getElementById('galleryModal').style.display='none'; }
            </script>
            
            <div class="pagination">
                {% if page > 1 %}
                    <a href="/?page={{page-1}}"><button>Previous</button></a>
                {% endif %}
                {% for i in range(1, total_pages + 1) %}
                    {% if i == page %}
                        <span class="current">{{i}}</span>
                    {% else %}
                        <a href="/?page={{i}}"><button>{{i}}</button></a>
                    {% endif %}
                {% endfor %}
                {% if page < total_pages %}
                    <a href="/?page={{page+1}}"><button>Next</button></a>
                {% endif %}
            </div>
            
            {% else %}
            <p class="no-data">No tickets in the system. Check in a vehicle to begin operations.</p>
            {% endif %}
        </div>
    </div>
</body>
</html>'''
# ===================== MAIN =====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚗 VALET OPERATIONS MANAGEMENT SYSTEM")
    print("="*70)
    port = int(os.getenv("PORT", 5050))
    print(f"🌐 URL: http://127.0.0.1:{port}")
    print(f"📱 SMS: {'ENABLED ✅' if SMS_ENABLED else 'DISABLED ⚠️'}")
    print(f"🌐 Roboflow: {'ENABLED ✅' if RF_ENABLED else 'DISABLED ⚠️'}")
    print(f"🤖 YOLOv8: {'ENABLED ✅' if YOLO_ENABLED else 'DISABLED ⚠️  (pip3 install ultralytics)'}")
    if YOLO_ENABLED: print(f"📦 YOLO Model: {os.getenv('YOLO_MODEL_PATH', 'yolov8n.pt')}")
    if RF_ENABLED:  print(f"📦 RF Model: {RF_MODEL_ID}")
    print("="*70 + "\n")
    app.run(host="0.0.0.0", port=port, debug=True)
