#!/usr/bin/env python3
"""
Valet Operations Management System with Damage Detection
- Red valet branding with blue gradient background
- Login required for all access
- Admin login for announcements
- Runner assignment during checkout
- Modal QR code display
"""

import os, io, json, datetime, qrcode, cv2
from functools import wraps
from flask import Flask, render_template_string, request, redirect, jsonify, session, send_from_directory, url_for
from PIL import Image, ImageDraw, ImageFont
import requests

# ===================== CONFIG =====================
STATIC_DIR = "static"
UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
QRCODE_FOLDER = os.path.join(STATIC_DIR, "qrcodes")

# Roboflow Hosted API
RF_API_KEY = "v7rqUvArsg97ISvd3PEj"
RF_MODEL_ID = "car-damage-assessment-8mb45-aigqn/1"
RF_ENABLED = True

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
    print(f"⚠️ YOLOv8 disabled: {e}")

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
    """Load image, downscale and JPEG-encode to bytes."""
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
    """Hosted API path first; otherwise fallback to YOLO if available."""
    def _area(b):
        return max(0, (b["x2"] - b["x1"])) * max(0, (b["y2"] - b["y1"]))

    def _iou(a, b):
        x1 = max(a["x1"], b["x1"]); y1 = max(a["y1"], b["y1"])
        x2 = min(a["x2"], b["x2"]); y2 = min(a["y2"], b["y2"])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        ua = _area(a) + _area(b) - inter + 1e-9
        return inter / ua

    HARD_MIN_AREA = 800
    NMS_IOU = 0.45

    if RF_ENABLED:
        try:
            if not os.path.isfile(img_path):
                raise RuntimeError(f"Cannot read image: {img_path}")

            url = f"https://detect.roboflow.com/{RF_MODEL_ID}"
            params = {"api_key": RF_API_KEY, "confidence": 0.4, "overlap": 0.5}
            filename, fileobj, mimetype = prepare_image_for_api(img_path)
            r = requests.post(url, params=params, files={"file": (filename, fileobj, mimetype)}, timeout=120)
            
            if r.status_code != 200:
                print(f"[RF ERROR] {r.status_code} -> {r.text}")
                raise RuntimeError(f"Roboflow error {r.status_code}")

            res = r.json()
            preds = res.get("predictions", [])
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
                conf = float(p.get("confidence", 0.0))

                if h and w:
                    if y < h/3:
                        locations.add("front")
                    elif y > 2*h/3:
                        locations.add("rear")
                    if x < w/3 or x > 2*w/3:
                        locations.add("side")

                boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": label, "score": round(conf, 2)})

            boxes = [b for b in boxes if _area(b) >= HARD_MIN_AREA]
            boxes = sorted(boxes, key=lambda b: b["score"], reverse=True)
            keep = []
            for b in boxes:
                if all(_iou(b, k) < NMS_IOU for k in keep):
                    keep.append(b)
            boxes = keep

            damage_found = len(boxes) > 0
            is_car = bool(damage_found)
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

    return {"is_car": True, "damage": False, "severity": "none", "location": [], "boxes": [], "version": "disabled"}

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
    print(f"⚠️ Vonage SMS disabled: {e}")

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
    """Admin login required."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return func(*args, **kwargs)
    return wrapper

def user_login_required(func):
    """General staff login required for main site."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("user_logged_in"):
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    return wrapper

# ===================== ROUTES =====================
@app.route("/", methods=["GET"])
@user_login_required
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

    return render_template_string(
        MAIN_PAGE_HTML,
        data=paged_data,
        runners=runners,
        keys_in_system=keys_in_system,
        page=page,
        total_pages=total_pages,
        message=message,
        message_type=message_type
    )

# ---------- Staff login ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        admins = load_json(ADMIN_FILE)
        if any(a.get("username") == username and a.get("password") == password for a in admins):
            session["user_logged_in"] = True
            return redirect(url_for("index"))
        error = "Invalid credentials. Please try again."

    return render_template_string(LOGIN_HTML, error=error, heading="Staff Login", subheading="Sign in to access the valet dashboard.")

@app.route("/logout")
def logout():
    session.pop("user_logged_in", None)
    session.pop("admin_logged_in", None)
    return redirect(url_for("login"))

@app.route("/checkin", methods=["POST"])
@user_login_required
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

    uploaded_files = []
    for field in ("damageImages", "capturedImages"):
        if field in request.files:
            for f in request.files.getlist(field):
                if f and f.filename:
                    uploaded_files.append(f)

    if len(uploaded_files) != 4:
        session["message"] = f"❌ Please upload exactly 4 photos (you provided {len(uploaded_files)})."
        session["message_type"] = "warning"
        return redirect("/")

    total_detections = 0
    union_locations = set()
    model_version_used = ""

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
@user_login_required
def checkout(ticket_id):
    data = load_json(DATA_FILE)
    now = datetime.datetime.now().strftime("%b %d, %Y %I:%M %p")
    runner = request.form.get("runnerName", "").strip()

    for t in data:
        if t.get("ticketID") == ticket_id and t.get("status") == "Checked-In":
            t["status"] = "Checked-Out"
            t["checkOutTime"] = now
            if runner:
                t["assignedRunner"] = runner
            save_json(DATA_FILE, data)
            send_sms(t.get("customerPhone", ""), f"Thank you! Ticket #{ticket_id} checked out.")
            session["message"] = f"✅ Ticket #{ticket_id} checked out" + (f" (Assigned to {runner})" if runner else "")
            session["message_type"] = "success"
            return redirect("/")

    session["message"] = f"❌ Ticket #{ticket_id} not found"
    session["message_type"] = "warning"
    return redirect("/")

@app.route("/checkout_manual", methods=["POST"])
@user_login_required
def checkout_manual():
    return checkout(request.form.get("ticket_id"))

@app.route("/delete/<ticket_id>", methods=["POST"])
@user_login_required
def delete_ticket(ticket_id):
    data = load_json(DATA_FILE)
    data = [t for t in data if t.get("ticketID") != ticket_id]
    save_json(DATA_FILE, data)
    session["message"] = f"🗑️ Ticket #{ticket_id} deleted"
    session["message_type"] = "warning"
    return redirect("/")

@app.route("/get_qr/<ticket_id>")
@user_login_required
def get_qr(ticket_id):
    """Return QR code image path for modal display"""
    qr_path = os.path.join(QRCODE_FOLDER, f"qr_{ticket_id}.png")
    if not os.path.exists(qr_path):
        generate_qr_code(ticket_id)
    return jsonify({"qr_url": f"/static/qrcodes/qr_{ticket_id}.png"})

@app.route("/vehicle_ready/<ticket_id>", methods=["POST"])
@user_login_required
def vehicle_ready(ticket_id):
    data = load_json(DATA_FILE)
    for t in data:
        if t.get("ticketID") == ticket_id:
            send_sms(t.get("customerPhone", ""), f"Your vehicle (#{ticket_id}) is ready!")
            session["message"] = "📱 SMS sent"
            break
    return redirect("/")

@app.route("/runner_clockin")
@user_login_required
def runner_clockin_page():
    runners = load_json(RUNNER_FILE)
    return render_template_string(RUNNER_PAGE_HTML, runners=runners)

@app.route("/clockin", methods=["POST"])
@user_login_required
def clockin():
    runners = load_json(RUNNER_FILE)
    name = request.form.get("runnerName", "").strip()
    if name and not any(r.get("name", "").lower() == name.lower() and "clockOutTime" not in r for r in runners):
        runners.append({"name": name, "clockInTime": datetime.datetime.now().strftime("%b %d, %Y %I:%M %p")})
        save_json(RUNNER_FILE, runners)
        session["message"] = f"✅ {name} clocked in"
    return redirect("/runner_clockin")

@app.route("/clockout", methods=["POST"])
@user_login_required
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

# ---------- Admin login ----------
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        admins = load_json(ADMIN_FILE)
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()
        if any(a.get("username") == u and a.get("password") == p for a in admins):
            session["admin_logged_in"] = True
            session["user_logged_in"] = True  # Also grant user access
            return redirect(url_for("admin_dashboard"))
        error = "Invalid credentials. Please try again."

    return render_template_string(LOGIN_HTML, error=error, heading="Admin Login", subheading="Sign in to manage announcements and operations.")

@app.route("/admin_logout")
@login_required
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))

@app.route("/admin")
@login_required
def admin_dashboard():
    return render_template_string(ADMIN_HTML)

@app.route("/admin_announcement", methods=["POST"])
@login_required
def admin_announcement():
    msg = request.form.get("message", "").strip()
    if msg:
        save_json(ANNOUNCEMENT_FILE, {"message": f"Admin: {msg}"})
    return redirect("/admin")

@app.route("/announcement_page")
@user_login_required
def announcement_page():
    return render_template_string(ANNOUNCEMENT_HTML)

@app.route("/announcement")
@user_login_required
def get_announcement():
    return jsonify(load_json(ANNOUNCEMENT_FILE))

@app.route("/shift_portal")
@user_login_required
def shift_portal():
    return render_template_string(SHIFT_PORTAL_HTML)

@app.route("/shifts")
@user_login_required
def get_shifts():
    return jsonify(load_json(SHIFTS_FILE))

@app.route("/add_shift", methods=["POST"])
@user_login_required
def add_shift():
    data = load_json(SHIFTS_FILE)
    payload = request.get_json(force=True)
    new_id = (max([s.get("id", 0) for s in data]) + 1) if data else 1
    data.append({"id": new_id, "day": payload.get("day", "").strip(), "time": payload.get("time", "").strip(), "assigned_to": None})
    save_json(SHIFTS_FILE, data)
    return jsonify({"ok": True})

@app.route("/pick_shift", methods=["POST"])
@user_login_required
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
@user_login_required
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
@user_login_required
def delete_shift():
    data = load_json(SHIFTS_FILE)
    sid = int(request.get_json(force=True).get("id"))
    data = [s for s in data if s.get("id") != sid]
    save_json(SHIFTS_FILE, data)
    return jsonify({"ok": True})

@app.route("/damage_check")
@user_login_required
def damage_check_page():
    return render_template_string(DAMAGE_CHECK_HTML)

@app.route("/get_damage_report/<ticket_id>")
@user_login_required
def get_damage_report(ticket_id):
    data = load_json(DATA_FILE)
    for t in data:
        if t.get("ticketID") == ticket_id:
            return jsonify({"success": True, "ticket": t})
    return jsonify({"success": False, "error": "Ticket not found"}), 404

@app.route("/incident_report/<ticket_id>")
@user_login_required
def incident_report(ticket_id):
    data = load_json(DATA_FILE)
    for t in data:
        if t.get("ticketID") == ticket_id:
            return render_template_string(INCIDENT_REPORT_HTML, ticket=t)
    return "Ticket not found", 404

@app.route("/calendar")
@user_login_required
def calendar_view():
    return render_template_string(CALENDAR_HTML)

@app.route("/<path:filename>")
def serve_file(filename):
    return send_from_directory(".", filename)

# ===================== HTML TEMPLATES =====================

# ===================== RUNNER PAGE HTML =====================
RUNNER_PAGE_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Runner Clock In/Out - Valet Operations</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 8px; }
        .header p { opacity: 0.9; font-size: 14px; }
        .content { padding: 30px; }
        .card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }
        .card h2 {
            font-size: 20px;
            color: #1e293b;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid #dc2626;
        }
        .form-group {
            margin-bottom: 16px;
        }
        label {
            display: block;
            font-weight: 600;
            color: #475569;
            margin-bottom: 8px;
            font-size: 14px;
        }
        input[type="text"] {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 15px;
            transition: all 0.2s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #dc2626;
            box-shadow: 0 0 0 3px rgba(220,38,38,0.1);
        }
        .btn {
            padding: 12px 28px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-block;
            text-decoration: none;
        }
        .btn-primary {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(220,38,38,0.3);
        }
        .btn-secondary {
            background: #64748b;
            color: white;
        }
        .btn-secondary:hover {
            background: #475569;
            transform: translateY(-2px);
        }
        .btn-back {
            background: white;
            color: #dc2626;
            border: 2px solid #dc2626;
            margin-bottom: 20px;
        }
        .btn-back:hover {
            background: #dc2626;
            color: white;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
        }
        th, td {
            padding: 14px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        th {
            background: #f1f5f9;
            font-weight: 600;
            color: #475569;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        td {
            color: #334155;
            font-size: 14px;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .status-active {
            background: #dcfce7;
            color: #166534;
        }
        .status-completed {
            background: #e0e7ff;
            color: #3730a3;
        }
        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: #94a3b8;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Runner Management</h1>
            <p>Clock In/Out • Shift Tracking</p>
        </div>
        <div class="content">
            <a href="/" class="btn btn-back">← Back to Dashboard</a>
            
            <div class="card">
                <h2>Clock In</h2>
                <form method="POST" action="/clockin">
                    <div class="form-group">
                        <label>Runner Name</label>
                        <input type="text" name="runnerName" required placeholder="Enter your name">
                    </div>
                    <button type="submit" class="btn btn-primary">Clock In</button>
                </form>
            </div>

            <div class="card">
                <h2>Clock Out</h2>
                <form method="POST" action="/clockout">
                    <div class="form-group">
                        <label>Runner Name</label>
                        <input type="text" name="runnerName" required placeholder="Enter your name">
                    </div>
                    <button type="submit" class="btn btn-secondary">Clock Out</button>
                </form>
            </div>

            <div class="card">
                <h2>Active Runners</h2>
                {% if runners %}
                <table>
                    <thead>
                        <tr>
                            <th>Runner Name</th>
                            <th>Clock In Time</th>
                            <th>Clock Out Time</th>
                            <th>Duration</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for r in runners %}
                        <tr>
                            <td><strong>{{ r.name }}</strong></td>
                            <td>{{ r.clockInTime }}</td>
                            <td>{{ r.get('clockOutTime', '—') }}</td>
                            <td>{{ r.get('duration', '—') }}</td>
                            <td>
                                {% if 'clockOutTime' in r %}
                                <span class="status-badge status-completed">Completed</span>
                                {% else %}
                                <span class="status-badge status-active">Active</span>
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% else %}
                <div class="empty-state">
                    <p>No runners clocked in yet.</p>
                </div>
                {% endif %}
            </div>
        </div>
    </div>
</body>
</html>
'''

# ===================== ANNOUNCEMENT HTML =====================
ANNOUNCEMENT_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Announcements - Valet Operations</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 8px; }
        .header p { opacity: 0.9; font-size: 14px; }
        .content { padding: 30px; }
        .announcement-box {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border-left: 5px solid #f59e0b;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(245,158,11,0.2);
        }
        .announcement-box h2 {
            font-size: 18px;
            color: #92400e;
            margin-bottom: 12px;
        }
        .announcement-box p {
            font-size: 16px;
            color: #78350f;
            line-height: 1.6;
        }
        .btn-back {
            background: white;
            color: #dc2626;
            border: 2px solid #dc2626;
            padding: 12px 28px;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            text-decoration: none;
            display: inline-block;
            transition: all 0.2s;
        }
        .btn-back:hover {
            background: #dc2626;
            color: white;
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>System Announcements</h1>
            <p>Important Updates & Information</p>
        </div>
        <div class="content">
            <a href="/" class="btn-back">← Back to Dashboard</a>
            
            <div class="announcement-box" id="announcement">
                <h2>Loading announcement...</h2>
            </div>
        </div>
    </div>
    <script>
        fetch('/announcement')
            .then(r => r.json())
            .then(data => {
                document.getElementById('announcement').innerHTML = 
                    '<h2>📢 Current Announcement</h2><p>' + data.message + '</p>';
            });
    </script>
</body>
</html>
'''

# ===================== SHIFT PORTAL HTML =====================
SHIFT_PORTAL_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shift Management - Valet Operations</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 8px; }
        .header p { opacity: 0.9; font-size: 14px; }
        .content { padding: 30px; }
        .btn-back {
            background: white;
            color: #dc2626;
            border: 2px solid #dc2626;
            padding: 12px 28px;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            text-decoration: none;
            display: inline-block;
            transition: all 0.2s;
            margin-bottom: 20px;
        }
        .btn-back:hover {
            background: #dc2626;
            color: white;
        }
        .card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }
        .card h2 {
            font-size: 20px;
            color: #1e293b;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid #dc2626;
        }
        .shift-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 16px;
            margin-top: 20px;
        }
        .shift-card {
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            transition: all 0.2s;
        }
        .shift-card:hover {
            border-color: #dc2626;
            box-shadow: 0 4px 12px rgba(220,38,38,0.1);
        }
        .shift-card h3 {
            font-size: 18px;
            color: #1e293b;
            margin-bottom: 8px;
        }
        .shift-card p {
            color: #64748b;
            font-size: 14px;
            margin-bottom: 12px;
        }
        .shift-status {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 12px;
        }
        .status-available {
            background: #dcfce7;
            color: #166534;
        }
        .status-assigned {
            background: #dbeafe;
            color: #1e40af;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            width: 100%;
        }
        .btn-primary {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(220,38,38,0.3);
        }
        .btn-danger {
            background: #ef4444;
            color: white;
            margin-top: 8px;
        }
        .btn-danger:hover {
            background: #dc2626;
        }
        .form-inline {
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
        }
        .form-inline input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 14px;
        }
        .form-inline input:focus {
            outline: none;
            border-color: #dc2626;
        }
        .empty-state {
            text-align: center;
            padding: 40px;
            color: #94a3b8;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Shift Management Portal</h1>
            <p>Pick, Drop & Manage Your Shifts</p>
        </div>
        <div class="content">
            <a href="/" class="btn-back">← Back to Dashboard</a>

            <div class="card">
                <h2>Add New Shift</h2>
                <div class="form-inline">
                    <input type="text" id="shiftDay" placeholder="Day (e.g., Monday)">
                    <input type="text" id="shiftTime" placeholder="Time (e.g., 9:00 AM - 5:00 PM)">
                    <button class="btn btn-primary" onclick="addShift()">Add Shift</button>
                </div>
            </div>

            <div class="card">
                <h2>Available Shifts</h2>
                <div class="shift-grid" id="shiftGrid">
                    <div class="empty-state">Loading shifts...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentUser = prompt("Enter your name to manage shifts:");
        if (!currentUser) currentUser = "Anonymous";

        function loadShifts() {
            fetch('/shifts')
                .then(r => r.json())
                .then(shifts => {
                    const grid = document.getElementById('shiftGrid');
                    if (shifts.length === 0) {
                        grid.innerHTML = '<div class="empty-state">No shifts available. Add a shift to get started.</div>';
                        return;
                    }
                    grid.innerHTML = shifts.map(s => `
                        <div class="shift-card">
                            <h3>${s.day}</h3>
                            <p>${s.time}</p>
                            ${s.assigned_to 
                                ? `<span class="shift-status status-assigned">Assigned to: ${s.assigned_to}</span>`
                                : `<span class="shift-status status-available">Available</span>`
                            }
                            ${!s.assigned_to 
                                ? `<button class="btn btn-primary" onclick="pickShift(${s.id})">Pick Shift</button>`
                                : s.assigned_to.toLowerCase() === currentUser.toLowerCase()
                                    ? `<button class="btn btn-danger" onclick="dropShift(${s.id})">Drop Shift</button>`
                                    : ''
                            }
                            <button class="btn btn-danger" onclick="deleteShift(${s.id})">Delete</button>
                        </div>
                    `).join('');
                });
        }

        function addShift() {
            const day = document.getElementById('shiftDay').value.trim();
            const time = document.getElementById('shiftTime').value.trim();
            if (!day || !time) return alert('Please fill in both fields');
            fetch('/add_shift', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({day, time})
            }).then(() => {
                document.getElementById('shiftDay').value = '';
                document.getElementById('shiftTime').value = '';
                loadShifts();
            });
        }

        function pickShift(id) {
            fetch('/pick_shift', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({id, user: currentUser})
            }).then(r => r.json()).then(data => {
                if (data.ok) loadShifts();
                else alert(data.error || 'Cannot pick this shift');
            });
        }

        function dropShift(id) {
            fetch('/drop_shift', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({id, user: currentUser})
            }).then(r => r.json()).then(data => {
                if (data.ok) loadShifts();
                else alert(data.error || 'Cannot drop this shift');
            });
        }

        function deleteShift(id) {
            if (!confirm('Delete this shift permanently?')) return;
            fetch('/delete_shift', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({id})
            }).then(() => loadShifts());
        }

        loadShifts();
    </script>
</body>
</html>
'''

# ===================== INCIDENT REPORT HTML =====================
INCIDENT_REPORT_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vehicle Incident Report - Ticket #{{ ticket.ticketID }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: white;
            padding: 20px;
            color: #000;
        }
        .report-container {
            max-width: 850px;
            margin: 0 auto;
            border: 3px solid #000;
            padding: 30px;
            background: white;
        }
        .header {
            text-align: center;
            border-bottom: 2px solid #000;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }
        .header h1 {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .form-row {
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
        }
        .form-field {
            flex: 1;
            display: flex;
            align-items: center;
        }
        .form-field label {
            font-weight: bold;
            margin-right: 8px;
            white-space: nowrap;
            font-size: 12px;
        }
        .form-field .underline {
            flex: 1;
            border-bottom: 1px solid #000;
            min-height: 20px;
            padding: 0 5px;
            font-size: 12px;
        }
        .section-title {
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 14px;
        }
        .repair-description {
            border: 1px solid #000;
            min-height: 150px;
            padding: 10px;
            margin-bottom: 20px;
            line-height: 1.8;
        }
        .damage-section {
            border: 2px solid #000;
            padding: 15px;
            margin: 20px 0;
        }
        .damage-key {
            margin-bottom: 15px;
        }
        .damage-key p {
            margin: 5px 0;
            font-size: 12px;
        }
        .checkbox-group {
            display: flex;
            gap: 30px;
            margin: 10px 0;
        }
        .checkbox-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .checkbox-item input {
            width: 18px;
            height: 18px;
        }
        .car-diagram {
            text-align: center;
            margin: 20px 0;
        }
        .car-diagram-container {
            display: inline-block;
            border: 2px solid #000;
            padding: 20px;
            background: white;
        }
        .car-view {
            margin: 15px 0;
            position: relative;
        }
        .car-outline {
            width: 100%;
            max-width: 400px;
            height: 80px;
            border: 2px solid #000;
            margin: 10px auto;
            position: relative;
            background: white;
        }
        .car-outline.top {
            border-radius: 60px 60px 20px 20px;
        }
        .car-outline.side {
            border-radius: 40px 10px 10px 40px;
        }
        .car-outline.bottom {
            border-radius: 20px 20px 60px 60px;
        }
        .car-label {
            text-align: center;
            font-size: 11px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .wheels {
            position: absolute;
            width: 15px;
            height: 30px;
            background: #000;
            border-radius: 4px;
        }
        .wheel-fl { top: 15px; left: -8px; }
        .wheel-fr { top: 15px; right: -8px; }
        .wheel-bl { bottom: 15px; left: -8px; }
        .wheel-br { bottom: 15px; right: -8px; }
        .terms {
            font-size: 10px;
            line-height: 1.4;
            margin: 20px 0;
            padding: 10px;
            border-top: 2px solid #000;
            border-bottom: 2px solid #000;
        }
        .terms-title {
            font-weight: bold;
            text-align: center;
            margin-bottom: 8px;
            font-size: 12px;
        }
        .signature-section {
            margin-top: 20px;
        }
        .signature-line {
            display: flex;
            align-items: center;
            margin: 15px 0;
        }
        .signature-line label {
            font-weight: bold;
            margin-right: 10px;
            white-space: nowrap;
            font-size: 12px;
        }
        .signature-line .line {
            flex: 1;
            border-bottom: 1px solid #000;
            min-height: 25px;
        }
        .footer {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
            font-size: 10px;
            padding-top: 10px;
            border-top: 1px solid #000;
        }
        .print-btn {
            background: #dc2626;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            margin: 20px auto;
            display: block;
        }
        .print-btn:hover {
            background: #b91c1c;
        }
        @media print {
            .print-btn { display: none; }
            body { padding: 0; }
            .report-container { border: none; }
        }
    </style>
</head>
<body>
    <button class="print-btn" onclick="window.print()">🖨️ Print Incident Report</button>
    
    <div class="report-container">
        <div class="header">
            <h1>Vehicle Inspection / Incident Report</h1>
        </div>

        <div class="form-row">
            <div class="form-field">
                <label>Date:</label>
                <div class="underline">{{ ticket.checkInTime }}</div>
            </div>
            <div class="form-field">
                <label>Tag Nr.:</label>
                <div class="underline"></div>
            </div>
            <div class="form-field">
                <label>Vin Nr.:</label>
                <div class="underline"></div>
            </div>
            <div class="form-field">
                <label>Mileage:</label>
                <div class="underline"></div>
            </div>
        </div>

        <div class="form-row">
            <div class="form-field">
                <label>Year:</label>
                <div class="underline"></div>
            </div>
            <div class="form-field" style="flex: 2;">
                <label>Make/Model:</label>
                <div class="underline">{{ ticket.carMake }} - {{ ticket.carColor }}</div>
            </div>
            <div class="form-field">
                <label>Time Promised:</label>
                <div class="underline"></div>
            </div>
        </div>

        <div class="form-row">
            <div class="form-field" style="flex: 2;">
                <label>Name:</label>
                <div class="underline">{{ ticket.customerName }}</div>
            </div>
            <div class="form-field" style="flex: 2;">
                <label>Address:</label>
                <div class="underline"></div>
            </div>
        </div>

        <div class="form-row">
            <div class="form-field">
                <label>City:</label>
                <div class="underline"></div>
            </div>
            <div class="form-field">
                <label>State:</label>
                <div class="underline"></div>
            </div>
            <div class="form-field">
                <label>Zip:</label>
                <div class="underline"></div>
            </div>
        </div>

        <div class="form-row">
            <div class="form-field">
                <label>Home Phone:</label>
                <div class="underline">{{ ticket.customerPhone }}</div>
            </div>
            <div class="form-field">
                <label>Business Phone:</label>
                <div class="underline"></div>
            </div>
            <div class="form-field">
                <label>Ext.:</label>
                <div class="underline"></div>
            </div>
        </div>

        <div class="form-row">
            <div class="form-field">
                <label>Email Address:</label>
                <div class="underline"></div>
            </div>
        </div>

        <div class="form-row">
            <div class="form-field">
                <label>License Plate:</label>
                <div class="underline">{{ ticket.licensePlate }}</div>
            </div>
            <div class="form-field">
                <label>Ticket Number:</label>
                <div class="underline">{{ ticket.ticketID }}</div>
            </div>
        </div>

        <div class="section-title">Incident / Repair Description:</div>
        <div class="repair-description">
            {{ ticket.notes or '' }}
            <br><br>
            {% if ticket.damageSummary.damage %}
            AI DAMAGE DETECTION RESULTS:<br>
            - Severity: {{ ticket.damageSummary.severity|upper }}<br>
            - Locations: {{ ', '.join(ticket.damageSummary.location)|upper if ticket.damageSummary.location else 'N/A' }}<br>
            - Total Detections: {{ ticket.damageSummary.totalDetections }}<br>
            {% endif %}
            <br>
            _________________________________________________________________________<br>
            _________________________________________________________________________<br>
            _________________________________________________________________________<br>
            _________________________________________________________________________<br>
        </div>

        <p style="font-size: 11px; text-align: center; margin-bottom: 15px;">
            <strong>(SEE REVERSE SIDE FOR ADDITIONAL INFORMATION)</strong>
        </p>

        <div class="damage-section">
            <div class="damage-key">
                <p><strong>X = Stone damage, scratch, dent or glass damage.</strong></p>
            </div>
            
            <div class="checkbox-group">
                <div class="checkbox-item">
                    <input type="checkbox" id="carDirty">
                    <label for="carDirty">Car Dirty</label>
                </div>
                <div class="checkbox-item">
                    <input type="checkbox" id="carClean">
                    <label for="carClean">Car Clean</label>
                </div>
            </div>

            <p style="font-size: 10px; margin: 10px 0;">
                If your car is dirty when it is dropped off, it's possible we won't see some damage.
            </p>

            <div class="car-diagram-container">
                <div class="car-view">
                    <div class="car-label">TOP VIEW</div>
                    <div class="car-outline top">
                        <div class="wheels wheel-fl"></div>
                        <div class="wheels wheel-fr"></div>
                    </div>
                </div>

                <div class="car-view">
                    <div class="car-label">SIDE VIEW</div>
                    <div class="car-outline side">
                        <div class="wheels wheel-fl"></div>
                        <div class="wheels wheel-bl"></div>
                    </div>
                </div>

                <div class="car-view">
                    <div class="car-label">BOTTOM VIEW</div>
                    <div class="car-outline bottom">
                        <div class="wheels wheel-bl"></div>
                        <div class="wheels wheel-br"></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="terms">
            <div class="terms-title">TERMS: STRICTLY CASH Unless Arrangements Made</div>
            <p>
                I hereby authorize the repair work specified on this form to be done along with the necessary material and/or parts, and hereby grant permission for anyone employed by this establishment to operate the vehicle herein described on streets, highways, or elsewhere for the purpose of testing and/or inspection. An express mechanic's lien is hereby acknowledged on above vehicle to secure the amount of repairs thereto. In case of fire, theft, or any other cause beyond your control for any delays caused by unavailability of parts or delays in parts shipments by the supplier or transporter, I hereby grant you and/or your insurance carrier subrogation rights against any carrier or others to seek damages for delays in shipment of parts. All disputes shall be governed by the laws of the jurisdiction in which the repair was performed. Disputes arising out of the interpretation or application of this agreement shall be subject to mediation and/or binding arbitration as provided by the rules and procedures of the American Arbitration Association. Customer acknowledges receipt of itemized invoice before authorizing repairs. Any legal action concerning any controversy, claim or dispute arising out of or relating to this contract shall be filed in the jurisdiction where the repairs were performed. Customer waives trial by jury. Vehicle owner shall be responsible for payment unless other written arrangements have been confirmed by management in advance. Declined services or repairs remain the customer's responsibility. All work performed carries a limited warranty as required by law. Storage fees may apply after 30 days. Management reserves the right to refuse service.
            </p>
        </div>

        <div class="signature-section">
            <div class="form-row">
                <div class="form-field">
                    <label>Minimum Diagnostic Charge $</label>
                    <div class="underline"></div>
                </div>
                <div class="form-field">
                    <label>Preliminary Estimate Total $</label>
                    <div class="underline"></div>
                </div>
            </div>

            <div class="signature-line">
                <label>Disclaimer and Damage noted:</label>
                <div class="line"></div>
            </div>

            <div class="signature-line">
                <label>Customer Signature:</label>
                <div class="line"></div>
            </div>

            <div class="signature-line">
                <label>Inspector/Valet Signature:</label>
                <div class="line"></div>
            </div>

            <div class="signature-line">
                <label>Date:</label>
                <div class="line"></div>
            </div>
        </div>

        <div class="footer">
            <div>VALET AUTO PRODUCTS</div>
            <div>Item #: 171</div>
            <div>MADE IN USA</div>
        </div>
    </div>

    <button class="print-btn" onclick="window.print()">🖨️ Print Incident Report</button>
</body>
</html>
'''

# ===================== DAMAGE CHECK HTML =====================
DAMAGE_CHECK_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Damage Check - Valet Operations</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 8px; }
        .header p { opacity: 0.9; font-size: 14px; }
        .content { padding: 30px; }
        .btn-back {
            background: white;
            color: #dc2626;
            border: 2px solid #dc2626;
            padding: 12px 28px;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            text-decoration: none;
            display: inline-block;
            transition: all 0.2s;
            margin-bottom: 20px;
        }
        .btn-back:hover {
            background: #dc2626;
            color: white;
        }
        .card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }
        .card h2 {
            font-size: 20px;
            color: #1e293b;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid #dc2626;
        }
        .search-box {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
        }
        .search-box input {
            flex: 1;
            padding: 14px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 15px;
            transition: all 0.2s;
        }
        .search-box input:focus {
            outline: none;
            border-color: #dc2626;
            box-shadow: 0 0 0 3px rgba(220,38,38,0.1);
        }
        .btn {
            padding: 14px 28px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(220,38,38,0.3);
        }
        .damage-report {
            display: none;
        }
        .damage-report.show {
            display: block;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .info-item {
            background: white;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
        .info-item label {
            display: block;
            font-size: 12px;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        .info-item .value {
            font-size: 16px;
            font-weight: 600;
            color: #1e293b;
        }
        .damage-badge {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 700;
            margin-top: 8px;
        }
        .damage-minor { background: #fef3c7; color: #92400e; }
        .damage-moderate { background: #fed7aa; color: #9a3412; }
        .damage-severe { background: #fecaca; color: #991b1b; }
        .damage-none { background: #dcfce7; color: #166534; }
        .photo-section {
            margin-top: 24px;
        }
        .photo-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }
        .photo-item {
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
            background: white;
        }
        .photo-item img {
            width: 100%;
            display: block;
        }
        .photo-label {
            padding: 8px;
            background: #f8fafc;
            font-size: 13px;
            font-weight: 600;
            color: #475569;
            text-align: center;
        }
        .detection-details {
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin-top: 16px;
        }
        .detection-item {
            padding: 12px;
            background: #f8fafc;
            border-left: 4px solid #dc2626;
            margin-bottom: 12px;
            border-radius: 4px;
        }
        .detection-item h4 {
            font-size: 14px;
            color: #1e293b;
            margin-bottom: 8px;
        }
        .detection-item p {
            font-size: 13px;
            color: #64748b;
            margin: 4px 0;
        }
        .alert {
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        .alert-error {
            background: #fee2e2;
            color: #991b1b;
            border-left: 4px solid #dc2626;
        }
        .report-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin-top: 20px;
        }
        .report-section h3 {
            font-size: 18px;
            color: #1e293b;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 2px solid #dc2626;
        }
        .report-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
        }
        .report-table th,
        .report-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        .report-table th {
            background: #f8fafc;
            font-weight: 600;
            color: #475569;
            font-size: 13px;
        }
        .report-table td {
            color: #334155;
            font-size: 14px;
        }
        .print-btn {
            margin-top: 20px;
            background: #475569;
            color: white;
            padding: 12px 28px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
        }
        .print-btn:hover {
            background: #334155;
        }
        @media print {
            .btn-back, .search-box, .print-btn, .nav-bar { display: none !important; }
            body { background: white; }
            .container { box-shadow: none; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Damage Check & Report</h1>
            <p>Comprehensive Vehicle Damage Assessment</p>
        </div>
        <div class="content">
            <a href="/" class="btn-back">← Back to Dashboard</a>
            
            <div class="card">
                <h2>Search Ticket</h2>
                <div class="search-box">
                    <input type="text" id="ticketSearch" placeholder="Enter Ticket Number (e.g., 0001)" autofocus>
                    <button class="btn btn-primary" onclick="searchTicket()">Search</button>
                </div>
                <div id="errorMessage"></div>
            </div>

            <div id="damageReport" class="damage-report">
                <div class="card">
                    <h2>Vehicle Information</h2>
                    <div class="info-grid">
                        <div class="info-item">
                            <label>Ticket Number</label>
                            <div class="value" id="reportTicketID">—</div>
                        </div>
                        <div class="info-item">
                            <label>License Plate</label>
                            <div class="value" id="reportLicense">—</div>
                        </div>
                        <div class="info-item">
                            <label>Customer Name</label>
                            <div class="value" id="reportCustomer">—</div>
                        </div>
                        <div class="info-item">
                            <label>Phone Number</label>
                            <div class="value" id="reportPhone">—</div>
                        </div>
                        <div class="info-item">
                            <label>Vehicle Make</label>
                            <div class="value" id="reportMake">—</div>
                        </div>
                        <div class="info-item">
                            <label>Vehicle Color</label>
                            <div class="value" id="reportColor">—</div>
                        </div>
                        <div class="info-item">
                            <label>Check-In Time</label>
                            <div class="value" id="reportCheckIn">—</div>
                        </div>
                        <div class="info-item">
                            <label>Status</label>
                            <div class="value" id="reportStatus">—</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h2>Damage Assessment Summary</h2>
                    <div class="info-grid">
                        <div class="info-item">
                            <label>Damage Status</label>
                            <div class="value" id="damageStatus">—</div>
                            <div id="damageBadge"></div>
                        </div>
                        <div class="info-item">
                            <label>Severity Level</label>
                            <div class="value" id="damageSeverity">—</div>
                        </div>
                        <div class="info-item">
                            <label>Damage Locations</label>
                            <div class="value" id="damageLocations">—</div>
                        </div>
                        <div class="info-item">
                            <label>Total Detections</label>
                            <div class="value" id="damageCount">—</div>
                        </div>
                    </div>

                    <div class="detection-details" id="detectionDetails" style="display: none;">
                        <h3 style="font-size: 16px; margin-bottom: 12px; color: #1e293b;">Detailed Detections</h3>
                        <div id="detectionList"></div>
                    </div>
                </div>

                <div class="card">
                    <h2>Damage Assessment Photos</h2>
                    <div class="photo-section">
                        <h3 style="font-size: 16px; margin-bottom: 12px; color: #475569;">Original Photos</h3>
                        <div class="photo-grid" id="originalPhotos"></div>
                    </div>
                    <div class="photo-section">
                        <h3 style="font-size: 16px; margin-bottom: 12px; margin-top: 24px; color: #475569;">AI-Annotated Photos (Damage Highlighted)</h3>
                        <div class="photo-grid" id="annotatedPhotos"></div>
                    </div>
                </div>

                <div class="report-section">
                    <h3>Damage Report Summary</h3>
                    <table class="report-table">
                        <tr>
                            <th>Report Generated</th>
                            <td id="reportDate">—</td>
                        </tr>
                        <tr>
                            <th>AI Model Used</th>
                            <td id="modelVersion">—</td>
                        </tr>
                        <tr>
                            <th>Inspector Notes</th>
                            <td id="inspectorNotes">—</td>
                        </tr>
                        <tr>
                            <th>Assigned Runner</th>
                            <td id="assignedRunner">—</td>
                        </tr>
                    </table>
                </div>

                <div style="text-align: center;">
                    <button class="print-btn" onclick="window.print()">🖨️ Print Report</button>
                    <button class="print-btn" onclick="openIncidentReport()" style="background: #dc2626; margin-left: 12px;">📋 Generate Incident Report</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentTicketId = null;

        function searchTicket() {
            const ticketId = document.getElementById('ticketSearch').value.trim();
            const errorDiv = document.getElementById('errorMessage');
            const reportDiv = document.getElementById('damageReport');
            
            if (!ticketId) {
                errorDiv.innerHTML = '<div class="alert alert-error">Please enter a ticket number</div>';
                reportDiv.classList.remove('show');
                return;
            }

            errorDiv.innerHTML = '';
            
            fetch(`/get_damage_report/${ticketId}`)
                .then(r => {
                    if (!r.ok) throw new Error('Ticket not found');
                    return r.json();
                })
                .then(data => {
                    if (data.success) {
                        displayReport(data.ticket);
                        reportDiv.classList.add('show');
                    } else {
                        throw new Error(data.error);
                    }
                })
                .catch(err => {
                    errorDiv.innerHTML = `<div class="alert alert-error">❌ ${err.message}</div>`;
                    reportDiv.classList.remove('show');
                });
        }

        function displayReport(ticket) {
            currentTicketId = ticket.ticketID; // Store for incident report
            
            // Basic info
            document.getElementById('reportTicketID').textContent = ticket.ticketID;
            document.getElementById('reportLicense').textContent = ticket.licensePlate;
            document.getElementById('reportCustomer').textContent = ticket.customerName;
            document.getElementById('reportPhone').textContent = ticket.customerPhone;
            document.getElementById('reportMake').textContent = ticket.carMake;
            document.getElementById('reportColor').textContent = ticket.carColor;
            document.getElementById('reportCheckIn').textContent = ticket.checkInTime;
            document.getElementById('reportStatus').textContent = ticket.status;

            // Damage summary
            const summary = ticket.damageSummary || {};
            document.getElementById('damageStatus').textContent = summary.damage ? 'Damage Detected' : 'No Damage';
            document.getElementById('damageSeverity').textContent = (summary.severity || 'none').toUpperCase();
            document.getElementById('damageLocations').textContent = 
                summary.location && summary.location.length > 0 
                    ? summary.location.join(', ').toUpperCase() 
                    : 'N/A';
            document.getElementById('damageCount').textContent = summary.totalDetections || 0;

            // Damage badge
            const badgeDiv = document.getElementById('damageBadge');
            if (summary.damage) {
                badgeDiv.innerHTML = `<span class="damage-badge damage-${summary.severity}">${summary.severity.toUpperCase()}</span>`;
            } else {
                badgeDiv.innerHTML = '<span class="damage-badge damage-none">NO DAMAGE</span>';
            }

            // Detection details
            const detections = ticket.damageDetections || [];
            const detailsDiv = document.getElementById('detectionDetails');
            const listDiv = document.getElementById('detectionList');
            
            if (detections.length > 0 && detections.some(d => d.boxes && d.boxes.length > 0)) {
                detailsDiv.style.display = 'block';
                listDiv.innerHTML = detections.map((det, idx) => {
                    if (!det.boxes || det.boxes.length === 0) return '';
                    return `
                        <div class="detection-item">
                            <h4>Photo ${idx + 1} - ${det.boxes.length} detection(s)</h4>
                            ${det.boxes.map(box => `
                                <p>• ${box.label} (Confidence: ${(box.score * 100).toFixed(1)}%)</p>
                            `).join('')}
                            ${det.location && det.location.length > 0 
                                ? `<p><strong>Location:</strong> ${det.location.join(', ')}</p>` 
                                : ''}
                        </div>
                    `;
                }).join('');
            } else {
                detailsDiv.style.display = 'none';
            }

            // Photos
            const originalDiv = document.getElementById('originalPhotos');
            const annotatedDiv = document.getElementById('annotatedPhotos');
            
            originalDiv.innerHTML = (ticket.damageImages || []).map((img, idx) => `
                <div class="photo-item">
                    <img src="${img}" alt="Original Photo ${idx + 1}">
                    <div class="photo-label">Photo ${idx + 1}</div>
                </div>
            `).join('');

            annotatedDiv.innerHTML = (ticket.damageAnnotated || []).map((img, idx) => `
                <div class="photo-item">
                    <img src="${img}" alt="Annotated Photo ${idx + 1}">
                    <div class="photo-label">Photo ${idx + 1} - Annotated</div>
                </div>
            `).join('');

            // Report details
            document.getElementById('reportDate').textContent = new Date().toLocaleString();
            document.getElementById('modelVersion').textContent = summary.modelVersion || 'N/A';
            document.getElementById('inspectorNotes').textContent = ticket.notes || 'No notes provided';
            document.getElementById('assignedRunner').textContent = ticket.assignedRunner || 'Not assigned';
        }

        // Allow Enter key to search
        document.getElementById('ticketSearch').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchTicket();
            }
        });

        function openIncidentReport() {
            if (currentTicketId) {
                window.open(`/incident_report/${currentTicketId}`, '_blank');
            } else {
                alert('Please search for a ticket first');
            }
        }
    </script>
</body>
</html>
'''

# ===================== CALENDAR HTML =====================
CALENDAR_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shift Calendar - Valet Operations</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 8px; }
        .header p { opacity: 0.9; font-size: 14px; }
        .content { padding: 30px; }
        .btn-back {
            background: white;
            color: #dc2626;
            border: 2px solid #dc2626;
            padding: 12px 28px;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            text-decoration: none;
            display: inline-block;
            transition: all 0.2s;
            margin-bottom: 20px;
        }
        .btn-back:hover {
            background: #dc2626;
            color: white;
        }
        .calendar-grid {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 2px;
            background: #e2e8f0;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
        }
        .calendar-header {
            background: #dc2626;
            color: white;
            padding: 16px;
            text-align: center;
            font-weight: 600;
            font-size: 14px;
        }
        .calendar-day {
            background: white;
            padding: 16px;
            min-height: 120px;
            position: relative;
        }
        .calendar-day-num {
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 8px;
        }
        .calendar-day.today {
            background: #fef3c7;
            border: 2px solid #f59e0b;
        }
        .shift-item {
            background: #dbeafe;
            color: #1e40af;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 12px;
            margin-bottom: 4px;
            font-weight: 500;
        }
        .info-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Shift Calendar</h1>
            <p>Visual Schedule Overview</p>
        </div>
        <div class="content">
            <a href="/" class="btn-back">← Back to Dashboard</a>
            
            <div class="info-box">
                <strong>Current Month:</strong> <span id="currentMonth"></span>
            </div>

            <div class="calendar-grid" id="calendar">
                <div class="calendar-header">Sunday</div>
                <div class="calendar-header">Monday</div>
                <div class="calendar-header">Tuesday</div>
                <div class="calendar-header">Wednesday</div>
                <div class="calendar-header">Thursday</div>
                <div class="calendar-header">Friday</div>
                <div class="calendar-header">Saturday</div>
            </div>
        </div>
    </div>

    <script>
        const monthNames = ["January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"];
        
        const now = new Date();
        document.getElementById('currentMonth').textContent = 
            `${monthNames[now.getMonth()]} ${now.getFullYear()}`;

        function generateCalendar() {
            const year = now.getFullYear();
            const month = now.getMonth();
            const firstDay = new Date(year, month, 1).getDay();
            const daysInMonth = new Date(year, month + 1, 0).getDate();
            const today = now.getDate();

            const calendar = document.getElementById('calendar');
            
            for (let i = 0; i < firstDay; i++) {
                const cell = document.createElement('div');
                cell.className = 'calendar-day';
                calendar.appendChild(cell);
            }

            for (let day = 1; day <= daysInMonth; day++) {
                const cell = document.createElement('div');
                cell.className = 'calendar-day' + (day === today ? ' today' : '');
                cell.innerHTML = `<div class="calendar-day-num">${day}</div>`;
                calendar.appendChild(cell);
            }
        }

        fetch('/shifts')
            .then(r => r.json())
            .then(shifts => {
                generateCalendar();
            });
    </script>
</body>
</html>
'''

# ===================== ADMIN DASHBOARD HTML =====================
ADMIN_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard - Valet Operations</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
            padding: 30px;
            border-radius: 16px 16px 0 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        .header h1 {
            font-size: 32px;
            margin-bottom: 8px;
        }
        .header p {
            opacity: 0.9;
            font-size: 14px;
        }
        .nav-buttons {
            background: white;
            padding: 20px 30px;
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            border-bottom: 2px solid #e2e8f0;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            text-decoration: none;
            display: inline-block;
        }
        .btn-primary {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(220,38,38,0.3);
        }
        .btn-logout {
            background: #ef4444;
            color: white;
            margin-left: auto;
        }
        .btn-logout:hover {
            background: #dc2626;
        }
        .tabs {
            background: white;
            padding: 0;
            display: flex;
            border-bottom: 2px solid #e2e8f0;
        }
        .tab {
            padding: 16px 32px;
            background: none;
            border: none;
            font-size: 15px;
            font-weight: 600;
            color: #64748b;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
        }
        .tab:hover {
            color: #dc2626;
            background: #f8fafc;
        }
        .tab.active {
            color: #dc2626;
            border-bottom-color: #dc2626;
        }
        .tab-content {
            background: white;
            padding: 30px;
            border-radius: 0 0 16px 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }
        .card h2 {
            font-size: 20px;
            color: #1e293b;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 2px solid #dc2626;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            font-weight: 600;
            color: #475569;
            margin-bottom: 8px;
            font-size: 14px;
        }
        textarea {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 15px;
            font-family: inherit;
            transition: all 0.2s;
            min-height: 120px;
            resize: vertical;
        }
        textarea:focus {
            outline: none;
            border-color: #dc2626;
            box-shadow: 0 0 0 3px rgba(220,38,38,0.1);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(220,38,38,0.2);
        }
        .stat-card h3 {
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-card .number {
            font-size: 36px;
            font-weight: 700;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
        }
        th, td {
            padding: 14px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        th {
            background: #f1f5f9;
            font-weight: 600;
            color: #475569;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        td {
            color: #334155;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Admin Dashboard</h1>
            <p>Valet Operations Management & Control Panel</p>
        </div>

        <div class="nav-buttons">
            <a href="/" class="btn btn-primary">Main Dashboard</a>
            <a href="/admin_logout" class="btn btn-logout">Logout</a>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('announcements')">Announcements</button>
            <button class="tab" onclick="switchTab('analytics')">Analytics</button>
            <button class="tab" onclick="switchTab('settings')">Settings</button>
        </div>

        <div id="announcements" class="tab-content active">
            <div class="card">
                <h2>Post System Announcement</h2>
                <form method="POST" action="/admin_announcement">
                    <div class="form-group">
                        <label>Announcement Message</label>
                        <textarea name="message" placeholder="Enter your announcement message for all staff members..." required></textarea>
                    </div>
                    <button type="submit" class="btn btn-primary">Publish Announcement</button>
                </form>
            </div>

            <div class="card">
                <h2>Current Announcement</h2>
                <div id="currentAnnouncement" style="padding: 16px; background: #fef3c7; border-radius: 8px; color: #78350f;">
                    Loading...
                </div>
            </div>
        </div>

        <div id="analytics" class="tab-content">
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Tickets</h3>
                    <div class="number" id="totalTickets">0</div>
                </div>
                <div class="stat-card">
                    <h3>Active Tickets</h3>
                    <div class="number" id="activeTickets">0</div>
                </div>
                <div class="stat-card">
                    <h3>Active Runners</h3>
                    <div class="number" id="activeRunners">0</div>
                </div>
                <div class="stat-card">
                    <h3>Damage Reports</h3>
                    <div class="number" id="damageReports">0</div>
                </div>
            </div>

            <div class="card">
                <h2>Recent Activity</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Ticket ID</th>
                            <th>Customer</th>
                            <th>License Plate</th>
                            <th>Status</th>
                            <th>Check-In Time</th>
                        </tr>
                    </thead>
                    <tbody id="recentActivity">
                        <tr><td colspan="5" style="text-align: center; color: #94a3b8;">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div id="settings" class="tab-content">
            <div class="card">
                <h2>System Settings</h2>
                <p style="color: #64748b; margin-bottom: 20px;">System configuration and status information.</p>
                
                <div class="form-group">
                    <label>System Status</label>
                    <input type="text" value="Operational" disabled style="width: 100%; padding: 12px; border: 2px solid #e2e8f0; border-radius: 8px;">
                </div>
                
                <div class="form-group">
                    <label>SMS Notifications</label>
                    <input type="text" value="Configured" disabled style="width: 100%; padding: 12px; border: 2px solid #e2e8f0; border-radius: 8px;">
                </div>
                
                <div class="form-group">
                    <label>AI Damage Detection</label>
                    <input type="text" value="Enabled (Roboflow)" disabled style="width: 100%; padding: 12px; border: 2px solid #e2e8f0; border-radius: 8px;">
                </div>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }

        fetch('/announcement')
            .then(r => r.json())
            .then(data => {
                document.getElementById('currentAnnouncement').textContent = data.message;
            });

        fetch('/data.json')
            .then(r => r.json())
            .then(data => {
                document.getElementById('totalTickets').textContent = data.length;
                document.getElementById('activeTickets').textContent = 
                    data.filter(t => t.status === 'Checked-In').length;
                document.getElementById('damageReports').textContent = 
                    data.filter(t => t.damageSummary?.damage).length;
                
                const tbody = document.getElementById('recentActivity');
                const recent = data.slice(-10).reverse();
                tbody.innerHTML = recent.map(t => `
                    <tr>
                        <td><strong>#${t.ticketID}</strong></td>
                        <td>${t.customerName}</td>
                        <td>${t.licensePlate}</td>
                        <td>${t.status}</td>
                        <td>${t.checkInTime}</td>
                    </tr>
                `).join('');
            });

        fetch('/runners.json')
            .then(r => r.json())
            .then(runners => {
                const active = runners.filter(r => !r.clockOutTime).length;
                document.getElementById('activeRunners').textContent = active;
            });
    </script>
</body>
</html>
'''

# ===================== SHARED LOGIN HTML =====================
LOGIN_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ heading }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .login-container {
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
            width: 100%;
            max-width: 450px;
        }
        .header {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 28px;
            margin-bottom: 8px;
        }
        .header p {
            opacity: 0.9;
            font-size: 14px;
        }
        .form-container {
            padding: 40px 30px;
        }
        .form-group {
            margin-bottom: 24px;
        }
        label {
            display: block;
            font-weight: 600;
            color: #475569;
            margin-bottom: 8px;
            font-size: 14px;
        }
        input {
            width: 100%;
            padding: 14px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 15px;
            transition: all 0.2s;
        }
        input:focus {
            outline: none;
            border-color: #dc2626;
            box-shadow: 0 0 0 3px rgba(220,38,38,0.1);
        }
        {% if error %}
        .error-message {
            background: #fee2e2;
            color: #991b1b;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #dc2626;
            font-size: 14px;
        }
        {% endif %}
        .btn-submit {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(220,38,38,0.3);
        }
        .footer {
            text-align: center;
            padding: 20px 30px;
            background: #f8fafc;
            border-top: 1px solid #e2e8f0;
        }
        .footer p {
            font-size: 13px;
            color: #64748b;
        }
        .footer a {
            color: #dc2626;
            text-decoration: none;
            font-weight: 600;
            display: inline-block;
            margin-top: 8px;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="header">
            <h1>{{ heading }}</h1>
            <p>{{ subheading }}</p>
        </div>
        <div class="form-container">
            {% if error %}
            <div class="error-message">
                {{ error }}
            </div>
            {% endif %}
            <form method="POST">
                <div class="form-group">
                    <label>Username</label>
                    <input type="text" name="username" required autofocus>
                </div>
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" name="password" required>
                </div>
                <button type="submit" class="btn-submit">Sign In</button>
            </form>
        </div>
        <div class="footer">
            <p>Default credentials: admin / valet123</p>
            <a href="/admin_login">Admin Login</a>
        </div>
    </div>
</body>
</html>
'''

# ===================== MAIN PAGE HTML =====================
MAIN_PAGE_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Valet Operations System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            min-height: 100vh;
            color: #1e293b;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
            padding: 30px;
            border-radius: 16px 16px 0 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        .header h1 {
            font-size: 32px;
            margin-bottom: 6px;
        }
        .header p {
            opacity: 0.9;
            font-size: 14px;
        }
        .nav-bar {
            background: white;
            padding: 16px 30px;
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
            border-bottom: 2px solid #e2e8f0;
        }
        .nav-btn {
            padding: 10px 20px;
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .nav-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(220,38,38,0.3);
        }
        .nav-btn.admin {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        }
        .nav-btn.admin:hover {
            box-shadow: 0 4px 12px rgba(99,102,241,0.3);
        }
        .logout-btn {
            background: #ef4444;
            margin-left: auto;
        }
        .logout-btn:hover {
            background: #dc2626;
            box-shadow: 0 4px 12px rgba(239,68,68,0.3);
        }
        .content {
            background: white;
            padding: 30px;
            border-radius: 0 0 16px 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .message {
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 24px;
            font-weight: 500;
        }
        .message.success {
            background: #dcfce7;
            color: #166534;
            border-left: 4px solid #16a34a;
        }
        .message.warning {
            background: #fef3c7;
            color: #92400e;
            border-left: 4px solid #f59e0b;
        }
        .stats-bar {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
            border: 2px solid #cbd5e1;
        }
        .stats-bar p {
            font-size: 18px;
            font-weight: 600;
            color: #334155;
        }
        .stats-bar span {
            color: #dc2626;
            font-size: 24px;
        }
        .section-title {
            font-size: 24px;
            color: #1e293b;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 3px solid #dc2626;
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-weight: 600;
            color: #475569;
            margin-bottom: 6px;
            font-size: 14px;
        }
        input, textarea, select {
            padding: 12px 14px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 14px;
            font-family: inherit;
            transition: all 0.2s;
        }
        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #dc2626;
            box-shadow: 0 0 0 3px rgba(220,38,38,0.1);
        }
        .file-upload-box {
            grid-column: 1 / -1;
            background: #f8fafc;
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            padding: 30px;
            text-align: center;
        }
        .file-upload-box h3 {
            color: #475569;
            margin-bottom: 12px;
            font-size: 16px;
        }
        .file-upload-box input {
            border: none;
            background: white;
            padding: 16px;
            border-radius: 8px;
        }
        .btn-submit {
            grid-column: 1 / -1;
            padding: 16px;
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(220,38,38,0.3);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 14px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        th {
            background: #f1f5f9;
            font-weight: 600;
            color: #475569;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .damage-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
        }
        .damage-badge:hover {
            transform: scale(1.05);
        }
        .damage-minor { background: #fef3c7; color: #92400e; }
        .damage-moderate { background: #fed7aa; color: #9a3412; }
        .damage-severe { background: #fecaca; color: #991b1b; }
        .damage-none { background: #dcfce7; color: #166534; }
        .btn-action {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            margin: 2px;
            transition: all 0.2s;
            text-decoration: none;
            color: white;
        }
        .btn-primary {
            background: #dc2626;
            color: white;
        }
        .btn-primary:hover {
            background: #b91c1c;
            transform: translateY(-1px);
        }
        .btn-danger {
            background: #dc2626;
            color: white;
        }
        .btn-danger:hover {
            background: #b91c1c;
        }
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        .modal.show {
            display: flex;
        }
        .modal-content {
            background: white;
            padding: 30px;
            max-width: 600px;
            border-radius: 16px;
            max-height: 90vh;
            overflow-y: auto;
            position: relative;
        }
        .modal-close {
            position: absolute;
            top: 15px;
            right: 20px;
            font-size: 32px;
            font-weight: bold;
            cursor: pointer;
            color: #64748b;
            background: none;
            border: none;
        }
        .modal-close:hover {
            color: #dc2626;
        }
        .photo-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 20px;
        }
        .photo-grid img {
            width: 100%;
            border-radius: 8px;
            border: 2px solid #e2e8f0;
        }
        .qr-display {
            text-align: center;
            padding: 20px;
        }
        .qr-display img {
            max-width: 300px;
            border: 2px solid #e2e8f0;
            padding: 20px;
            border-radius: 12px;
            background: white;
        }
        .pagination {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-top: 30px;
            flex-wrap: wrap;
        }
        .pagination a {
            padding: 10px 16px;
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            text-decoration: none;
            color: #334155;
            font-weight: 600;
            transition: all 0.2s;
        }
        .pagination a:hover {
            border-color: #dc2626;
            color: #dc2626;
        }
        .pagination a.active {
            background: #dc2626;
            color: white;
            border-color: #dc2626;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #94a3b8;
        }
        .empty-state h2 {
            font-size: 24px;
            margin-bottom: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Valet Operations System</h1>
            <p>Red Ramp Valet • Professional Service</p>
        </div>

        <div class="nav-bar">
            <a href="/admin_login" class="nav-btn admin">Administrator</a>
            <a href="/runner_clockin" class="nav-btn">Runner Management</a>
            <a href="/shift_portal" class="nav-btn">Shift Management</a>
            <a href="/calendar" class="nav-btn">Shift Calendar</a>
            <a href="/damage_check" class="nav-btn">Damage Check</a>
            <a href="/announcement_page" class="nav-btn">Announcements</a>
            <a href="/logout" class="nav-btn logout-btn">Logout</a>
        </div>

        <div class="content">
            {% if message %}
            <div class="message {{ message_type }}">
                {{message}}
            </div>
            {% endif %}

            <div class="stats-bar">
                <p>🔑 Keys in System: <span>{{keys_in_system}}</span> Live inventory</p>
            </div>

            <h2 class="section-title">Vehicle Check-In</h2>
            <form method="POST" action="/checkin" enctype="multipart/form-data">
                <div class="form-grid">
                    <div class="form-group">
                        <label>License Plate</label>
                        <input type="text" name="licensePlate" required>
                    </div>
                    <div class="form-group">
                        <label>Customer Name</label>
                        <input type="text" name="customerName" required>
                    </div>
                    <div class="form-group">
                        <label>Customer Phone</label>
                        <input type="text" name="customerPhone" required>
                    </div>
                    <div class="form-group">
                        <label>Car Make</label>
                        <input type="text" name="carMake" required>
                    </div>
                    <div class="form-group">
                        <label>Car Color</label>
                        <input type="text" name="carColor" required>
                    </div>
                    <div class="form-group">
                        <label>Notes</label>
                        <textarea name="notes" rows="1"></textarea>
                    </div>
                    <div class="file-upload-box">
                        <h3>Damage assessment photos (exactly four required)</h3>
                        <input type="file" name="damageImages" accept="image/*" multiple required>
                    </div>
                    <button type="submit" class="btn-submit">Complete Check-In</button>
                </div>
            </form>

            <h2 class="section-title" style="margin-top: 50px;">Vehicle Checkout</h2>
            <form method="POST" action="/checkout_manual" style="margin-bottom: 50px;">
                <div class="form-grid">
                    <div class="form-group">
                        <label>Ticket Number</label>
                        <input type="text" name="ticket_id" required>
                    </div>
                    <div class="form-group">
                        <label>Assign Runner (Optional)</label>
                        <select name="runnerName">
                            <option value="">-- Select Runner --</option>
                            {% for r in runners %}
                            <option>{{r.name}}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <button type="submit" class="btn-submit">Process Checkout</button>
                </div>
            </form>

            <h2 class="section-title">Active Tickets</h2>
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
                        <td><strong>{{t.ticketID}}</strong></td>
                        <td>{{t.licensePlate}}</td>
                        <td>{{t.customerName}}</td>
                        <td>{{t.customerPhone}}</td>
                        <td>{{t.carMake}}</td>
                        <td>{{t.carColor}}</td>
                        <td>{{t.status}}</td>
                        <td>{{t.checkInTime}}</td>
                        <td>{{t.assignedRunner or 'Unassigned'}}</td>
                        <td>
                            {% set ds = t.get('damageSummary', {}) %}
                            {% if ds.get('damage') %}
                            <span class="damage-badge damage-{{ds.get('severity','none')}}" onclick="showDamageModal('{{t.ticketID}}', {{t.damageAnnotated|tojson}})">
                                {{ds.get('severity','none')|capitalize}}
                            </span>
                            {% if ds.get('location') %}
                            <br><small style="color: #64748b;">{{', '.join(ds.get('location'))}}</small>
                            {% endif %}
                            {% else %}
                            <span class="damage-badge damage-none">None Detected</span>
                            {% endif %}
                        </td>
                        <td>
                            {% if t.status == 'Checked-In' %}
                            <button class="btn-action btn-primary" onclick="showQRModal('{{t.ticketID}}')">View QR</button>
                            {% else %}
                            —
                            {% endif %}
                        </td>
                        <td>
                            {% if t.status == 'Checked-In' %}
                            <form method="POST" action="/vehicle_ready/{{t.ticketID}}" style="display: inline-block;">
                                <button type="submit" class="btn-action btn-primary">Ready</button>
                            </form>
                            <br>
                            <form method="POST" action="/checkout/{{t.ticketID}}" style="display: inline-block; margin: 4px 0;">
                                <select name="runnerName" style="padding: 6px; border-radius: 6px; border: 1px solid #e2e8f0; margin-right: 4px;">
                                    <option value="">Select Runner</option>
                                    {% for r in runners %}
                                    <option>{{r.name}}</option>
                                    {% endfor %}
                                </select>
                                <button type="submit" class="btn-action btn-primary">Checkout</button>
                            </form>
                            {% endif %}
                            <br>
                            <form method="POST" action="/delete/{{t.ticketID}}" style="display: inline-block;">
                                <button type="submit" class="btn-action btn-danger" onclick="return confirm('Delete this ticket?')">Delete</button>
                            </form>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>

            <div class="pagination">
                {% if page > 1 %}
                <a href="?page={{page-1}}">Previous</a>
                {% endif %}
                {% for i in range(1, total_pages + 1) %}
                    {% if i == page %}
                    <a href="?page={{i}}" class="active">{{i}}</a>
                    {% else %}
                    <a href="?page={{i}}">{{i}}</a>
                    {% endif %}
                {% endfor %}
                {% if page < total_pages %}
                <a href="?page={{page+1}}">Next</a>
                {% endif %}
            </div>
            {% else %}
            <div class="empty-state">
                <h2>No tickets in the system.</h2>
                <p>Check in a vehicle to begin operations.</p>
            </div>
            {% endif %}
        </div>
    </div>

    <!-- Damage Modal -->
    <div id="damageModal" class="modal">
        <div class="modal-content">
            <button class="modal-close" onclick="closeDamageModal()">&times;</button>
            <h2 style="margin-bottom: 20px;">Damage Assessment Photos</h2>
            <div class="photo-grid" id="damagePhotos"></div>
        </div>
    </div>

    <!-- QR Code Modal -->
    <div id="qrModal" class="modal">
        <div class="modal-content">
            <button class="modal-close" onclick="closeQRModal()">&times;</button>
            <h2 style="margin-bottom: 20px; text-align: center;">Ticket QR Code</h2>
            <div class="qr-display" id="qrDisplay"></div>
            <div style="text-align: center; margin-top: 20px;">
                <button onclick="window.print()" class="btn-action btn-primary">Print</button>
            </div>
        </div>
    </div>

    <script>
        function showDamageModal(ticketID, photos) {
            const modal = document.getElementById('damageModal');
            const grid = document.getElementById('damagePhotos');
            grid.innerHTML = '';
            photos.forEach(p => {
                const img = document.createElement('img');
                img.src = p;
                img.alt = 'Damage photo';
                grid.appendChild(img);
            });
            modal.classList.add('show');
        }

        function closeDamageModal() {
            document.getElementById('damageModal').classList.remove('show');
        }

        function showQRModal(ticketID) {
            const modal = document.getElementById('qrModal');
            const display = document.getElementById('qrDisplay');
            
            fetch(`/get_qr/${ticketID}`)
                .then(r => r.json())
                .then(data => {
                    display.innerHTML = `
                        <h3>Ticket #${ticketID}</h3>
                        <img src="${data.qr_url}" alt="QR Code">
                    `;
                    modal.classList.add('show');
                });
        }

        function closeQRModal() {
            document.getElementById('qrModal').classList.remove('show');
        }

        // Close modals when clicking outside
        window.onclick = function(event) {
            const damageModal = document.getElementById('damageModal');
            const qrModal = document.getElementById('qrModal');
            if (event.target == damageModal) {
                closeDamageModal();
            }
            if (event.target == qrModal) {
                closeQRModal();
            }
        }
    </script>
</body>
</html>
'''

# ===================== MAIN =====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚗 VALET OPERATIONS MANAGEMENT SYSTEM")
    print("   Red Valet Branding + Blue Background")
    print("="*70)
    port = int(os.getenv("PORT", 5050))
    print(f"🌐 Main URL: http://127.0.0.1:{port}")
    print(f"👤 Staff Login: http://127.0.0.1:{port}/login")
    print(f"👨‍💼 Admin Login: http://127.0.0.1:{port}/admin_login")
    print(f"🔑 Credentials: admin / valet123")
    print(f"🎨 Red branding (#dc2626) + Blue background (#1e3a8a)")
    print("="*70 + "\n")
    app.run(host="0.0.0.0", port=port, debug=True)
