import os, json, datetime, qrcode
from functools import wraps

from flask import (
    Flask, render_template_string, request, redirect, jsonify,
    session, send_from_directory, url_for
)
from apscheduler.schedulers.background import BackgroundScheduler

# ---------- Optional: Vonage SMS ----------
SMS_ENABLED = False
VONAGE_API_KEY = "57915c86"
VONAGE_API_SECRET = "Valet123"
VONAGE_PHONE_NUMBER = "+13073963461"
try:
    import vonage
    vonage_client = vonage.Client(key=VONAGE_API_KEY, secret=VONAGE_API_SECRET)
    sms = vonage.Sms(vonage_client)
    SMS_ENABLED = True
    print("✅ Vonage SMS enabled")
except Exception as e:
    print(f"⚠️ Vonage SMS disabled: {e}")

# ---------- App ----------
app = Flask(__name__)
app.secret_key = "valet_secret_key"

# ---------- Folders ----------
STATIC_DIR = "static"
UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
QRCODE_FOLDER = os.path.join(STATIC_DIR, "qrcodes")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(QRCODE_FOLDER, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
app.config.update(UPLOAD_FOLDER=UPLOAD_FOLDER, QRCODE_FOLDER=QRCODE_FOLDER)

# ---------- Data Files ----------
DATA_FILE = "data.json"      # tickets (list)
RUNNER_FILE = "runners.json" # runner history (list)
SHIFTS_FILE = "shifts.json"  # shifts (list of {id, day, time, assigned_to})
ADMIN_FILE = "admins.json"   # admins (list of {username,password})
ANNOUNCEMENT_FILE = "announcement.json"

for fpath, default in [
    (DATA_FILE, []),
    (RUNNER_FILE, []),
    (SHIFTS_FILE, []),
    (ADMIN_FILE, [{"username": "admin", "password": "valet123"}]),
    (ANNOUNCEMENT_FILE, {"message": "Welcome to the Valet Operations System!"}),
]:
    if not os.path.exists(fpath):
        with open(fpath, "w") as f:
            json.dump(default, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ---------- Admin / Auth ----------
def load_admins():
    return load_json(ADMIN_FILE)

def check_admin_credentials(username, password):
    return any(a["username"] == username and a["password"] == password for a in load_admins())

def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return func(*args, **kwargs)
    return wrapper

# ---------- Announcement & Scheduler ----------
def load_announcement():
    return load_json(ANNOUNCEMENT_FILE)

def save_announcement(data):
    save_json(ANNOUNCEMENT_FILE, data)

def update_announcement():
    message = f"Weekly Valet Update – {datetime.datetime.now().strftime('%A, %B %d, %Y')}"
    save_announcement({"message": message})
    print(f"[Scheduler] Updated announcement: {message}")

scheduler = BackgroundScheduler()
scheduler.add_job(func=update_announcement, trigger="interval", weeks=1)
scheduler.start()

# ---------- Helpers ----------
def generate_ticket_id():
    data = load_json(DATA_FILE)
    return str(int(data[-1]["ticketID"]) + 1).zfill(4) if data else "0000"

def generate_qr_code(ticket_id):
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(f"CHECKOUT:{ticket_id}")
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    qr_path = os.path.join(QRCODE_FOLDER, f"qr_{ticket_id}.png")
    img.save(qr_path)
    return qr_path

def send_sms(to, message):
    if not SMS_ENABLED:
        print(f"⚠️ SMS disabled (would send to {to}): {message}")
        return False
    try:
        res = sms.send_message({"from": VONAGE_PHONE_NUMBER, "to": to, "text": message})
        ok = res["messages"][0]["status"] == "0"
        if not ok:
            print("Vonage error:", res["messages"][0].get("error-text"))
        return ok
    except Exception as e:
        print("SMS exception:", e)
        return False

def login_form():
    return '''
    <html><head><title>Admin Login</title></head>
    <body style="font-family:Segoe UI,Arial;background:#f8f9fa;text-align:center;padding:80px;">
        <h2>🔐 Admin Login</h2>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required><br><br>
            <input type="password" name="password" placeholder="Password" required><br><br>
            <button type="submit">Login</button>
        </form>
        <br><button onclick="window.location.href='/'">← Back</button>
    </body></html>
    '''

# ---------- Pages ----------
@app.route("/", methods=["GET"])
def index():
    data = load_json(DATA_FILE)
    runners = load_json(RUNNER_FILE)
    keys_in_system = len([t for t in data if t["status"] == "Checked-In"])
    message = session.pop("message", None)
    message_type = session.pop("message_type", None)

    # Pagination
    page = int(request.args.get("page", 1))
    per_page = 15
    total = len(data)
    total_pages = (total + per_page - 1) // per_page
    start, end = (page - 1) * per_page, (page - 1) * per_page + per_page
    paged_data = data[start:end]

    html = '''
<!doctype html>
<html>
<head>
  <title>Valet Operations Management System</title>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
  <style>
    body{font-family:"Segoe UI",Arial,sans-serif;background:#f8f9fa;margin:0;}
    .header{display:flex;align-items:center;justify-content:center;position:relative;padding-top:20px;}
    .header img{position:absolute;left:40px;top:15px;width:110px;border-radius:6px;}
    h1{text-align:center;color:#2c3e50;margin-bottom:5px;}
    h2{text-align:center;color:#b30000;margin-top:0;}
    .container{width:92%;margin:30px auto;background:white;padding:30px 40px;border-radius:10px;box-shadow:0 4px 12px rgba(0,0,0,0.1);}
    h3{color:#2c3e50;border-bottom:2px solid #b30000;padding-bottom:5px;}
    input,button,textarea,select{margin:5px;padding:8px;border-radius:5px;border:1px solid #ccc;}
    button{cursor:pointer;font-weight:bold;}
    .checkin-btn{background:#27ae60;color:white;border:none;}
    .checkout-btn{background:#3498db;color:white;border:none;}
    .clockin-btn{background:#8e44ad;color:white;border:none;}
    .delete-btn{background:#e74c3c;color:white;border:none;padding:5px 10px;}
    .scan-btn{background:#f39c12;color:white;border:none;}
    .key-count{background:#b30000;color:white;padding:8px 15px;border-radius:8px;font-weight:bold;display:inline-block;margin:15px 0;}
    table{width:100%;border-collapse:collapse;margin-top:15px;}
    th,td{border:1px solid #ddd;padding:10px;text-align:left;}
    th{background:#b30000;color:white;}
    .ticket-id-col{background:#b30000;color:white;font-weight:bold;text-align:center;}
    tr:nth-child(even){background:#f9f9f9;}
    .status-checkedin{color:#27ae60;font-weight:bold;}
    .status-checkedout{color:#95a5a6;font-weight:bold;}
    .message{padding:15px;margin:10px 0;border-radius:5px;font-weight:bold;}
    .message.success{background:#d4edda;color:#155724;border:1px solid #c3e6cb;}
    .message.warning{background:#fff3cd;color:#856404;border:1px solid #ffeaa7;}
    .pagination{text-align:center;margin-top:25px;}
    .pagination button{background:#b30000;color:white;border:none;padding:8px 14px;margin:3px;border-radius:5px;}
    .pagination strong{color:#b30000;font-size:18px;margin:0 6px;}
    .top-buttons{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}
    .search-box{width: 100%; max-width: 420px;}
    a.as-button{position:absolute;right:40px;top:20px;background:#2c3e50;color:white;
      padding:10px 15px;border-radius:6px;text-decoration:none;font-weight:bold;}
  </style>
</head>
<body>
  <div class="header">
    <img src="/static/moffitt_logo.jpeg" alt="Moffitt Logo" onerror="this.style.display='none'">
    <div>
      <h1>Valet Operations Management System</h1>
      <h2>Moffitt Cancer Center — Red Ramp Valet</h2>
    </div>
    <a class="as-button" href="/admin_login">🔐 Admin Login</a>
  </div>

  <div class="container">
    {% if message %}
      <div class="message {{ message_type or 'success' }}">{{ message }}</div>
    {% endif %}

    <div class="top-buttons">
      <button class="clockin-btn" onclick="location.href='/runner_clockin'">👷 Runner Clock-In Portal</button>
      <button class="checkout-btn" onclick="document.getElementById('quickCheckout').style.display='block'">🚗 Quick Checkout</button>
      <button class="checkout-btn" style="background:#16a085" onclick="location.href='/shift_portal'">📋 Shift Portal</button>
      <button class="checkout-btn" style="background:#2980b9" onclick="location.href='/announcement_page'">📢 Announcements</button>
    </div>

    <h3>🚗 Check-In Vehicle</h3>
    <form id="checkinForm" action="/checkin" method="post" enctype="multipart/form-data">
      <input type="text" name="licensePlate" placeholder="License Plate" required>
      <input type="text" name="customerName" placeholder="Customer Name" required>
      <input type="tel" name="customerPhone" placeholder="+12345678901" required>
      <input type="text" name="carMake" placeholder="Car Make" required>
      <input type="text" name="carColor" placeholder="Car Color" required>
      <input type="text" name="notes" placeholder="Notes (optional)">
      <br><br>
      <input type="file" name="damageImages" id="damageImages" accept="image/*" multiple>
      <button type="submit" class="checkin-btn">✅ Check In Vehicle</button>
    </form>

    <h3>🔑 Checkout Keys</h3>
    <form action="/checkout_manual" method="post" style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
      <input type="text" name="ticket_id" placeholder="Enter Ticket Number" required style="padding:8px;width:200px;border-radius:5px;border:1px solid #ccc;">
      <button type="submit" class="checkout-btn">Check Out</button>
      <button type="button" class="scan-btn" onclick="document.getElementById('quickCheckout').style.display='block'">📷 Use Camera</button>
    </form>
    <hr>

    <h3>📋 All Tickets</h3>
    <input type="text" id="searchBox" class="search-box" placeholder="🔍 Search by ticket #, license plate, name..." onkeyup="searchTable()">
    <div style="text-align:center; margin-top:10px;">
      <div class="key-count">🔑 Keys in System: {{ keys_in_system }}</div>
    </div>

    {% if data %}
      <table id="ticketsTable">
        <thead>
          <tr>
            <th>Ticket ID</th><th>License Plate</th><th>Customer Name</th><th>Phone</th>
            <th>Car Make</th><th>Color</th><th>Status</th><th>Check-In Time</th>
            <th>Assigned Runner</th><th>QR Code</th><th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {% for ticket in data %}
          <tr>
            <td class="ticket-id-col">#{{ ticket.ticketID }}</td>
            <td>{{ ticket.licensePlate }}</td>
            <td>{{ ticket.customerName }}</td>
            <td>{{ ticket.customerPhone }}</td>
            <td>{{ ticket.carMake }}</td>
            <td>{{ ticket.carColor }}</td>
            <td class="status-{{ ticket.status.lower().replace(' ', '').replace('-', '') }}">{{ ticket.status }}</td>
            <td>{{ ticket.checkInTime }}</td>
            <td>{{ ticket.assignedRunner or 'Not Assigned' }}</td>
            <td>
              {% if ticket.status == "Checked-In" %}
                <a href="/qrcode/{{ ticket.ticketID }}" target="_blank"><button class="scan-btn">View QR</button></a>
              {% else %}-{% endif %}
            </td>
            <td>
              {% if ticket.status == "Checked-In" %}
                <form action="/assign_runner/{{ ticket.ticketID }}" method="post" style="display:inline;">
                  <select name="runnerName" required>
                    <option value="">Assign Runner</option>
                    {% for runner in runners %}
                      <option value="{{ runner.name }}">{{ runner.name }}</option>
                    {% endfor %}
                  </select>
                  <button type="submit" class="clockin-btn">Assign</button>
                </form>
                <form action="/vehicle_ready/{{ ticket.ticketID }}" method="post" style="display:inline;">
                  <button type="submit" class="scan-btn">📱 Vehicle Ready</button>
                </form>
                <form action="/checkout/{{ ticket.ticketID }}" method="post" style="display:inline;">
                  <button type="submit" class="checkout-btn">Check Out</button>
                </form>
              {% endif %}
              <form action="/delete/{{ ticket.ticketID }}" method="post" style="display:inline;">
                <button type="submit" class="delete-btn" onclick="return confirm('Delete this ticket?')">Delete</button>
              </form>
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>

      <div class="pagination">
        {% if page > 1 %}<a href="/?page={{ page - 1 }}"><button>⬅️ Prev</button></a>{% endif %}
        {% for i in range(1, total_pages + 1) %}
          {% if i == page %}<strong>[{{ i }}]</strong>{% else %}<a href="/?page={{ i }}"><button>{{ i }}</button></a>{% endif %}
        {% endfor %}
        {% if page < total_pages %}<a href="/?page={{ page + 1 }}"><button>Next ➡️</button></a>{% endif %}
      </div>
    {% else %}
      <p>No tickets yet.</p>
    {% endif %}
  </div>

  <!-- Quick Checkout Modal -->
  <div id="quickCheckout" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.6); text-align:center; z-index:1000;">
    <div style="background:white; margin:100px auto; padding:30px; width:400px; border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.2);">
      <h3>🚗 Quick Checkout</h3>
      <p>Enter or scan a ticket number</p>
      <form action="/checkout_manual" method="post">
        <input type="text" name="ticket_id" placeholder="Ticket #" required style="padding:8px; width:80%;">
        <button type="submit" class="checkout-btn">Check Out</button>
      </form>
      <br>
      <button onclick="startScanner()" class="scan-btn">📷 Scan QR Code</button>
      <button onclick="document.getElementById('quickCheckout').style.display='none'" class="delete-btn">Close</button>
      <div id="qrVideo" style="margin-top:10px; display:none;"></div>
    </div>
  </div>

  <script src="https://unpkg.com/html5-qrcode"></script>
  <script>
    let scanner;
    function startScanner() {
      const el = document.getElementById('qrVideo');
      el.style.display = 'block';
      scanner = new Html5Qrcode("qrVideo");
      scanner.start({ facingMode: "environment" }, { fps: 10, qrbox: 250 },
        (decodedText) => {
          const match = decodedText.match(/CHECKOUT:(\\d+)/);
          if (match) {
            const id = match[1];
            document.querySelector('#quickCheckout input[name="ticket_id"]').value = id;
            scanner.stop().then(()=>{ el.style.display='none'; alert("✅ Ticket " + id + " ready for checkout!"); });
          }
        }, ()=>{}).catch(err => alert("Camera error: " + err));
    }
    function searchTable(){
      const q = document.getElementById('searchBox').value.toLowerCase();
      const rows = document.querySelectorAll('#ticketsTable tbody tr');
      rows.forEach(r=>{
        r.style.display = r.innerText.toLowerCase().includes(q) ? '' : 'none';
      });
    }
  </script>
</body>
</html>
    '''
    return render_template_string(
        html,
        data=paged_data,
        runners=runners,
        keys_in_system=keys_in_system,
        page=page,
        total_pages=total_pages
    )

# ---------- Runner Clock-In ----------
@app.route("/runner_clockin", methods=["GET"])
def runner_clockin_page():
    runners = load_json(RUNNER_FILE)
    html = '''
<html>
<head><title>Runner Clock Portal</title>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<style>
  body{font-family:"Segoe UI",Arial;background:#f8f9fa;margin:0;padding:50px;text-align:center;}
  .portal{max-width:650px;margin:0 auto;background:white;padding:50px;border-radius:15px;box-shadow:0 4px 20px rgba(0,0,0,0.1);}
  h1{color:#2c3e50;margin-bottom:10px;} h2{color:#8e44ad;margin-top:0;}
  input{padding:15px;font-size:18px;width:80%;margin:20px 0;border-radius:5px;border:2px solid #8e44ad;}
  button{padding:12px 30px;font-size:16px;border:none;border-radius:5px;cursor:pointer;margin:10px;}
  .clockin-btn{background:#8e44ad;color:white;} .clockout-btn{background:#e67e22;color:white;} .back-btn{background:#e74c3c;color:white;}
  .runner-list{margin-top:30px;text-align:left;} .runner-item{background:#f8f9fa;padding:15px;margin:10px 0;border-radius:5px;border-left:4px solid #8e44ad;}
</style></head>
<body>
  <a href="/" style="position:absolute;top:20px;right:30px;font-size:40px;color:#e74c3c;text-decoration:none;">✖</a>
  <div class="portal">
    <h1>👷 Runner Clock Portal</h1><h2>Moffitt Cancer Center Valet</h2>
    <form action="/clockin" method="post">
      <input type="text" name="runnerName" placeholder="Enter Your Name" required autofocus><br>
      <button type="submit" class="clockin-btn">🕐 Clock In</button>
    </form>
    <div class="runner-list">
      <h3>🧾 Runner History</h3>
      {% if runners %}
        {% for r in runners %}
          <div class="runner-item">
            <strong>{{ r.name }}</strong><br>
            {% if r.clockOutTime %}
              <small>Clocked In: {{ r.clockInTime }}</small><br>
              <small>Clocked Out: {{ r.clockOutTime }}</small><br>
              <small>⏱ Duration: {{ r.duration }}</small>
            {% else %}
              <small>Clocked In: {{ r.clockInTime }}</small><br>
              <form action="/clockout" method="post" style="display:inline;">
                <input type="hidden" name="runnerName" value="{{ r.name }}">
                <button type="submit" class="clockout-btn">⏹ Clock Out</button>
              </form>
            {% endif %}
          </div>
        {% endfor %}
      {% else %}
        <p style="color:#95a5a6;">No runners recorded yet.</p>
      {% endif %}
    </div>
    <br><br><button class="back-btn" onclick="location.href='/'">← Back to Dashboard</button>
  </div>
</body></html>
    '''
    return render_template_string(html, runners=runners)

@app.route("/clockin", methods=["POST"])
def clockin():
    runners = load_json(RUNNER_FILE)
    name = request.form.get("runnerName", "").strip()
    if not name:
        return redirect("/runner_clockin")
    if any(r["name"].lower() == name.lower() and "clockOutTime" not in r for r in runners):
        session["message"] = f"⚠️ Runner {name} is already clocked in."
        session["message_type"] = "warning"
    else:
        runners.append({"name": name, "clockInTime": datetime.datetime.now().strftime("%b %d, %Y %I:%M %p")})
        save_json(RUNNER_FILE, runners)
        session["message"] = f"✅ Runner {name} clocked in successfully."
        session["message_type"] = "success"
    # If came from portal, go back there
    if request.referrer and "/runner_clockin" in request.referrer:
        return redirect("/runner_clockin")
    return redirect("/")

@app.route("/clockout", methods=["POST"])
def clockout():
    runners = load_json(RUNNER_FILE)
    name = request.form.get("runnerName", "").strip()
    now = datetime.datetime.now()
    for r in runners:
        if r["name"].lower() == name.lower() and "clockOutTime" not in r:
            r["clockOutTime"] = now.strftime("%b %d, %Y %I:%M %p")
            cin = datetime.datetime.strptime(r["clockInTime"], "%b %d, %Y %I:%M %p")
            secs = (now - cin).total_seconds()
            r["duration"] = f"{int(secs//3600)}h {int((secs%3600)//60)}m"
            break
    save_json(RUNNER_FILE, runners)
    session["message"] = f"🕒 {name} clocked out successfully."
    session["message_type"] = "success"
    return redirect("/runner_clockin")

# ---------- Tickets ----------
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
        "damageImages": []
    }

    # Save uploaded damage images (both names supported)
    for field in ("damageImages", "capturedImages"):
        if field in request.files:
            for img in request.files.getlist(field):
                if img.filename:
                    safe_name = f"{ticket_id}_{img.filename}"
                    img_path = os.path.join(UPLOAD_FOLDER, safe_name)
                    img.save(img_path)
                    new_ticket["damageImages"].append(f"/static/uploads/{safe_name}")

    data.append(new_ticket)
    save_json(DATA_FILE, data)
    generate_qr_code(ticket_id)

    # Send Check-in SMS with live QR link
    try:
        qr_url = url_for('static', filename=f"qrcodes/qr_{ticket_id}.png", _external=True)
        msg = (
            f"Hi {new_ticket['customerName']}! Your vehicle has been checked in at Moffitt Valet.\n"
            f"Ticket #{ticket_id}.\n"
            f"📲 Scan your QR code: {qr_url}"
        )
        ok = send_sms(new_ticket["customerPhone"], msg)
        session["message"] = f"✅ Vehicle checked in (Ticket #{ticket_id}). " + ("SMS sent." if ok else "SMS failed.")
    except Exception:
        session["message"] = f"✅ Vehicle checked in (Ticket #{ticket_id})."
    session["message_type"] = "success"
    return redirect("/")

@app.route("/assign_runner/<ticket_id>", methods=["POST"])
def assign_runner(ticket_id):
    data = load_json(DATA_FILE)
    runner_name = request.form.get("runnerName")
    for t in data:
        if t["ticketID"] == ticket_id:
            t["assignedRunner"] = runner_name
            break
    save_json(DATA_FILE, data)
    session["message"] = f"✅ Runner {runner_name} assigned to Ticket #{ticket_id}."
    session["message_type"] = "success"
    return redirect("/")

@app.route("/vehicle_ready/<ticket_id>", methods=["POST"])
def vehicle_ready(ticket_id):
    data = load_json(DATA_FILE)
    for t in data:
        if t["ticketID"] == ticket_id:
            ok = send_sms(t["customerPhone"], f"Your vehicle (Ticket #{ticket_id}) is ready for pickup at Moffitt Valet.")
            session["message"] = "📱 SMS sent to customer!" if ok else "⚠️ Failed to send SMS."
            session["message_type"] = "success"
            break
    return redirect("/")

@app.route("/checkout/<ticket_id>", methods=["POST"])
def checkout(ticket_id):
    data = load_json(DATA_FILE)
    now = datetime.datetime.now().strftime("%b %d, %Y %I:%M %p")
    for t in data:
        if t["ticketID"] == ticket_id and t["status"] == "Checked-In":
            t["status"] = "Checked-Out"
            t["checkOutTime"] = now
            save_json(DATA_FILE, data)
            ok = send_sms(t["customerPhone"], f"Thank you {t['customerName']}! Ticket #{ticket_id} has been checked out. Drive safely!")
            session["message"] = "✅ Checked out. SMS sent." if ok else "✅ Checked out. SMS failed."
            session["message_type"] = "success"
            return redirect("/")
    session["message"] = f"❌ Ticket #{ticket_id} not found or already checked out."
    session["message_type"] = "warning"
    return redirect("/")

@app.route("/checkout_manual", methods=["POST"])
def checkout_manual():
    ticket_id = request.form.get("ticket_id")
    data = load_json(DATA_FILE)
    now = datetime.datetime.now().strftime("%b %d, %Y %I:%M %p")
    for t in data:
        if t["ticketID"] == ticket_id and t["status"] == "Checked-In":
            t["status"] = "Checked-Out"
            t["checkOutTime"] = now
            save_json(DATA_FILE, data)
            send_sms(t["customerPhone"], f"Thank you {t['customerName']}! Ticket #{ticket_id} has been checked out. Drive safely!")
            session["message"] = f"✅ Ticket #{ticket_id} checked out successfully."
            session["message_type"] = "success"
            return redirect("/")
    session["message"] = f"❌ Ticket #{ticket_id} not found or already checked out."
    session["message_type"] = "warning"
    return redirect("/")

@app.route("/delete/<ticket_id>", methods=["POST"])
def delete_ticket(ticket_id):
    data = load_json(DATA_FILE)
    data = [t for t in data if t["ticketID"] != ticket_id]
    save_json(DATA_FILE, data)
    session["message"] = f"🗑️ Ticket #{ticket_id} deleted."
    session["message_type"] = "warning"
    return redirect("/")

@app.route("/qrcode/<ticket_id>")
def view_qrcode(ticket_id):
    qr_path = os.path.join(QRCODE_FOLDER, f"qr_{ticket_id}.png")
    if not os.path.exists(qr_path):
        generate_qr_code(ticket_id)
    html = f'''
<html><head><title>QR Code - Ticket #{ticket_id}</title>
<style>
  body{{text-align:center;padding:50px;font-family:Arial;background:#f8f9fa;}}
  .qr-container{{background:white;padding:40px;border-radius:10px;display:inline-block;box-shadow:0 4px 12px rgba(0,0,0,0.1);}}
  h1{{color:#2c3e50;}} p{{color:#666;font-size:18px;}}
  img{{border:4px solid #2c3e50;padding:20px;background:white;border-radius:10px;}}
  button{{padding:15px 30px;font-size:16px;margin:10px;border:none;border-radius:5px;cursor:pointer;}}
  .print-btn{{background:#8e44ad;color:white;}} .close-btn{{background:#e74c3c;color:white;}}
</style></head>
<body>
  <div class="qr-container">
    <h1>Ticket #{ticket_id}</h1>
    <p>Scan this QR code at checkout</p>
    <img src="/static/qrcodes/qr_{ticket_id}.png" alt="QR Code">
    <br><br>
    <button class="print-btn" onclick="window.print()">🖨️ Print QR Code</button>
    <button class="close-btn" onclick="window.close()">Close</button>
  </div>
</body></html>
    '''
    return html

# Serve any file path (fallback). Static files are under /static/* already.
@app.route("/<path:filename>")
def serve_file(filename):
    return send_from_directory(".", filename)

# ---------- Announcements ----------
@app.route("/announcement")
def get_announcement():
    return jsonify(load_announcement())

@app.route("/announcement_page")
def announcement_page():
    return '''
<html>
<head><title>Announcements</title></head>
<body style="font-family:Segoe UI,Arial;background:#f8f9fa;padding:40px;">
  <h2>📢 Weekly Announcements</h2>
  <div id="announcement" style="font-size:18px;color:#2c3e50;margin-top:20px;"></div>
  <script>
    async function load(){
      let res = await fetch("/announcement");
      let data = await res.json();
      document.getElementById("announcement").innerText = data.message;
    }
    load();
  </script>
  <br><button onclick="location.href='/'">← Back to Dashboard</button>
</body></html>
    '''

# ---------- Shifts (separate file) ----------
@app.route("/shift_portal")
def shift_portal():
    return '''
<html>
<head>
  <title>Shift Management</title>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
  <style>
    body{font-family:Segoe UI,Arial;background:#f8f9fa;padding:40px;}
    h2{color:#2c3e50;}
    .add-shift-form{background:white;padding:20px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);margin-bottom:30px;}
    table{width:100%;border-collapse:collapse;margin-top:20px;background:white;box-shadow:0 2px 8px rgba(0,0,0,0.1);}
    th,td{border:1px solid #ddd;padding:12px;text-align:center;}
    th{background:#b30000;color:white;font-weight:bold;}
    tr:nth-child(even){background:#f9f9f9;}
    .available{color:#27ae60;font-weight:bold;}
    .taken{color:#95a5a6;}
    button{padding:8px 16px;border:none;border-radius:5px;cursor:pointer;font-weight:bold;margin:2px;}
    .pick-btn{background:#27ae60;color:white;}
    .drop-btn{background:#e74c3c;color:white;}
    .delete-btn{background:#c0392b;color:white;}
    .add-btn{background:#3498db;color:white;padding:10px 20px;}
    .back-btn{background:#2c3e50;color:white;padding:12px 24px;margin-top:20px;}
    select,input{padding:8px;border-radius:5px;border:1px solid #ccc;margin:5px;}
  </style>
</head>
<body>
  <h2>📋 Shift Management Portal</h2>
  <p>Add new shifts, pick available shifts, or drop assigned shifts</p>
  <div class="add-shift-form">
    <h3>➕ Add New Shift</h3>
    <select id="daySelect">
      <option value="">Select Day</option>
      <option>Monday</option><option>Tuesday</option><option>Wednesday</option>
      <option>Thursday</option><option>Friday</option><option>Saturday</option><option>Sunday</option>
    </select>
    <select id="timeSelect">
      <option value="">Select Time</option>
      <option>6am-2pm</option><option>7am-3pm</option><option>8am-4pm</option>
      <option>8:30am-4:30pm</option><option>9am-5pm</option><option>10am-6pm</option>
      <option>11am-7pm</option><option>12pm-8pm</option><option>2pm-10pm</option>
    </select>
    <button class="add-btn" onclick="addShift()">Add Shift</button>
  </div>

  <div id="shiftTable"></div>

  <script>
    async function loadShifts(){
      const res = await fetch("/shifts");
      const data = await res.json();
      let html = "<table><tr><th>Shift ID</th><th>Day</th><th>Time</th><th>Assigned To</th><th>Actions</th></tr>";
      if(data.length === 0){
        html += "<tr><td colspan='5' style='text-align:center;color:#95a5a6;'>No shifts yet. Add one above!</td></tr>";
      }
      data.forEach(s=>{
        html += `<tr>
          <td><strong>#${s.id}</strong></td>
          <td><strong>${s.day}</strong></td>
          <td>${s.time}</td>
          <td class="${s.assigned_to ? 'taken' : 'available'}">${s.assigned_to || 'Available'}</td>
          <td>
            ${s.assigned_to
              ? `<button class="drop-btn" onclick="dropAssignment(${s.id}, '${s.assigned_to}')">Unassign</button>`
              : `<button class="pick-btn" onclick="pick(${s.id}, '${s.day}', '${s.time}')">Pick Shift</button>`
            }
            <button class="delete-btn" onclick="deleteShift(${s.id})">Delete</button>
          </td></tr>`;
      });
      html += "</table>";
      document.getElementById("shiftTable").innerHTML = html;
    }

    async function addShift(){
      const day = document.getElementById("daySelect").value;
      const time = document.getElementById("timeSelect").value;
      if(!day || !time){ alert("Select both day and time"); return; }
      const res = await fetch("/add_shift",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({day,time})});
      if(res.ok){ alert("✅ Shift added"); loadShifts(); } else { alert("❌ Failed to add"); }
    }

    async function pick(id, day, time){
      const user = prompt(`Enter your name to pick ${day} ${time}:`); if(!user) return;
      const res = await fetch("/pick_shift",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id,user:user.trim()})});
      if(res.ok){ alert("✅ Picked"); loadShifts(); } else { alert("❌ Failed"); }
    }

    async function dropAssignment(id, user){
      const u = prompt(`Enter your name to unassign (must match: ${user}):`); if(!u) return;
      if(u.trim().toLowerCase() !== user.toLowerCase()) return alert("Name doesn't match.");
      const res = await fetch("/drop_shift",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id,user:u.trim()})});
      if(res.ok){ alert("✅ Unassigned"); loadShifts(); } else { alert("❌ Failed"); }
    }

    async function deleteShift(id){
      if(!confirm("Delete this shift?")) return;
      const res = await fetch("/delete_shift",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id})});
      if(res.ok){ alert("✅ Deleted"); loadShifts(); } else { alert("❌ Failed"); }
    }

    loadShifts();
  </script>

  <button class="back-btn" onclick="location.href='/'">← Back to Dashboard</button>
</body></html>
    '''

@app.route("/shifts")
def shifts():
    return jsonify(load_json(SHIFTS_FILE))

@app.route("/add_shift", methods=["POST"])
def add_shift():
    data = load_json(SHIFTS_FILE)
    payload = request.get_json(force=True)
    day = payload.get("day"); time = payload.get("time")
    if not day or not time:
        return jsonify({"error": "day and time required"}), 400
    new_id = (max([s["id"] for s in data]) + 1) if data else 1
    data.append({"id": new_id, "day": day, "time": time, "assigned_to": None})
    save_json(SHIFTS_FILE, data)
    return jsonify({"id": new_id})

@app.route("/pick_shift", methods=["POST"])
def pick_shift():
    data = load_json(SHIFTS_FILE)
    payload = request.get_json(force=True)
    sid = int(payload.get("id")); user = payload.get("user","").strip()
    for s in data:
        if s["id"] == sid and not s["assigned_to"]:
            s["assigned_to"] = user
            save_json(SHIFTS_FILE, data)
            return jsonify({"ok": True})
    return jsonify({"error": "Shift not found or already taken"}), 400

@app.route("/drop_shift", methods=["POST"])
def drop_shift():
    data = load_json(SHIFTS_FILE)
    payload = request.get_json(force=True)
    sid = int(payload.get("id")); user = payload.get("user","").strip()
    for s in data:
        if s["id"] == sid and s["assigned_to"] and s["assigned_to"].lower() == user.lower():
            s["assigned_to"] = None
            save_json(SHIFTS_FILE, data)
            return jsonify({"ok": True})
    return jsonify({"error": "Cannot unassign"}), 400

@app.route("/delete_shift", methods=["POST"])
def delete_shift():
    data = load_json(SHIFTS_FILE)
    sid = int(request.get_json(force=True).get("id"))
    new = [s for s in data if s["id"] != sid]
    if len(new) == len(data):
        return jsonify({"error": "Shift not found"}), 404
    save_json(SHIFTS_FILE, new)
    return jsonify({"ok": True})

# ---------- Admin ----------
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        u = request.form.get("username"); p = request.form.get("password")
        if check_admin_credentials(u, p):
            session["admin_logged_in"] = True
            session["admin_username"] = u
            return redirect("/admin")
        return "<h3 style='color:red;text-align:center;'>Invalid credentials</h3>" + login_form()
    return login_form()

@app.route("/admin_logout")
@login_required
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect("/")

@app.route("/admin")
@login_required
def admin_dashboard():
    return '''
<html><head><title>Admin Panel</title></head>
<body style="font-family:Segoe UI,Arial;background:#f8f9fa;padding:40px;">
  <h2>🧑‍💼 Admin Dashboard</h2>
  <h3>📢 Post Announcement</h3>
  <form method="POST" action="/admin_announcement">
    <textarea name="message" placeholder="Type announcement..." rows="3" cols="50" required></textarea><br>
    <button type="submit">Post Announcement</button>
  </form>
  <hr>
  <h3>🗓️ Shifts (use Shift Portal for full UI)</h3>
  <form method="POST" action="/admin_add_shift">
    <input type="text" name="date" placeholder="Date (e.g. 2025-11-05)" required>
    <input type="text" name="time" placeholder="Time (e.g. Morning)" required>
    <button type="submit">Add Shift</button>
  </form>
  <form method="POST" action="/admin_remove_shift">
    <input type="number" name="id" placeholder="Shift ID to remove" required>
    <button type="submit">Remove Shift</button>
  </form>
  <hr>
  <button onclick="location.href='/'">← Back to Dashboard</button>
  <button onclick="location.href='/admin_logout'">Logout</button>
</body></html>
    '''

@app.route("/admin_announcement", methods=["POST"])
@login_required
def admin_announcement():
    message = request.form.get("message", "").strip()
    if message:
        save_announcement({"message": f"Admin Update: {message}"})
    return redirect("/admin")

# Admin add/remove shift (writes to SHIFTS_FILE)
@app.route("/admin_add_shift", methods=["POST"])
@login_required
def admin_add_shift():
    date = request.form.get("date", "").strip()
    time = request.form.get("time", "").strip()
    # Store date in "day" field for compatibility (or parse weekday if needed)
    shifts = load_json(SHIFTS_FILE)
    new_id = (max([s["id"] for s in shifts]) + 1) if shifts else 1
    shifts.append({"id": new_id, "day": date, "time": time, "assigned_to": None})
    save_json(SHIFTS_FILE, shifts)
    return redirect("/admin")

@app.route("/admin_remove_shift", methods=["POST"])
@login_required
def admin_remove_shift():
    sid = int(request.form.get("id"))
    shifts = load_json(SHIFTS_FILE)
    shifts = [s for s in shifts if s["id"] != sid]
    save_json(SHIFTS_FILE, shifts)
    return redirect("/admin")

# ---------- Run ----------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 VALET OPERATIONS MANAGEMENT SYSTEM")
    print("="*60)
    port = int(os.getenv("PORT", 5050))  # Railway sets PORT
    print(f"URL: http://127.0.0.1:{port}")
    print(f"SMS: {'✅ ENABLED' if SMS_ENABLED else '⚠️ DISABLED'}")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=True)
