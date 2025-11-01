from flask import Flask, render_template_string, request, redirect, jsonify, session, send_from_directory
import json, os, datetime, base64, qrcode
from io import BytesIO
# from damage_detector import detect_damage  # ← COMMENTED OUT
import vonage
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, render_template_string, request, redirect, jsonify, session, send_from_directory, url_for
from functools import wraps


app = Flask(__name__)
app.secret_key = "valet_secret_key"

# --- Vonage SMS Configuration ---
VONAGE_API_KEY = "57915c86"
VONAGE_API_SECRET = "Valet123"  # Your actual Vonage API secret from the dashboard
VONAGE_PHONE_NUMBER = "+13073963461"  # Your Vonage virtual number

# Initialize Vonage client
try:
    vonage_client = vonage.Client(key=VONAGE_API_KEY, secret=VONAGE_API_SECRET)
    sms = vonage.Sms(vonage_client)
    SMS_ENABLED = True
    print("✅ Vonage SMS enabled")
except Exception as e:
    SMS_ENABLED = False
    print("⚠️ Vonage SMS disabled (check credentials)")
    print(f"Error: {e}")


# --- Upload setup ---
UPLOAD_FOLDER = os.path.join("static", "uploads")
QRCODE_FOLDER = os.path.join("static", "qrcodes")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["QRCODE_FOLDER"] = QRCODE_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(QRCODE_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

# --- Announcement & Shift Setup ---
ANNOUNCEMENT_FILE = "announcement.json"
if not os.path.exists(ANNOUNCEMENT_FILE):
    with open(ANNOUNCEMENT_FILE, "w") as f:
        json.dump({"message": "Welcome to the Valet Operations System!"}, f)

def load_announcement():
    with open(ANNOUNCEMENT_FILE, "r") as f:
        return json.load(f)

def save_announcement(data):
    with open(ANNOUNCEMENT_FILE, "w") as f:
        json.dump(data, f, indent=2)

def update_announcement():
    """Update weekly announcement automatically"""
    message = f"Weekly Valet Update – {datetime.datetime.now().strftime('%A, %B %d, %Y')}"
    save_announcement({"message": message})
    print(f"[Scheduler] Updated announcement: {message}")

# Start the background scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_announcement, trigger="interval", weeks=1)
scheduler.start()
# --- Admin Authentication Setup ---
# --- Admin Authentication Setup (JSON-based) ---
ADMIN_FILE = "admins.json"

if not os.path.exists(ADMIN_FILE):
    with open(ADMIN_FILE, "w") as f:
        json.dump([{"username": "admin", "password": "valet123"}], f, indent=2)

def load_admins():
    with open(ADMIN_FILE, "r") as f:
        return json.load(f)

def check_admin_credentials(username, password):
    admins = load_admins()
    for admin in admins:
        if admin["username"] == username and admin["password"] == password:
            return True
    return False

def login_required(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return func(*args, **kwargs)
    return wrapper


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return func(*args, **kwargs)
    return wrapper



# --- Data setup ---
DATA_FILE = "data.json"
RUNNER_FILE = "runners.json"

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump([], f)

if not os.path.exists(RUNNER_FILE):
    with open(RUNNER_FILE, "w") as f:
        json.dump([], f)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def generate_ticket_id():
    data = load_json(DATA_FILE)
    if not data:
        return "0000"
    last_id = data[-1]["ticketID"]
    next_id = str(int(last_id) + 1).zfill(4)
    return next_id

def generate_qr_code(ticket_id):
    """Generate QR code for ticket checkout"""
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(f"CHECKOUT:{ticket_id}")
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    qr_path = os.path.join(QRCODE_FOLDER, f"qr_{ticket_id}.png")
    img.save(qr_path)
    return qr_path

# --- SMS Functions ---
def send_sms(to_number, message):
    """Send SMS via Vonage"""
    if not SMS_ENABLED:
        print(f"⚠️ SMS disabled - would have sent to {to_number}: {message}")
        return False
    
    try:
        response = sms.send_message({
            "from": VONAGE_PHONE_NUMBER,
            "to": to_number,
            "text": message
        })
        
        if response["messages"][0]["status"] == "0":
            print(f"✅ SMS sent successfully to {to_number}")
            return True
        else:
            print(f"❌ SMS failed: {response['messages'][0]['error-text']}")
            return False
    except Exception as e:
        print(f"❌ SMS error: {str(e)}")
        return False

def send_checkin_sms(customer_name, customer_phone, ticket_id):
    # Local IP so your phone can open it directly
    server_ip = "192.168.1.22"  # ← Replace with your Mac’s IP
    qr_link = f"http://{server_ip}:5050/static/qrcodes/qr_{ticket_id}.png"

    message = (
        f"Hi {customer_name}! Your vehicle has been checked in at Moffitt Valet.\n"
        f"Ticket #{ticket_id}.\n"
        f"📲 Scan your QR code: {qr_link}"
    )
    return send_sms(customer_phone, message)



def send_ready_sms(customer_phone, ticket_id):
    """Send SMS when vehicle is ready"""
    message = f"Your vehicle (Ticket #{ticket_id}) is ready for pickup at Moffitt Valet. See you soon!"
    return send_sms(customer_phone, message)

def send_checkout_sms(customer_name, customer_phone, ticket_id):
    """Send SMS when vehicle is checked out"""
    message = f"Thank you {customer_name}! Your vehicle (Ticket #{ticket_id}) has been checked out. Drive safely!"
    return send_sms(customer_phone, message)


# -----------------------------
@app.route("/", methods=["GET"])
def index():
    data = load_json(DATA_FILE)
    runners = load_json(RUNNER_FILE)
    keys_in_system = len([t for t in data if t["status"] == "Checked-In"])
    message = session.pop("message", None)
    message_type = session.pop("message_type", None)

    # ---------- Pagination ----------
    page = int(request.args.get("page", 1))
    tickets_per_page = 15
    total_tickets = len(data)
    total_pages = (total_tickets + tickets_per_page - 1) // tickets_per_page

    start = (page - 1) * tickets_per_page
    end = start + tickets_per_page
    paged_data = data[start:end]

    html = ''' 
    <html>
    <head>
        <title>Valet Operations Management System</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body{font-family:"Segoe UI",Arial,sans-serif;background:#f8f9fa;margin:0;}
            .header{display:flex;align-items:center;justify-content:center;position:relative;padding-top:20px;}
            .header img{position:absolute;left:40px;top:15px;width:110px;border-radius:6px;}
            h1{text-align:center;color:#2c3e50;margin-bottom:5px;}
            h2{text-align:center;color:#b30000;margin-top:0;}
            .container{width:90%;margin:30px auto;background:white;padding:30px 40px;border-radius:10px;box-shadow:0 4px 12px rgba(0,0,0,0.1);}
            h3{color:#2c3e50;border-bottom:2px solid #b30000;padding-bottom:5px;}
            input,button,textarea,select{margin:5px;padding:8px;border-radius:5px;border:1px solid #ccc;}
            button{cursor:pointer;font-weight:bold;}
            .checkin-btn{background:#27ae60;color:white;border:none;}
            .checkout-btn{background:#3498db;color:white;border:none;}
            .clockin-btn{background:#8e44ad;color:white;border:none;}
            .delete-btn{background:#e74c3c;color:white;border:none;padding:5px 10px;}
            .scan-btn{background:#f39c12;color:white;border:none;}
            .runner-pill{background:#b30000;color:white;padding:6px 10px;border-radius:20px;margin-right:8px;display:inline-block;}
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
        </style>
    </head>
    <body>
        <div class="header">
            <img src="static/moffitt_logo.jpeg" alt="Moffitt Logo">
            <div>
                <h1>Valet Operations Management System</h1>
                <h2>Moffitt Cancer Center — Red Ramp Valet</h2>
            </div>
            <a href="/admin_login" 
   style="position:absolute;right:40px;top:20px;background:#2c3e50;color:white;
          padding:10px 15px;border-radius:6px;text-decoration:none;font-weight:bold;">
   🔐 Admin Login
</a>

        </div>

        <div class="container">
            {% if message %}
            <div class="message {{ message_type or 'success' }}">{{ message }}</div>
            {% endif %}

            <div class="top-buttons">
    <button class="clockin-btn" onclick="window.location.href='/runner_clockin'">👷 Runner Clock-In Portal</button>
    <button class="checkout-btn" onclick="document.getElementById('quickCheckout').style.display='block'">🚗 Quick Checkout</button>

    <button class="checkout-btn" style="background:#16a085" onclick="window.location.href='/shift_portal'">📋 Shift Portal</button>
    <button class="checkout-btn" style="background:#2980b9" onclick="window.location.href='/announcement_page'">📢 Announcements</button>
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

  <!-- AI Damage Capture & Analysis -->
  <input type="file" name="damageImages" id="damageImages" accept="image/*" multiple>
  <button type="button" class="scan-btn" onclick="startCamera()">📷 Use Camera</button>
  <br>

  <video id="cameraFeed" autoplay style="display:none;"></video>
  <canvas id="cameraCanvas" style="display:none;"></canvas>
  <button type="button" id="captureBtn" class="checkin-btn" style="display:none;" onclick="capturePhoto()">Capture Photo</button>
  <button type="button" id="stopCameraBtn" class="delete-btn" style="display:none;" onclick="stopCamera()">Stop Camera</button>

  <div id="imagePreviews"></div>

  <div id="analysisResults" class="damage-results" style="display:none;">
    <div id="analysisContent"></div>
  </div>

  <br><br>
  <button type="submit" class="checkin-btn">✅ Check In Vehicle</button>
</form>

        

            <h3>🔑 Checkout Keys</h3>
<form action="/checkout_manual" method="post" style="display:flex;align-items:center;justify-content:center;gap:10px;">
    <input type="text" name="ticket_id" placeholder="Enter Ticket Number" required 
           style="padding:8px;width:200px;border-radius:5px;border:1px solid #ccc;">
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
                        <th>Ticket ID</th>
                        <th>License Plate</th>
                        <th>Customer Name</th>
                        <th>Phone</th>
                        <th>Car Make</th>
                        <th>Color</th>
                        <th>Status</th>
                        <th>Check-In Time</th>
                        <th>Assigned Runner</th>
                        <th>QR Code</th>
                        <th>Actions</th>
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
                        <td class="status-{{ ticket.status.lower().replace(' ', '').replace('-', '') }}">
                            {{ ticket.status }}
                        </td>
                        <td>{{ ticket.checkInTime }}</td>
                        <td>{{ ticket.assignedRunner or 'Not Assigned' }}</td>
                        <td>
                            {% if ticket.status == "Checked-In" %}
                            <a href="/qrcode/{{ ticket.ticketID }}" target="_blank">
                                <button class="scan-btn">View QR</button>
                            </a>
                            {% else %}
                            -
                            {% endif %}
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
                {% if page > 1 %}
                    <a href="/?page={{ page - 1 }}"><button>⬅️ Prev</button></a>
                {% endif %}
                {% for i in range(1, total_pages + 1) %}
                    {% if i == page %}
                        <strong>[{{ i }}]</strong>
                    {% else %}
                        <a href="/?page={{ i }}"><button>{{ i }}</button></a>
                    {% endif %}
                {% endfor %}
                {% if page < total_pages %}
                    <a href="/?page={{ page + 1 }}"><button>Next ➡️</button></a>
                {% endif %}
            </div>

            {% else %}
<p>No tickets yet.</p>
{% endif %}
</div>

<!-- 🔽 Paste the Quick Checkout Modal block here -->
<!-- Quick Checkout Modal -->
<div id="quickCheckout" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; 
background:rgba(0,0,0,0.6); text-align:center; z-index:1000;">
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
    <video id="qrVideo" width="300" height="200" style="margin-top:10px; display:none;"></video>
  </div>
</div>

<script src="https://unpkg.com/html5-qrcode"></script>
<script>
let scanner;
function startScanner() {
  document.getElementById('qrVideo').style.display = 'block';
  scanner = new Html5Qrcode("qrVideo");
  scanner.start({ facingMode: "environment" }, 
      { fps: 10, qrbox: 250 },
      (decodedText) => {
          const match = decodedText.match(/CHECKOUT:(\d+)/);
          if (match) {
              const ticketID = match[1];
              document.querySelector('#quickCheckout input[name="ticket_id"]').value = ticketID;
              scanner.stop();
              document.getElementById('qrVideo').style.display = 'none';
              alert("✅ Ticket " + ticketID + " ready for checkout!");
          }
      },
      (error) => {}
  ).catch(err => alert("Camera error: " + err));
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
        message=message, 
        message_type=message_type,
        page=page, 
        total_pages=total_pages
    )

#ai detection


#-----------------------------
# Runner Clock-In Portal (Separate Page)
# -----------------------------
@app.route("/runner_clockin", methods=["GET"])
def runner_clockin_page():
    runners = load_json(RUNNER_FILE)

    html = '''
    <html>
    <head>
        <title>Runner Clock Portal</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body{font-family:"Segoe UI",Arial,sans-serif;background:#f8f9fa;margin:0;padding:50px;text-align:center;}
            .portal{max-width:650px;margin:0 auto;background:white;padding:50px;border-radius:15px;box-shadow:0 4px 20px rgba(0,0,0,0.1);}
            h1{color:#2c3e50;margin-bottom:10px;}
            h2{color:#8e44ad;margin-top:0;}
            input{padding:15px;font-size:18px;width:80%;margin:20px 0;border-radius:5px;border:2px solid #8e44ad;}
            button{padding:12px 30px;font-size:16px;border:none;border-radius:5px;cursor:pointer;margin:10px;}
            .clockin-btn{background:#8e44ad;color:white;}
            .clockout-btn{background:#e67e22;color:white;}
            .back-btn{background:#e74c3c;color:white;}
            .runner-list{margin-top:30px;text-align:left;}
            .runner-item{background:#f8f9fa;padding:15px;margin:10px 0;border-radius:5px;border-left:4px solid #8e44ad;}
            .close-icon{position:absolute;top:20px;right:30px;font-size:40px;color:#e74c3c;cursor:pointer;text-decoration:none;}
        </style>
    </head>
    <body>
        <a href="/" class="close-icon">✖</a>
        
        <div class="portal">
            <h1>👷 Runner Clock Portal</h1>
            <h2>Moffitt Cancer Center Valet</h2>
            
            <form action="/clockin" method="post">
                <input type="text" name="runnerName" placeholder="Enter Your Name" required autofocus>
                <br>
                <button type="submit" class="clockin-btn">🕐 Clock In</button>
            </form>
            
            <div class="runner-list">
                <h3>🧾 Runner History</h3>
                {% if runners %}
                    {% for runner in runners %}
                    <div class="runner-item">
                        <strong>{{ runner.name }}</strong><br>
                        {% if runner.clockOutTime %}
                            <small>Clocked In: {{ runner.clockInTime }}</small><br>
                            <small>Clocked Out: {{ runner.clockOutTime }}</small><br>
                            <small>⏱ Duration: {{ runner.duration }}</small>
                        {% else %}
                            <small>Clocked In: {{ runner.clockInTime }}</small><br>
                            <form action="/clockout" method="post" style="display:inline;">
                                <input type="hidden" name="runnerName" value="{{ runner.name }}">
                                <button type="submit" class="clockout-btn">⏹ Clock Out</button>
                            </form>
                        {% endif %}
                    </div>
                    {% endfor %}
                {% else %}
                    <p style="color:#95a5a6;">No runners recorded yet.</p>
                {% endif %}
            </div>
            
            <br><br>
            <button class="back-btn" onclick="window.location.href='/'">← Back to Dashboard</button>
        </div>
    </body>
    </html>
    '''
    return render_template_string(html, runners=runners)



# -----------------------------
# Routes
# -----------------------------

#CLOCK IN
@app.route("/clockin", methods=["POST"])
def clockin():
    runners = load_json(RUNNER_FILE)
    name = request.form.get("runnerName").strip()
    
    if any(r["name"].lower() == name.lower() for r in runners):
        session["message"] = f"⚠️ Runner {name} is already clocked in."
        session["message_type"] = "warning"
    else:
        runners.append({"name": name, "clockInTime": datetime.datetime.now().strftime("%b %d, %Y %I:%M %p")})
        save_json(RUNNER_FILE, runners)
        session["message"] = f"✅ Runner {name} clocked in successfully."
        session["message_type"] = "success"
    
    # Redirect back to clock-in page if came from there
    referer = request.referrer or '/'
    if 'runner_clockin' in referer:
        return redirect('/runner_clockin')
    return redirect("/")
#CLOCK OUT
@app.route("/clockout", methods=["POST"])
def clockout():
    runners = load_json(RUNNER_FILE)
    name = request.form.get("runnerName").strip()

    now = datetime.datetime.now()
    for r in runners:
        if r["name"].lower() == name.lower() and "clockOutTime" not in r:
            r["clockOutTime"] = now.strftime("%b %d, %Y %I:%M %p")

            # Calculate total time worked
            clock_in_time = datetime.datetime.strptime(r["clockInTime"], "%b %d, %Y %I:%M %p")
            worked_seconds = (now - clock_in_time).total_seconds()
            hours = int(worked_seconds // 3600)
            minutes = int((worked_seconds % 3600) // 60)
            r["duration"] = f"{hours}h {minutes}m"

    save_json(RUNNER_FILE, runners)
    session["message"] = f"🕒 {name} clocked out successfully."
    session["message_type"] = "success"

    return redirect("/runner_clockin")


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
    
    # Save uploaded/captured damage images
    if 'capturedImages' in request.files:
        images = request.files.getlist('capturedImages')
        for img in images:
            if img.filename:
                img_path = os.path.join(UPLOAD_FOLDER, f"{ticket_id}_{img.filename}")
                img.save(img_path)
                new_ticket["damageImages"].append(img_path)

    data.append(new_ticket)
    save_json(DATA_FILE, data)
    generate_qr_code(ticket_id)
    
    # 📱 SEND CHECK-IN SMS
    sms_sent = send_checkin_sms(
        new_ticket["customerName"], 
        new_ticket["customerPhone"], 
        ticket_id
    )
    
    if sms_sent:
        session["message"] = f"✅ Vehicle checked in (Ticket #{ticket_id}). SMS sent to customer."
    else:
        session["message"] = f"✅ Vehicle checked in (Ticket #{ticket_id}). SMS failed - check Vonage config."
    
    session["message_type"] = "success"
    return redirect("/")

@app.route("/assign_runner/<ticket_id>", methods=["POST"])
def assign_runner(ticket_id):
    data = load_json(DATA_FILE)
    runner_name = request.form.get("runnerName")
    
    for ticket in data:
        if ticket["ticketID"] == ticket_id:
            ticket["assignedRunner"] = runner_name
            break
    
    save_json(DATA_FILE, data)
    session["message"] = f"✅ Runner {runner_name} assigned to Ticket #{ticket_id}."
    session["message_type"] = "success"
    return redirect("/")

@app.route("/vehicle_ready/<ticket_id>", methods=["POST"])
def vehicle_ready(ticket_id):
    """Send SMS when vehicle is ready for pickup"""
    data = load_json(DATA_FILE)
    
    for ticket in data:
        if ticket["ticketID"] == ticket_id:
            # 📱 SEND VEHICLE READY SMS
            sms_sent = send_ready_sms(ticket["customerPhone"], ticket_id)
            
            if sms_sent:
                session["message"] = f"📱 SMS sent to customer - Vehicle #{ticket_id} is ready for pickup!"
            else:
                session["message"] = f"⚠️ Failed to send SMS - check Vonage configuration"
            
            session["message_type"] = "success"
            break
    
    return redirect("/")

@app.route("/checkout/<ticket_id>", methods=["POST"])
def checkout(ticket_id):
    data = load_json(DATA_FILE)
    now = datetime.datetime.now().strftime("%b %d, %Y %I:%M %p")
    
    found = False
    customer_name = ""
    customer_phone = ""
    
    for ticket in data:
        if ticket["ticketID"] == ticket_id and ticket["status"] == "Checked-In":
            ticket["status"] = "Checked-Out"
            ticket["checkOutTime"] = now
            customer_name = ticket["customerName"]
            customer_phone = ticket["customerPhone"]
            found = True
            break
    
    if found:
        save_json(DATA_FILE, data)
        
        # 📱 SEND CHECKOUT SMS
        sms_sent = send_checkout_sms(customer_name, customer_phone, ticket_id)
        
        if sms_sent:
            session["message"] = f"✅ Ticket #{ticket_id} checked out. SMS confirmation sent to customer."
        else:
            session["message"] = f"✅ Ticket #{ticket_id} checked out. SMS failed - check Vonage config."
        
        session["message_type"] = "success"
        return redirect("/")
    else:
        session["message"] = f"❌ Ticket #{ticket_id} not found or already checked out."
        session["message_type"] = "warning"
        return redirect("/")

@app.route("/checkout_manual", methods=["POST"])
def checkout_manual():
    ticket_id = request.form.get("ticket_id")
    data = load_json(DATA_FILE)
    now = datetime.datetime.now().strftime("%b %d, %Y %I:%M %p")
    
    found = False
    for ticket in data:
        if ticket["ticketID"] == ticket_id and ticket["status"] == "Checked-In":
            ticket["status"] = "Checked-Out"
            ticket["checkOutTime"] = now
            found = True
            send_checkout_sms(ticket["customerName"], ticket["customerPhone"], ticket_id)
            break
    
    if found:
        save_json(DATA_FILE, data)
        session["message"] = f"✅ Ticket #{ticket_id} checked out successfully."
        session["message_type"] = "success"
    else:
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
    <html>
    <head>
        <title>QR Code - Ticket #{ticket_id}</title>
        <style>
            body{{text-align:center;padding:50px;font-family:Arial;background:#f8f9fa;}}
            .qr-container{{background:white;padding:40px;border-radius:10px;display:inline-block;box-shadow:0 4px 12px rgba(0,0,0,0.1);}}
            h1{{color:#2c3e50;}}
            p{{color:#666;font-size:18px;}}
            img{{border:4px solid #2c3e50;padding:20px;background:white;border-radius:10px;}}
            button{{padding:15px 30px;font-size:16px;margin:10px;border:none;border-radius:5px;cursor:pointer;}}
            .print-btn{{background:#8e44ad;color:white;}}
            .close-btn{{background:#e74c3c;color:white;}}
        </style>
    </head>
    <body>
        <div class="qr-container">
            <h1>Ticket #{ticket_id}</h1>
            <p>Scan this QR code at checkout</p>
            <img src="/{qr_path}" alt="QR Code">
            <br><br>
            <button class="print-btn" onclick="window.print()">🖨️ Print QR Code</button>
            <button class="close-btn" onclick="window.close()">Close</button>
        </div>
    </body>
    </html>
    '''
    return html

@app.route("/<path:filename>")
def serve_file(filename):
    return send_from_directory(".", filename)

@app.route("/admin_announcement", methods=["POST"])
@login_required
def admin_announcement():
    message = request.form.get("message", "").strip()
    if message:
        save_announcement({"message": f"Admin Update: {message}"})
    return redirect("/admin")

@app.route("/admin_add_shift", methods=["POST"])
@login_required
def admin_add_shift():
    date = request.form.get("date")
    time = request.form.get("time")

    all_data = {"tickets": [], "runners": [], "shifts": []}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            existing = json.load(f)
            if isinstance(existing, list):
                all_data["tickets"] = existing
            else:
                all_data.update(existing)

    new_id = len(all_data.get("shifts", [])) + 1
    all_data.setdefault("shifts", []).append({"id": new_id, "date": date, "time": time, "assigned_to": None})

    with open(DATA_FILE, "w") as f:
        json.dump(all_data, f, indent=2)

    return redirect("/admin")

@app.route("/admin_remove_shift", methods=["POST"])
@login_required
def admin_remove_shift():
    shift_id = int(request.form.get("id"))
    all_data = {"tickets": [], "runners": [], "shifts": []}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            existing = json.load(f)
            if isinstance(existing, list):
                all_data["tickets"] = existing
            else:
                all_data.update(existing)

    all_data["shifts"] = [s for s in all_data.get("shifts", []) if s["id"] != shift_id]

    with open(DATA_FILE, "w") as f:
        json.dump(all_data, f, indent=2)

    return redirect("/admin")


# -----------------------------
# Announcements and Shift Portal
# -----------------------------
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
        <br><button onclick="window.location.href='/'">← Back to Dashboard</button>
    </body>
    </html>
    '''

# Simple Shift Portal using data.json
@app.route("/shift_portal")
def shift_portal():
    return '''
    <html>
    <head><title>Shift Management</title></head>
    <body style="font-family:Segoe UI,Arial;background:#f8f9fa;padding:40px;">
        <h2>📋 Shift Management</h2>
        <div id="shiftTable"></div>
        <script>
        async function loadShifts(){
            let res = await fetch("/shifts");
            let data = await res.json();
            let html = "<table border='1' cellpadding='8'><tr><th>ID</th><th>Date</th><th>Time</th><th>Assigned To</th><th>Action</th></tr>";
            data.forEach(s=>{
                html += `<tr><td>${s.id}</td><td>${s.date}</td><td>${s.time}</td><td>${s.assigned_to||"Available"}</td>
                <td>${s.assigned_to ? 
                    `<button onclick="drop(${s.id}, '${s.assigned_to}')">Drop</button>` : 
                    `<button onclick="pick(${s.id})">Pick</button>`}</td></tr>`;
            });
            html += "</table>";
            document.getElementById("shiftTable").innerHTML = html;
        }

        async function pick(id){
            const user = prompt("Enter your name:");
            await fetch("/pick_shift",{method:"POST",headers:{"Content-Type":"application/json"},
                body:JSON.stringify({id,user})});
            loadShifts();
        }

        async function drop(id,user){
            await fetch("/drop_shift",{method:"POST",headers:{"Content-Type":"application/json"},
                body:JSON.stringify({id,user})});
            loadShifts();
        }

        loadShifts();
        </script>
        <br><button onclick="window.location.href='/'">← Back to Dashboard</button>
    </body>
    </html>
    '''


@app.route("/shifts", methods=["GET"])
def get_shifts():
    # We'll store shifts inside data.json under a 'shifts' key
    all_data = {"tickets": [], "runners": [], "shifts": []}
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    all_data["tickets"] = existing
                else:
                    all_data.update(existing)
        except Exception:
            pass
    return jsonify(all_data.get("shifts", []))

@app.route("/pick_shift", methods=["POST"])
def pick_shift():
    user_data = request.json
    user = user_data.get("user")
    shift_id = user_data.get("id")

    all_data = {"tickets": [], "runners": [], "shifts": []}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            existing = json.load(f)
            if isinstance(existing, list):
                all_data["tickets"] = existing
            else:
                all_data.update(existing)

    for s in all_data.get("shifts", []):
        if s["id"] == shift_id:
            if s["assigned_to"]:
                return jsonify({"error": "Shift already taken"}), 400
            s["assigned_to"] = user

    with open(DATA_FILE, "w") as f:
        json.dump(all_data, f, indent=2)

    return jsonify({"message": f"{user} picked shift {shift_id}"})

@app.route("/drop_shift", methods=["POST"])
def drop_shift():
    user_data = request.json
    user = user_data.get("user")
    shift_id = user_data.get("id")

    all_data = {"tickets": [], "runners": [], "shifts": []}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            existing = json.load(f)
            if isinstance(existing, list):
                all_data["tickets"] = existing
            else:
                all_data.update(existing)

    for s in all_data.get("shifts", []):
        if s["id"] == shift_id and s["assigned_to"] == user:
            s["assigned_to"] = None

    with open(DATA_FILE, "w") as f:
        json.dump(all_data, f, indent=2)

    return jsonify({"message": f"{user} dropped shift {shift_id}"})


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
        <br>
        <button onclick="window.location.href='/'">← Back</button>
    </body></html>
    '''

# -----------------------------
# Admin Login and Dashboard
# -----------------------------
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if check_admin_credentials(username, password):
            session["admin_logged_in"] = True
            session["admin_username"] = username
            return redirect("/admin")
        else:
            return "<h3 style='color:red;text-align:center;'>Invalid credentials</h3>" + login_form()
    return login_form()

    return '''
    <html><head><title>Admin Login</title></head>
    <body style="font-family:Segoe UI,Arial;background:#f8f9fa;text-align:center;padding:80px;">
        <h2>🔐 Admin Login</h2>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required><br><br>
            <input type="password" name="password" placeholder="Password" required><br><br>
            <button type="submit">Login</button>
        </form>
        <br>
        <button onclick="window.location.href='/'">← Back</button>
    </body></html>
    '''

@app.route("/admin_logout")
@login_required
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect("/")

@app.route("/admin")
@login_required
def admin_dashboard():
    return '''
    <html>
    <head><title>Admin Panel</title></head>
    <body style="font-family:Segoe UI,Arial;background:#f8f9fa;padding:40px;">
        <h2>🧑‍💼 Admin Dashboard</h2>
        <h3>📢 Post Announcement</h3>
        <form method="POST" action="/admin_announcement">
            <textarea name="message" placeholder="Type announcement..." rows="3" cols="50" required></textarea><br>
            <button type="submit">Post Announcement</button>
        </form>
        <hr>
        <h3>🗓️ Add or Remove Shifts</h3>
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
        <button onclick="window.location.href='/'">← Back to Dashboard</button>
        <button onclick="window.location.href='/admin_logout'">Logout</button>
    </body>
    </html>
    '''



# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 VALET OPERATIONS MANAGEMENT SYSTEM")
    print("="*60)
    print(f"Main Dashboard: http://127.0.0.1:5050")
    print(f"Runner Portal:  http://127.0.0.1:5050/runner_clockin")
    print(f"SMS Status:     {'✅ ENABLED' if SMS_ENABLED else '⚠️ DISABLED (check Vonage credentials)'}")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5050, debug=True)
