
from flask import Flask, render_template, request
import sqlite3
import cv2
import base64
import re
import numpy as np
from palm_recognition import get_palm_recognizer

app = Flask(__name__)

DB = "palm_pay.db"

# Initialize palm recognizer
palm_recognizer = get_palm_recognizer()

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            account_number TEXT UNIQUE NOT NULL,
            phone TEXT,
            address TEXT,
            account_type TEXT,
            pin TEXT NOT NULL,
            balance REAL DEFAULT 0 NOT NULL,
            hand_image BLOB,
            palm_features BLOB
        )
    ''')
    # Add palm_features column if it doesn't exist (for existing databases)
    try:
        c.execute("ALTER TABLE users ADD COLUMN palm_features BLOB")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_number TEXT NOT NULL,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            balance_after REAL NOT NULL,
            note TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    ''')
    conn.commit()
    conn.close()

def get_user(acc):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id, name, account_number, phone, address, account_type, pin, balance, hand_image, palm_features FROM users WHERE account_number=?", (acc,))
    row = c.fetchone()
    conn.close()
    return row

def capture_palm_image():
    """Capture palm image from camera - works in web server context"""
    import time
    
    cap = cv2.VideoCapture(0)
    hand_image = None
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Give camera a moment to stabilize
    time.sleep(0.5)
    
    # Capture image - reduced wait time for web requests
    # First, try to capture immediately, then wait for GUI if available
    import time
    start_time = time.time()
    capture_time = 3  # Reduced to 3 seconds for web requests
    frames_captured = 0
    last_progress_second = -1
    
    print("Capturing palm image... Please position your palm in front of the camera.")
    
    gui_available = True
    captured = False
    
    # Read a few frames to let camera stabilize
    for _ in range(5):
        cap.read()
    
    while not captured:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        frames_captured += 1
        elapsed = time.time() - start_time
        
        # Try to show window if GUI is available
        if gui_available:
            try:
                cv2.imshow("PalmPay - Press 's' to save palm image, 'q' to cancel", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    success, encoded = cv2.imencode('.png', frame)
                    if success:
                        hand_image = encoded.tobytes()
                        print(f"Palm image captured successfully (user pressed 's') - Size: {len(hand_image)} bytes")
                        captured = True
                        break
                elif key == ord('q'):
                    print("Palm capture cancelled (user pressed 'q')")
                    break
            except (cv2.error, Exception) as e:
                # No GUI available (web server context)
                gui_available = False
                print(f"No GUI available, will capture automatically after {capture_time} seconds")
        
        # If no GUI, capture automatically after delay
        if not gui_available:
            # Show progress every second
            current_second = int(elapsed)
            if current_second != last_progress_second and current_second > 0 and current_second < capture_time:
                last_progress_second = current_second
                remaining = int(capture_time - elapsed)
                if remaining > 0:
                    print(f"Capturing in {remaining} seconds... Please position your palm.")
            
            # Capture after timeout
            if elapsed >= capture_time:
                try:
                    success, encoded = cv2.imencode('.png', frame)
                    if success:
                        hand_image = encoded.tobytes()
                        print(f"Palm image captured automatically after {elapsed:.1f} seconds - Size: {len(hand_image)} bytes")
                        if hand_image and len(hand_image) > 0:
                            captured = True
                            break
                        else:
                            print("Warning: Image encoded but empty, retrying...")
                    else:
                        print("Warning: Failed to encode image, retrying...")
                except Exception as e:
                    print(f"Error encoding image: {e}")
                    # Try one more time
                    ret, frame = cap.read()
                    if ret:
                        success, encoded = cv2.imencode('.png', frame)
                        if success:
                            hand_image = encoded.tobytes()
                            captured = True
                    break
        
        # Safety timeout - capture after 8 seconds max
        if elapsed >= 8:
            print("Timeout reached, capturing image now")
            ret, frame = cap.read()
            if ret:
                success, encoded = cv2.imencode('.png', frame)
                if success:
                    hand_image = encoded.tobytes()
                    print(f"Captured at timeout - Size: {len(hand_image)} bytes")
                    captured = True
            break
        
        time.sleep(0.1)  # Small delay to avoid busy waiting
    
    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass  # Ignore if no windows to close
    
    if hand_image is None:
        if frames_captured == 0:
            print("Error: Could not read any frames from camera")
        else:
            print(f"Error: No image captured after {frames_captured} frames")
    else:
        print(f"✅ Successfully captured image: {len(hand_image)} bytes")
    
    return hand_image

def verify_palm_authentication(account_number, new_palm_image):
    """Verify palm authentication for a user"""
    user = get_user(account_number)
    if not user or not user[9]:  # user[9] is palm_features
        return False, 0.0, "Palm features not found. Please re-register."
    
    try:
        stored_features = user[9]  # palm_features BLOB
        is_verified, similarity = palm_recognizer.verify_palm(stored_features, new_palm_image)
        return is_verified, similarity, ""
    except Exception as e:
        return False, 0.0, f"Palm verification error: {str(e)}"

def authenticate(acc, pin):
    if pin is None:
        return False
    pin = str(pin).strip()
    if not re.fullmatch(r"\d{4}", pin):
        return False
    u = get_user(acc)
    if not u:
        return False
    return str(u[6]).strip() == pin

def set_balance(acc, new_bal):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("UPDATE users SET balance=? WHERE account_number=?", (new_bal, acc))
    conn.commit()
    conn.close()

def add_txn(acc, ttype, amount, balance_after, note=""):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO transactions (account_number, type, amount, balance_after, note) VALUES (?, ?, ?, ?, ?)",
        (acc, ttype, float(amount), float(balance_after), note)
    )
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html", message=None)
    name = request.form.get("name","").strip()
    account_number = request.form.get("account_number","").strip()
    phone = request.form.get("phone","").strip()
    address = request.form.get("address","").strip()
    account_type = request.form.get("account_type","").strip()
    pin = str(request.form.get("pin","")).strip()

    errors = []
    if not name or not account_number or not pin:
        errors.append("Name, Account Number and PIN are required.")
    if not re.fullmatch(r"\d{4}", pin):
        errors.append("PIN must be exactly 4 digits.")
    if errors:
        return render_template("register.html", message=" ".join(errors))

    # Use the capture_palm_image function for consistency
    hand_image = capture_palm_image()

    if hand_image is None:
        error_msg = "Hand image not captured. "
        error_msg += "Possible issues: "
        error_msg += "1) Camera permissions not granted (System Preferences → Security & Privacy → Camera), "
        error_msg += "2) Camera is being used by another app, "
        error_msg += "3) Camera is not connected. "
        error_msg += "Please check your camera settings and try again."
        return render_template("register.html", message=error_msg)

    # Extract palm features for future authentication
    try:
        palm_features = palm_recognizer.extract_features(hand_image)
        palm_features_bytes = palm_features.tobytes()
    except Exception as e:
        return render_template("register.html", message=f"Error extracting palm features: {str(e)}")

    try:
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute(
            "INSERT INTO users (name, account_number, phone, address, account_type, pin, hand_image, palm_features, balance) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (name, account_number, phone, address, account_type, pin, hand_image, palm_features_bytes)
        )
        conn.commit()
        conn.close()
        return render_template("register.html", message="Registration successful! Palm features extracted and stored. Use the navbar to Deposit, Withdraw or Transfer.")
    except sqlite3.IntegrityError:
        return render_template("register.html", message="Account number already exists. Try a different one.")

@app.route("/users")
def users():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id, name, account_number, balance, hand_image FROM users ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    processed = []
    for r in rows:
        img_b64 = base64.b64encode(r[4]).decode("utf-8") if r[4] else ""
        processed.append((r[0], r[1], r[2], r[3], img_b64))
    return render_template("users.html", users=processed)

@app.route("/deposit", methods=["GET", "POST"])
def deposit():
    if request.method == "GET":
        return render_template("deposit.html", message=None)
    acc = request.form.get("account_number","").strip()
    pin = request.form.get("pin","").strip()
    amt = request.form.get("amount","0").strip()

    user = get_user(acc)
    if not user:
        return render_template("deposit.html", message="Account not found. Please register first.")

    if not authenticate(acc, pin):
        return render_template("deposit.html", message="Invalid PIN.")

    # Palm authentication for UPI transaction
    print("Please position your palm for verification...")
    palm_image = capture_palm_image()
    if palm_image is None:
        return render_template("deposit.html", message="Palm image not captured. Please try again.")
    
    is_verified, similarity, error_msg = verify_palm_authentication(acc, palm_image)
    if not is_verified:
        return render_template("deposit.html", message=f"Palm authentication failed. Similarity: {similarity:.2%}. {error_msg} Please ensure you're using the same palm registered during signup.")

    try:
        amount = float(amt)
        if amount <= 0:
            return render_template("deposit.html", message="Amount must be greater than 0.")
    except ValueError:
        return render_template("deposit.html", message="Invalid amount.")

    new_balance = float(user[7]) + amount
    set_balance(acc, new_balance)
    add_txn(acc, "deposit", amount, new_balance, "Cash deposit - Palm verified")
    return render_template("deposit.html", message=f"Deposit successful! Palm verified (similarity: {similarity:.2%}). New balance: {new_balance:.2f}")

@app.route("/withdraw", methods=["GET", "POST"])
def withdraw():
    if request.method == "GET":
        return render_template("withdraw.html", message=None)
    acc = request.form.get("account_number","").strip()
    pin = request.form.get("pin","").strip()
    amt = request.form.get("amount","0").strip()

    user = get_user(acc)
    if not user:
        return render_template("withdraw.html", message="Account not found. Please register first.")

    if not authenticate(acc, pin):
        return render_template("withdraw.html", message="Invalid PIN.")

    # Palm authentication for UPI transaction
    print("Please position your palm for verification...")
    palm_image = capture_palm_image()
    if palm_image is None:
        return render_template("withdraw.html", message="Palm image not captured. Please try again.")
    
    is_verified, similarity, error_msg = verify_palm_authentication(acc, palm_image)
    if not is_verified:
        return render_template("withdraw.html", message=f"Palm authentication failed. Similarity: {similarity:.2%}. {error_msg} Please ensure you're using the same palm registered during signup.")

    try:
        amount = float(amt)
        if amount <= 0:
            return render_template("withdraw.html", message="Amount must be greater than 0.")
    except ValueError:
        return render_template("withdraw.html", message="Invalid amount.")

    if float(user[7]) < amount:
        return render_template("withdraw.html", message="Insufficient balance.")
    new_balance = float(user[7]) - amount
    set_balance(acc, new_balance)
    add_txn(acc, "withdraw", amount, new_balance, "Cash withdrawal - Palm verified")
    return render_template("withdraw.html", message=f"Withdrawal successful! Palm verified (similarity: {similarity:.2%}). New balance: {new_balance:.2f}")

@app.route("/transfer", methods=["GET", "POST"])
def transfer():
    if request.method == "GET":
        return render_template("transfer.html", message=None)
    from_acc = request.form.get("from_account","").strip()
    pin = request.form.get("pin","").strip()
    to_acc = request.form.get("to_account","").strip()
    amt = request.form.get("amount","0").strip()

    sender = get_user(from_acc)
    receiver = get_user(to_acc)
    if not sender:
        return render_template("transfer.html", message="Sender account not found.")
    if not receiver:
        return render_template("transfer.html", message="Receiver account not found.")
    if not authenticate(from_acc, pin):
        return render_template("transfer.html", message="Invalid PIN.")

    # Palm authentication for UPI transaction (sender only)
    print("Please position your palm for verification...")
    palm_image = capture_palm_image()
    if palm_image is None:
        return render_template("transfer.html", message="Palm image not captured. Please try again.")
    
    is_verified, similarity, error_msg = verify_palm_authentication(from_acc, palm_image)
    if not is_verified:
        return render_template("transfer.html", message=f"Palm authentication failed. Similarity: {similarity:.2%}. {error_msg} Please ensure you're using the same palm registered during signup.")

    try:
        amount = float(amt)
        if amount <= 0:
            return render_template("transfer.html", message="Amount must be greater than 0.")
    except ValueError:
        return render_template("transfer.html", message="Invalid amount.")

    if float(sender[7]) < amount:
        return render_template("transfer.html", message="Insufficient balance in sender account.")

    new_sender_bal = float(sender[7]) - amount
    set_balance(from_acc, new_sender_bal)
    add_txn(from_acc, "transfer_out", amount, new_sender_bal, f"To {to_acc} - Palm verified")

    new_receiver_bal = float(receiver[7]) + amount
    set_balance(to_acc, new_receiver_bal)
    add_txn(to_acc, "transfer_in", amount, new_receiver_bal, f"From {from_acc}")

    return render_template("transfer.html", message=f"Transfer successful! Palm verified (similarity: {similarity:.2%}). Transferred {amount:.2f} from {from_acc} to {to_acc}. Sender new balance: {new_sender_bal:.2f}")

@app.route("/balance", methods=["GET", "POST"])
def balance():
    if request.method == "GET":
        return render_template("balance.html", message=None)
    acc = request.form.get("account_number","").strip()
    pin = request.form.get("pin","").strip()

    user = get_user(acc)
    if not user:
        return render_template("balance.html", message="Account not found. Please register first.")
    if not authenticate(acc, pin):
        return render_template("balance.html", message="Invalid PIN.")

    return render_template("balance.html", message=f"Your current balance is: {float(user[7]):.2f}")

@app.route("/history", methods=["GET", "POST"])
def history():
    if request.method == "GET":
        return render_template("history.html", message=None, txns=[])
    acc = request.form.get("account_number","").strip()
    pin = request.form.get("pin","").strip()

    user = get_user(acc)
    if not user:
        return render_template("history.html", message="Account not found.", txns=[])
    if not authenticate(acc, pin):
        return render_template("history.html", message="Invalid PIN.", txns=[])

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''
        SELECT type, amount, balance_after, note, created_at
        FROM transactions
        WHERE account_number=?
        ORDER BY id DESC
        LIMIT 100
    ''', (acc,))
    rows = c.fetchall()
    conn.close()
    return render_template("history.html", message=f"Showing last {len(rows)} transactions for {acc}.", txns=rows)

if __name__ == "__main__":
    # Use port 5001 to avoid conflict with macOS AirPlay Receiver on port 5000
    app.run(debug=True, port=5001, host='0.0.0.0')
