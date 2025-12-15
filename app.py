from flask import Flask, render_template, request, jsonify
import sqlite3
import cv2
import base64
import re
import time
import numpy as np
import os
import joblib
from palm_recognition import get_palm_recognizer, _cv2_import_error
from cryptography.fernet import Fernet

app = Flask(__name__)

# ===== ENCRYPTION SETUP =====
KEY_FILE = ".db_key"
DB = "palm_pay.db"
DB_ENCRYPTED = "palm_pay.db.enc"

def ensure_encryption_key():
    """Generate or load encryption key."""
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, 'wb') as f:
            f.write(key)
        # Protect key file (read/write owner only)
        os.chmod(KEY_FILE, 0o600)
        print("[SECURITY] Generated encryption key (stored in .db_key)")
    
    with open(KEY_FILE, 'rb') as f:
        return f.read()

cipher_key = ensure_encryption_key()
cipher = Fernet(cipher_key)

def encrypt_database():
    """Encrypt database file."""
    if not os.path.exists(DB):
        return
    
    with open(DB, 'rb') as f:
        db_data = f.read()
    
    encrypted = cipher.encrypt(db_data)
    
    with open(DB_ENCRYPTED, 'wb') as f:
        f.write(encrypted)
    
    os.remove(DB)
    os.chmod(DB_ENCRYPTED, 0o600)
    print(f"[SECURITY] Encrypted database to {DB_ENCRYPTED}")

def decrypt_database():
    """Decrypt database file to temporary location for access."""
    if not os.path.exists(DB_ENCRYPTED):
        return
    
    with open(DB_ENCRYPTED, 'rb') as f:
        encrypted_data = f.read()
    
    try:
        decrypted = cipher.decrypt(encrypted_data)
        with open(DB, 'wb') as f:
            f.write(decrypted)
        os.chmod(DB, 0o600)
    except Exception as e:
        print(f"[ERROR] Failed to decrypt database: {e}")
        raise

# Decrypt on startup
decrypt_database()

# ===== DB FUNCTIONS (unchanged) =====
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

def capture_palm_image(timeout=1.0, frames=6, min_size_bytes=800):
    """Fast burst capture; returns PNG bytes or None."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CAM] Cannot open camera")
        return None
    try:
        time.sleep(0.12)
        best_frame = None
        best_score = -1.0
        start = time.time()
        captured = 0
        while captured < frames and (time.time() - start) < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
            captured += 1
            h, w = frame.shape[:2]
            if max(h, w) > 800:
                scale = 800 / max(h, w)
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
        if best_frame is None:
            return None
        ok, enc = cv2.imencode('.png', best_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            return None
        b = enc.tobytes()
        if len(b) < min_size_bytes:
            print(f"[CAPTURE] image too small {len(b)}")
            return None
        return b
    finally:
        cap.release()

def capture_and_average_features(n=3, timeout=0.6):
    """Fast: capture up to n usable frames, extract features, return (avg_features, representative_image_bytes)"""
    feats = []
    rep_img = None
    for attempt in range(n):
        img = capture_palm_image(timeout=timeout, frames=3, min_size_bytes=500)
        if not img:
            print(f"[CAPTURE] Attempt {attempt+1}/{n} failed")
            continue
        try:
            f = palm_recognizer.extract_features(img)
            feats.append(f)
            if rep_img is None:
                rep_img = img
            print(f"[CAPTURE] Attempt {attempt+1}/{n} success")
        except Exception as e:
            print(f"[CAPTURE] Extract failed: {e}")
            continue
    if not feats:
        return None, None
    avg = np.mean(np.vstack(feats), axis=0)
    return avg.astype(np.float32), rep_img

def verify_palm_authentication(account_number, new_palm_image, threshold=0.60):
    user = get_user(account_number)
    if not user:
        return False, "Account not found"
    stored = user[9]
    if not stored:
        return False, "No palm registered"
    try:
        ok, sim, err = palm_recognizer.verify_palm(stored, new_palm_image, threshold=threshold)
        if err:
            return False, err
        return bool(ok), f"Similarity: {sim:.2%}"
    except Exception as e:
        return False, str(e)

def authenticate(acc, pin):
    if not acc or not pin:
        return False
    if not re.fullmatch(r"\d{4}", str(pin)):
        return False
    u = get_user(acc)
    if not u:
        return False
    return str(u[6]).strip() == str(pin).strip()

def set_balance(acc, new_bal):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("UPDATE users SET balance=? WHERE account_number=?", (new_bal, acc))
    conn.commit()
    conn.close()

def add_txn(acc, ttype, amount, balance_after, note=""):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO transactions (account_number, type, amount, balance_after, note) VALUES (?, ?, ?, ?, ?)",
              (acc, ttype, float(amount), float(balance_after), note))
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "GET":
        return render_template("register.html", message=None)
    try:
        # capture + extract
        avg_feats, rep_img = capture_and_average_features(n=4, timeout=1.5)
        if avg_feats is None:
            return render_template("register.html", message="Capture failed. Ensure camera permission and good lighting.")
        
        name = request.form.get("name","").strip()
        account_number = request.form.get("account_number","").strip()
        phone = request.form.get("phone","").strip()
        address = request.form.get("address","").strip()
        account_type = request.form.get("account_type","").strip()
        pin = str(request.form.get("pin","")).strip()
        
        if not name or not account_number or not pin:
            return render_template("register.html", message="Name, account and PIN required")
        if not re.fullmatch(r"\d{4}", pin):
            return render_template("register.html", message="PIN must be 4 digits")
        
        try:
            conn = sqlite3.connect(DB)
            c = conn.cursor()
            c.execute("INSERT INTO users (name, account_number, phone, address, account_type, pin, hand_image, palm_features, balance) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                      (name, account_number, phone, address, account_type, pin, rep_img, avg_feats.tobytes()))
            conn.commit()
            conn.close()
            print(f"[REGISTER] User {account_number} registered. Retrain model with: python3 train.py")
            return render_template("register.html", message="Registration successful. Run 'python3 train.py' to improve accuracy.")
        except sqlite3.IntegrityError:
            return render_template("register.html", message="Account exists")
    except Exception as e:
        import traceback, sys
        traceback.print_exc()
        return render_template("register.html", message=f"Internal error: {str(e)}"), 500

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("signin.html", message=None)
    account_number = request.form.get("account_number","").strip()
    if not account_number:
        return render_template("signin.html", message="Account required")
    
    print("[SIGNIN] Capturing palm...")
    avg_feats, rep_img = capture_and_average_features(n=2, timeout=0.5)
    if avg_feats is None:
        return render_template("signin.html", message="Capture failed. Try again.")
    
    user = get_user(account_number)
    if not user:
        return render_template("signin.html", message="Account not found")
    
    ok, sim, err = palm_recognizer.verify_palm(user[9], rep_img, threshold=0.58)
    if err:
        return render_template("signin.html", message=f"Auth error: {err}")
    if not ok:
        return render_template("signin.html", message=f"Auth failed. Similarity: {sim:.2%}")
    return render_template("signin.html", message=f"Signed in. Similarity: {sim:.2%}")

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
        return render_template("deposit.html", message="Account not found")
    if not authenticate(acc, pin):
        return render_template("deposit.html", message="Invalid PIN")
    
    avg_feats, rep_img = capture_and_average_features(n=2, timeout=0.5)
    if avg_feats is None:
        return render_template("deposit.html", message="Capture failed")
    ok, msg = verify_palm_authentication(acc, rep_img, threshold=0.58)
    if not ok:
        return render_template("deposit.html", message=f"Palm auth failed. {msg}")
    
    try:
        amount = float(amt)
        if amount <= 0:
            return render_template("deposit.html", message="Amount > 0 required")
    except ValueError:
        return render_template("deposit.html", message="Invalid amount")
    new_balance = float(user[7]) + amount
    set_balance(acc, new_balance)
    add_txn(acc, "deposit", amount, new_balance, "Palm verified deposit")
    return render_template("deposit.html", message=f"Deposit successful. New balance: {new_balance:.2f}")

@app.route("/withdraw", methods=["GET", "POST"])
def withdraw():
    if request.method == "GET":
        return render_template("withdraw.html", message=None)
    acc = request.form.get("account_number","").strip()
    pin = request.form.get("pin","").strip()
    amt = request.form.get("amount","0").strip()
    user = get_user(acc)
    if not user:
        return render_template("withdraw.html", message="Account not found")
    if not authenticate(acc, pin):
        return render_template("withdraw.html", message="Invalid PIN")
    
    avg_feats, rep_img = capture_and_average_features(n=2, timeout=0.5)
    if avg_feats is None:
        return render_template("withdraw.html", message="Capture failed")
    ok, msg = verify_palm_authentication(acc, rep_img, threshold=0.58)
    if not ok:
        return render_template("withdraw.html", message=f"Palm auth failed. {msg}")
    
    try:
        amount = float(amt)
        if amount <= 0:
            return render_template("withdraw.html", message="Amount > 0 required")
    except ValueError:
        return render_template("withdraw.html", message="Invalid amount")
    if float(user[7]) < amount:
        return render_template("withdraw.html", message="Insufficient balance")
    new_balance = float(user[7]) - amount
    set_balance(acc, new_balance)
    add_txn(acc, "withdraw", amount, new_balance, "Palm verified withdrawal")
    return render_template("withdraw.html", message=f"Withdrawal successful. New balance: {new_balance:.2f}")

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
        return render_template("transfer.html", message="Sender not found")
    if not receiver:
        return render_template("transfer.html", message="Receiver not found")
    if not authenticate(from_acc, pin):
        return render_template("transfer.html", message="Invalid PIN")
    
    avg_feats, rep_img = capture_and_average_features(n=2, timeout=0.5)
    if avg_feats is None:
        return render_template("transfer.html", message="Capture failed")
    ok, msg = verify_palm_authentication(from_acc, rep_img, threshold=0.58)
    if not ok:
        return render_template("transfer.html", message=f"Palm auth failed. {msg}")
    
    try:
        amount = float(amt)
        if amount <= 0:
            return render_template("transfer.html", message="Amount > 0 required")
    except ValueError:
        return render_template("transfer.html", message="Invalid amount")
    if float(sender[7]) < amount:
        return render_template("transfer.html", message="Insufficient funds")
    new_sender_bal = float(sender[7]) - amount
    set_balance(from_acc, new_sender_bal)
    add_txn(from_acc, "transfer_out", amount, new_sender_bal, f"To {to_acc}")
    new_receiver_bal = float(receiver[7]) + amount
    set_balance(to_acc, new_receiver_bal)
    add_txn(to_acc, "transfer_in", amount, new_receiver_bal, f"From {from_acc}")
    return render_template("transfer.html", message=f"Transfer successful. {msg}")

@app.route("/balance", methods=["GET", "POST"])
def balance():
    if request.method == "GET":
        return render_template("balance.html", message=None)
    acc = request.form.get("account_number","").strip()
    pin = request.form.get("pin","").strip()
    user = get_user(acc)
    if not user:
        return render_template("balance.html", message="Account not found")
    if not authenticate(acc, pin):
        return render_template("balance.html", message="Invalid PIN")
    return render_template("balance.html", message=f"Your balance: {float(user[7]):.2f}")

@app.route("/history", methods=["GET", "POST"])
def history():
    if request.method == "GET":
        return render_template("history.html", message=None, txns=[])
    acc = request.form.get("account_number","").strip()
    pin = request.form.get("pin","").strip()
    if not authenticate(acc, pin):
        return render_template("history.html", message="Invalid PIN", txns=[])
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('SELECT type, amount, balance_after, note, created_at FROM transactions WHERE account_number=? ORDER BY id DESC LIMIT 100', (acc,))
    rows = c.fetchall()
    conn.close()
    return render_template("history.html", message=f"Showing last {len(rows)} transactions", txns=rows)

@app.route("/debug_capture")
def debug_capture():
    img = capture_palm_image(timeout=1.0, frames=8, min_size_bytes=500)
    if not img:
        return "Capture failed"
    with open("capture_debug.png", "wb") as f:
        f.write(img)
    return "Saved capture_debug.png"

_palm_recognizer = None

def get_recognizer_or_error():
    global _palm_recognizer
    if _palm_recognizer is None:
        try:
            _palm_recognizer = get_palm_recognizer()
        except ImportError as e:
            app.logger.error("Palm recognizer import error: %s", e)
            return None, str(e)
    return _palm_recognizer, None

@app.route("/debug/palm", methods=["POST"])
def debug_palm():
    file = request.files.get("image")
    if not file:
        return jsonify({"ok": False, "error": "No image file (field 'image')"}), 400
    rec, err = get_recognizer_or_error()
    if rec is None:
        return jsonify({"ok": False, "error": f"Recognizer unavailable: {err}"}), 500
    info = rec.debug_summary(file.read())
    if not info.get("ok"):
        return jsonify({"ok": False, "error": info.get("error", "unknown")}), 400
    return jsonify({"ok": True, "info": info}), 200

@app.teardown_appcontext
def cleanup(exception=None):
    """Encrypt database on shutdown."""
    if os.path.exists(DB):
        encrypt_database()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
