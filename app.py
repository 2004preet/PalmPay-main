
from flask import Flask, render_template, request, jsonify
import sqlite3
import cv2
import base64
import re
import numpy as np
import threading
import time
import logging
import json
from stripe_config import (
    STRIPE_ENABLED, STRIPE_PUBLISHABLE_KEY, STRIPE_CURRENCY,
    create_payment_intent, verify_payment_intent, construct_webhook_event
)

# ── Palm recognizer — uses trained v7 model (preferred), falls back to legacy
try:
    from train_palm_pro import PalmRecognizerPro
    HAS_PRO = True
except ImportError:
    HAS_PRO = False

try:
    from advanced_train_model import PalmVerifier, assess_capture_quality
    HAS_LEGACY = True
except ImportError:
    HAS_LEGACY = False

    # Stub quality check
    def assess_capture_quality(img_bgr):
        return True, {"sharpness": 100, "brightness": 128, "contrast": 50, "issues": []}

# Configure logging for confidence tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
auth_logger = logging.getLogger("palm_auth")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024   # 50 MB — needed for 3 base64 palm frames

DB = "palm_pay.db"

# ── Initialize palm recognizer (v7 trained model > legacy fallback)
def _init_recognizer():
    import os
    # Prefer the new trained model
    if HAS_PRO and os.path.exists("palm_model_v7.h5"):
        print("✓ Using PalmRecognizerPro (trained v7 model)")
        return PalmRecognizerPro("palm_model_v7.h5", threshold=0.75)
    # Fallback to legacy
    if HAS_LEGACY and os.path.exists("palm_feature_extractor_v5_pro.h5"):
        print("⚠ Falling back to legacy PalmVerifier")
        try:
            v = PalmVerifier("palm_feature_extractor_v5_pro.h5", threshold=0.82)
            return v
        except Exception as e:
            print(f"  Legacy load failed: {e}")
    # Stub
    print("⚠ No model available — palm features will be zeros")
    class StubRecognizer:
        threshold = 0.75
        def enroll(self, img): return np.zeros(256, dtype=np.float32)
        def verify(self, img, emb): return False, 0.0
    return StubRecognizer()

palm_recognizer = _init_recognizer()

# Global variable to track retraining status
retraining_status = {"in_progress": False, "last_retrained": None, "message": ""}

def retrain_model_background():
    """Background function to retrain the model with all stored palm data"""
    global retraining_status
    try:
        retraining_status["in_progress"] = True
        retraining_status["message"] = "Starting model retraining..."

        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("SELECT hand_image FROM users WHERE hand_image IS NOT NULL")
        rows = c.fetchall()
        conn.close()

        total_images = len(rows)
        if total_images < 5:
            retraining_status["message"] = (
                f"Not enough data for retraining "
                f"(need at least 5, have {total_images})"
            )
            retraining_status["in_progress"] = False
            return

        # Adaptive training parameters
        if total_images <= 10:
            epochs, batch_size, augment_factor = 25, 4, 8
        elif total_images <= 50:
            epochs, batch_size, augment_factor = 20, 8, 6
        elif total_images <= 100:
            epochs, batch_size, augment_factor = 15, 12, 4
        elif total_images <= 500:
            epochs, batch_size, augment_factor = 12, 16, 3
        else:
            epochs, batch_size, augment_factor = 8, 24, 2

        retraining_status["message"] = (
            f"Adaptive training: {epochs} epochs, "
            f"batch {batch_size}, {total_images * augment_factor} samples"
        )

        import os, tempfile, shutil
        from advanced_train_model import create_pro_model, load_images_with_labels, create_training_data
        from tensorflow import keras

        temp_dir  = tempfile.mkdtemp()
        train_dir = os.path.join(temp_dir, "Train")
        os.makedirs(train_dir)

        for i, (img_blob,) in enumerate(rows):
            try:
                nparr = np.frombuffer(img_blob, np.uint8)
                img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imwrite(os.path.join(train_dir, f"IMG_{i+1:04d}.jpg"), img)
            except Exception as e:
                print(f"Error saving image {i}: {e}")

        retraining_status["message"] = (
            f"Saved {len(os.listdir(train_dir))} images to temp directory"
        )

        import advanced_train_model as atm
        orig_train = atm.TRAIN_DIR; orig_valid = atm.VALID_DIR
        orig_model = atm.MODEL_SAVE_PATH
        orig_ep    = atm.EPOCHS;    orig_bs    = atm.BATCH_SIZE

        atm.TRAIN_DIR      = train_dir
        atm.VALID_DIR      = None
        atm.MODEL_SAVE_PATH = "palm_feature_extractor_advanced.h5"
        atm.EPOCHS         = epochs
        atm.BATCH_SIZE     = batch_size

        try:
            import types
            orig_ctd = atm.create_training_data
            def fast_ctd(imgs, lbls, augment_factor=augment_factor):
                return orig_ctd(imgs, lbls, augment_factor)
            atm.create_training_data = fast_ctd

            retraining_status["message"] = (
                f"🚀 Fast training: {epochs} epochs, batch {batch_size}..."
            )
            atm.train_pro_model()
            retraining_status["message"] = (
                f"✅ Retraining complete! Trained on {total_images} users "
                f"with {epochs} epochs."
            )
            retraining_status["last_retrained"] = time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            retraining_status["message"] = f"Retraining failed: {str(e)}"
        finally:
            atm.TRAIN_DIR          = orig_train
            atm.VALID_DIR          = orig_valid
            atm.MODEL_SAVE_PATH    = orig_model
            atm.EPOCHS             = orig_ep
            atm.BATCH_SIZE         = orig_bs
            atm.create_training_data = orig_ctd
            shutil.rmtree(temp_dir, ignore_errors=True)

        retraining_status["in_progress"] = False

    except Exception as e:
        retraining_status["message"] = f"Retraining failed: {str(e)}"
        retraining_status["in_progress"] = False


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
    try:
        c.execute("ALTER TABLE users ADD COLUMN palm_features BLOB")
    except sqlite3.OperationalError:
        pass  # Already exists

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
    c.execute(
        "SELECT id, name, account_number, phone, address, account_type, "
        "pin, balance, hand_image, palm_features FROM users WHERE account_number=?",
        (acc,)
    )
    row = c.fetchone()
    conn.close()
    return row


def decode_base64_image(b64_string):
    """Decode a base64 image string into raw JPEG bytes at high quality."""
    try:
        if not b64_string:
            return None
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        img_data = base64.b64decode(b64_string)
        nparr    = np.frombuffer(img_data, np.uint8)
        img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        # ← JPEG quality raised from 95 to 98 for maximum detail preservation
        success, encoded = cv2.imencode(".jpg", img,
                                         [cv2.IMWRITE_JPEG_QUALITY, 98])
        return encoded.tobytes() if success else None
    except Exception as e:
        print(f"decode_base64_image error: {e}")
        return None


def collect_palm_images(form):
    """
    Collect up to 5 palm images from a form submission (palm_image_1..5).
    Returns a list of decoded image bytes (non-empty).
    """
    images = []
    for i in range(1, 6):   # ← expanded from 3 to 5 frames
        b64 = form.get(f"palm_image_{i}", "")
        if b64:
            img = decode_base64_image(b64)
            if img is not None:
                images.append(img)
    return images


def verify_palm_multi(account_number, palm_images):
    """
    Verify palm against stored features using multi-frame ensemble.
    Returns (is_verified: bool, similarity: float, error_msg: str).
    Includes confidence logging for security audit trail.
    """
    user = get_user(account_number)
    if not user or not user[9]:
        auth_logger.warning(f"VERIFY FAIL: account={account_number} reason=no_features")
        return False, 0.0, "Palm features not found. Please re-register."
    try:
        stored_features = user[9]

        # Quality-gate each frame before verification
        quality_pass = 0
        quality_fail = 0
        for img_bytes in palm_images:
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                passed, qinfo = assess_capture_quality(img)
                if passed:
                    quality_pass += 1
                else:
                    quality_fail += 1
                    auth_logger.info(f"  Quality fail: {qinfo}")

        if quality_pass == 0:
            auth_logger.warning(f"VERIFY FAIL: account={account_number} "
                                f"reason=all_frames_low_quality")
            return False, 0.0, ("All palm images failed quality check. "
                                "Ensure good lighting and hold palm steady.")

        sims = []
        for img_bytes in palm_images:
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            match, sim = palm_recognizer.verify(img, np.frombuffer(stored_features, dtype=np.float32))
            sims.append(sim)

        similarity = max(sims) if sims else 0.0
        is_verified = similarity >= palm_recognizer.threshold

        # Confidence logging
        auth_logger.info(
            f"VERIFY {'PASS' if is_verified else 'FAIL'}: "
            f"account={account_number}  sim={similarity:.4f}  "
            f"frames={len(palm_images)}  quality_ok={quality_pass}/{len(palm_images)}  "
            f"threshold={palm_recognizer.threshold}"
        )

        return is_verified, similarity, ""
    except Exception as e:
        auth_logger.error(f"VERIFY ERROR: account={account_number} error={e}")
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
    c.execute("UPDATE users SET balance=? WHERE account_number=?",
              (new_bal, acc))
    conn.commit()
    conn.close()


def add_txn(acc, ttype, amount, balance_after, note=""):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO transactions "
        "(account_number, type, amount, balance_after, note) VALUES (?,?,?,?,?)",
        (acc, ttype, float(amount), float(balance_after), note)
    )
    conn.commit()
    conn.close()


init_db()


@app.route("/")
def home():
    return render_template("home.html")


# ── API: real-time account lookup for transfer page ───────────────────────────
@app.route("/api/check_account")
def check_account():
    acc = request.args.get("account_number", "").strip()
    if not acc:
        return jsonify({"found": False})
    user = get_user(acc)
    if not user:
        return jsonify({"found": False})
    return jsonify({
        "found":        True,
        "name":         user[1],
        "account_type": user[5] or "Standard",
    })


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html", message=None)

    name           = request.form.get("name", "").strip()
    account_number = request.form.get("account_number", "").strip()
    phone          = request.form.get("phone", "").strip()
    address        = request.form.get("address", "").strip()
    account_type   = request.form.get("account_type", "").strip()
    pin            = str(request.form.get("pin", "")).strip()

    errors = []
    if not name or not account_number or not pin:
        errors.append("Name, Account Number and PIN are required.")
    if not re.fullmatch(r"\d{4}", pin):
        errors.append("PIN must be exactly 4 digits.")
    if errors:
        return render_template("register.html", message=" ".join(errors))

    # Collect up to 3 palm images for registration
    hand_images = collect_palm_images(request.form)

    if not hand_images:
        return render_template("register.html",
            message="❌ No palm images captured. Please allow camera access "
                    "and keep your hand steady during registration.")

    # Quality gate — reject frames that fail quality check
    quality_results = []
    good_images = []
    for img_bytes in hand_images:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            passed, qinfo = assess_capture_quality(img)
            quality_results.append(qinfo)
            if passed:
                good_images.append(img_bytes)
            else:
                print(f"  ⚠ Registration frame rejected: {qinfo}")

    if len(good_images) < 2:
        failed_info = ", ".join(
            f"frame {i+1}: sharp={q['sharpness']:.0f} bright={q['brightness']:.0f}"
            for i, q in enumerate(quality_results)
        )
        return render_template("register.html",
            message=(f"❌ Too few quality frames ({len(good_images)}/{len(hand_images)} passed). "
                     f"Details: {failed_info}. "
                     f"Ensure good lighting, hold palm steady and flat."))

    try:
        feature_list = []
        for img_bytes in good_images:
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            emb = palm_recognizer.enroll(img)
            feature_list.append(emb)

        palm_features = np.mean(feature_list, axis=0)
        palm_features = palm_features / (np.linalg.norm(palm_features) + 1e-9)
        palm_feats_b  = palm_features.tobytes()
        hand_image    = good_images[0]
    except Exception as e:
        return render_template("register.html",
            message=f"Error extracting palm features: {str(e)}")

    # Cross-user duplicate check — prevent same palm under different accounts
    try:
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("SELECT account_number, palm_features FROM users WHERE palm_features IS NOT NULL")
        existing_users = c.fetchall()
        conn.close()

        existing_features = [(row[0], row[1]) for row in existing_users if row[1]]
        for existing_acc, existing_feat in existing_features:
            existing_emb = np.frombuffer(existing_feat, dtype=np.float32)
            sim = float(np.dot(palm_features, existing_emb))
            if sim >= 0.75:  # Cross-user threshold
                auth_logger.warning(
                    f"REGISTER BLOCKED: new_acc={account_number} "
                    f"matches existing_acc={existing_acc} sim={sim:.4f}"
                )
                return render_template("register.html",
                    message=(f"❌ This palm is too similar to an existing account "
                             f"(similarity: {sim:.1%}). Each palm can only be "
                             f"registered once. If this is an error, contact support."))
    except Exception as e:
        print(f"Cross-user check warning: {e}")

    try:
        conn = sqlite3.connect(DB)
        c    = conn.cursor()
        c.execute(
            "INSERT INTO users "
            "(name, account_number, phone, address, account_type, pin, "
            "hand_image, palm_features, balance) VALUES (?,?,?,?,?,?,?,?,0)",
            (name, account_number, phone, address, account_type, pin,
             hand_image, palm_feats_b)
        )
        conn.commit()

        c.execute("SELECT COUNT(*) FROM users WHERE hand_image IS NOT NULL")
        total_users = c.fetchone()[0]
        conn.close()

        auth_logger.info(
            f"REGISTER OK: account={account_number} name={name} "
            f"frames={len(good_images)}/{len(hand_images)} quality_filtered"
        )

        # Auto-retrain if enough data and not already running
        should_retrain = False
        if total_users >= 5 and not retraining_status["in_progress"]:
            should_retrain = True
            if retraining_status["last_retrained"]:
                from datetime import datetime
                last = datetime.strptime(
                    retraining_status["last_retrained"], "%Y-%m-%d %H:%M:%S")
                if (datetime.now() - last).total_seconds() < 3600:
                    should_retrain = False

        if should_retrain:
            t = threading.Thread(target=retrain_model_background)
            t.daemon = True
            t.start()
            msg = (f"✅ Registration successful! {len(good_images)} high-quality palm "
                   f"frames captured (out of {len(hand_images)} total). "
                   f"Background model retrain started.")
        else:
            msg = (f"✅ Registration successful! {len(good_images)} high-quality palm "
                   f"frames captured and features extracted.")

        return render_template("register.html", message=msg)

    except sqlite3.IntegrityError:
        return render_template("register.html",
            message="Account number already exists. Try a different one.")


@app.route("/users")
def users():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id, name, account_number, balance, hand_image "
              "FROM users ORDER BY id DESC")
    rows = conn.cursor().fetchall() if False else c.fetchall()
    conn.close()
    processed = []
    for r in rows:
        img_b64 = base64.b64encode(r[4]).decode("utf-8") if r[4] else ""
        processed.append((r[0], r[1], r[2], r[3], img_b64))
    return render_template("users.html", users=processed)


@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    global retraining_status
    if request.method == "GET":
        return render_template("retrain.html", status=retraining_status)
    if retraining_status["in_progress"]:
        return render_template("retrain.html", status=retraining_status,
                               message="Retraining already in progress")
    t = threading.Thread(target=retrain_model_background)
    t.daemon = True
    t.start()
    return render_template("retrain.html", status=retraining_status,
                           message="Retraining started in background")


# ── Deposit ───────────────────────────────────────────────────────────────────
@app.route("/deposit", methods=["GET", "POST"])
def deposit():
    ctx = {"stripe_enabled": STRIPE_ENABLED, "stripe_pk": STRIPE_PUBLISHABLE_KEY}

    if request.method == "GET":
        return render_template("deposit.html", message=None, **ctx)

    acc = request.form.get("account_number", "").strip()
    pin = request.form.get("pin", "").strip()
    amt = request.form.get("amount", "0").strip()

    try:
        user = get_user(acc)
        if not user:
            return render_template("deposit.html",
                message="❌ Account not found. Please register first.", **ctx)

        if not authenticate(acc, pin):
            return render_template("deposit.html",
                message="❌ Invalid PIN. Please try again.", **ctx)

        palm_images = collect_palm_images(request.form)
        if not palm_images:
            return render_template("deposit.html",
                message="❌ Palm image not captured. Allow camera access and try again.", **ctx)

        is_verified, similarity, error_msg = verify_palm_multi(acc, palm_images)
        if not is_verified:
            tip = ("Tip: ensure good lighting and hold your palm flat and "
                   "steady in front of the camera.")
            return render_template("deposit.html",
                message=(f"❌ Palm authentication failed "
                         f"(similarity: {similarity:.1%}). {tip}"), **ctx)

        try:
            amount = float(amt)
            if amount <= 0:
                return render_template("deposit.html",
                    message="❌ Amount must be greater than ₹0.", **ctx)
        except ValueError:
            return render_template("deposit.html",
                message="❌ Invalid amount format.", **ctx)

        # ── Stripe-enabled flow: create PaymentIntent, show card form ─────
        if STRIPE_ENABLED:
            result = create_payment_intent(
                amount, acc,
                description=f"PalmPay deposit — {user[1]} ({acc})"
            )
            if "error" in result:
                return render_template("deposit.html",
                    message=f"❌ Payment error: {result['error']}", **ctx)

            auth_logger.info(
                f"STRIPE INTENT: account={acc} amount={amount} "
                f"pi={result['payment_intent_id']}"
            )

            return render_template("deposit.html",
                message=None,
                show_stripe_form=True,
                client_secret=result["client_secret"],
                payment_intent_id=result["payment_intent_id"],
                deposit_amount=amount,
                deposit_account=acc,
                palm_verified=True,
                palm_similarity=f"{similarity:.1%}",
                **ctx)

        # ── Non-Stripe fallback: direct balance credit ────────────────────
        new_balance = float(user[7]) + amount
        set_balance(acc, new_balance)
        add_txn(acc, "deposit", amount, new_balance,
                f"Cash deposit — Palm verified ({len(palm_images)} frames)")
        return render_template("deposit.html",
            message=(f"✅ Deposit successful! Palm verified "
                     f"(match: {similarity:.1%}, {len(palm_images)} frames). "
                     f"New balance: ₹{new_balance:,.2f}"), **ctx)

    except Exception as e:
        print(f"deposit error: {e}")
        return render_template("deposit.html",
            message=f"❌ Error processing deposit: {str(e)}", **ctx)


# ── Withdraw ──────────────────────────────────────────────────────────────────
@app.route("/withdraw", methods=["GET", "POST"])
def withdraw():
    if request.method == "GET":
        return render_template("withdraw.html", message=None)

    acc = request.form.get("account_number", "").strip()
    pin = request.form.get("pin", "").strip()
    amt = request.form.get("amount", "0").strip()

    try:
        user = get_user(acc)
        if not user:
            return render_template("withdraw.html",
                message="❌ Account not found. Please register first.")

        if not authenticate(acc, pin):
            return render_template("withdraw.html",
                message="❌ Invalid PIN. Please try again.")

        palm_images = collect_palm_images(request.form)
        if not palm_images:
            return render_template("withdraw.html",
                message="❌ Palm image not captured. Allow camera access and try again.")

        is_verified, similarity, error_msg = verify_palm_multi(acc, palm_images)
        if not is_verified:
            tip = ("Tip: ensure good lighting and hold your palm flat and "
                   "steady in front of the camera.")
            return render_template("withdraw.html",
                message=(f"❌ Palm authentication failed "
                         f"(similarity: {similarity:.1%}). {tip}"))

        try:
            amount = float(amt)
            if amount <= 0:
                return render_template("withdraw.html",
                    message="❌ Amount must be greater than ₹0.")
            if amount > float(user[7]):
                return render_template("withdraw.html",
                    message=f"❌ Insufficient funds. Available: ₹{user[7]:,.2f}")
        except ValueError:
            return render_template("withdraw.html",
                message="❌ Invalid amount format.")

        new_balance = float(user[7]) - amount
        set_balance(acc, new_balance)
        add_txn(acc, "withdraw", amount, new_balance,
                f"Cash withdrawal — Palm verified ({len(palm_images)} frames)")
        return render_template("withdraw.html",
            message=(f"✅ Withdrawal successful! Palm verified "
                     f"(match: {similarity:.1%}, {len(palm_images)} frames). "
                     f"New balance: ₹{new_balance:,.2f}"))

    except Exception as e:
        print(f"withdraw error: {e}")
        return render_template("withdraw.html",
            message=f"❌ Error processing withdrawal: {str(e)}")


# ── Transfer ──────────────────────────────────────────────────────────────────
@app.route("/transfer", methods=["GET", "POST"])
def transfer():
    if request.method == "GET":
        return render_template("transfer.html", message=None)

    from_acc = request.form.get("from_account", "").strip()
    pin      = request.form.get("pin", "").strip()
    to_acc   = request.form.get("to_account", "").strip()
    amt      = request.form.get("amount", "0").strip()

    sender   = get_user(from_acc)
    receiver = get_user(to_acc)
    if not sender:
        return render_template("transfer.html",
            message="❌ Sender account not found.")
    if not receiver:
        return render_template("transfer.html",
            message="❌ Receiver account not found.")
    if not authenticate(from_acc, pin):
        return render_template("transfer.html",
            message="❌ Invalid PIN.")

    palm_images = collect_palm_images(request.form)
    if not palm_images:
        return render_template("transfer.html",
            message="❌ Palm not captured. Allow camera access and try again.")

    is_verified, similarity, error_msg = verify_palm_multi(from_acc, palm_images)
    if not is_verified:
        tip = ("Tip: ensure good lighting and hold your palm flat and steady. "
               "Use the same palm you registered with.")
        return render_template("transfer.html",
            message=(f"❌ Palm authentication failed "
                     f"(similarity: {similarity:.1%}). {tip}"))

    try:
        amount = float(amt)
        if amount <= 0:
            return render_template("transfer.html",
                message="❌ Amount must be greater than ₹0.")
    except ValueError:
        return render_template("transfer.html",
            message="❌ Invalid amount.")

    if float(sender[7]) < amount:
        return render_template("transfer.html",
            message=f"❌ Insufficient balance. Available: ₹{sender[7]:,.2f}")

    new_sender_bal   = float(sender[7])   - amount
    new_receiver_bal = float(receiver[7]) + amount
    set_balance(from_acc, new_sender_bal)
    set_balance(to_acc,   new_receiver_bal)
    add_txn(from_acc, "transfer_out", amount, new_sender_bal,
            f"To {to_acc} — Palm verified ({len(palm_images)} frames)")
    add_txn(to_acc, "transfer_in", amount, new_receiver_bal,
            f"From {from_acc} — Palm verified")

    return render_template("transfer.html",
        message=(f"✅ Transfer successful! Palm verified "
                 f"(match: {similarity:.1%}, {len(palm_images)} frames). "
                 f"₹{amount:,.2f} sent to {receiver[1]} ({to_acc}). "
                 f"Your new balance: ₹{new_sender_bal:,.2f}"))


# ── Balance ───────────────────────────────────────────────────────────────────
@app.route("/balance", methods=["GET", "POST"])
def balance():
    if request.method == "GET":
        return render_template("balance.html", message=None)

    acc = request.form.get("account_number", "").strip()
    pin = request.form.get("pin", "").strip()
    user = get_user(acc)
    if not user:
        return render_template("balance.html",
            message="Account not found. Please register first.")
    if not authenticate(acc, pin):
        return render_template("balance.html", message="Invalid PIN.")
    return render_template("balance.html",
        message=f"Your current balance is: ₹{float(user[7]):,.2f}")


# ── History ───────────────────────────────────────────────────────────────────
@app.route("/history", methods=["GET", "POST"])
def history():
    if request.method == "GET":
        return render_template("history.html", message=None, txns=[])

    acc = request.form.get("account_number", "").strip()
    pin = request.form.get("pin", "").strip()
    user = get_user(acc)
    if not user:
        return render_template("history.html",
            message="Account not found.", txns=[])
    if not authenticate(acc, pin):
        return render_template("history.html",
            message="Invalid PIN.", txns=[])

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "SELECT type, amount, balance_after, note, created_at "
        "FROM transactions WHERE account_number=? "
        "ORDER BY id DESC LIMIT 100",
        (acc,)
    )
    rows = c.fetchall()
    conn.close()
    return render_template("history.html",
        message=f"Showing last {len(rows)} transactions for {acc}.",
        txns=rows)



# ── Stripe: Confirm Deposit API ─────────────────────────────────────────────
@app.route("/api/stripe/confirm-deposit", methods=["POST"])
def stripe_confirm_deposit():
    """
    Called by frontend after Stripe.confirmCardPayment() succeeds.
    Verifies the PaymentIntent on server side, then credits balance.
    """
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data received"}), 400

    pi_id   = data.get("payment_intent_id", "")
    acc     = data.get("account_number", "")
    amount  = data.get("amount", 0)

    if not pi_id or not acc:
        return jsonify({"success": False, "error": "Missing payment_intent_id or account"}), 400

    # Verify on Stripe's side
    succeeded, info = verify_payment_intent(pi_id)
    if not succeeded:
        auth_logger.warning(f"STRIPE VERIFY FAIL: pi={pi_id} account={acc} info={info}")
        return jsonify({"success": False,
                        "error": f"Payment not confirmed: {info.get('status', 'unknown')}"}), 400

    # Get user and credit balance
    user = get_user(acc)
    if not user:
        return jsonify({"success": False, "error": "Account not found"}), 404

    try:
        amount = float(amount)
        new_balance = float(user[7]) + amount
        set_balance(acc, new_balance)
        add_txn(acc, "deposit", amount, new_balance,
                f"Stripe deposit — PI: {pi_id[:20]}...")

        auth_logger.info(
            f"STRIPE DEPOSIT OK: account={acc} amount={amount} "
            f"pi={pi_id} new_balance={new_balance}"
        )

        return jsonify({
            "success": True,
            "new_balance": new_balance,
            "payment_intent_id": pi_id,
            "message": f"✅ ₹{amount:,.2f} deposited via Stripe. New balance: ₹{new_balance:,.2f}"
        })
    except Exception as e:
        auth_logger.error(f"STRIPE DEPOSIT ERROR: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Stripe: Webhook ───────────────────────────────────────────────────────
@app.route("/api/stripe/webhook", methods=["POST"])
def stripe_webhook():
    """
    Stripe webhook endpoint for backup payment confirmation.
    Handles payment_intent.succeeded events.
    """
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get("Stripe-Signature", "")

    event, error = construct_webhook_event(payload, sig_header)
    if error:
        auth_logger.warning(f"STRIPE WEBHOOK ERROR: {error}")
        return jsonify({"error": error}), 400

    if event["type"] == "payment_intent.succeeded":
        intent = event["data"]["object"]
        acc = intent.get("metadata", {}).get("account_number", "")
        amount_paise = intent.get("amount", 0)
        amount_rupees = amount_paise / 100.0

        auth_logger.info(
            f"STRIPE WEBHOOK: payment_intent.succeeded "
            f"pi={intent['id']} account={acc} amount={amount_rupees}"
        )

    return jsonify({"received": True})


# ── Payments page ─────────────────────────────────────────────────────────
@app.route("/payments", methods=["GET", "POST"])
def payments():
    if request.method == "GET":
        return render_template("payments.html", message=None, txns=[],
                               stripe_enabled=STRIPE_ENABLED)

    acc = request.form.get("account_number", "").strip()
    pin = request.form.get("pin", "").strip()
    user = get_user(acc)
    if not user:
        return render_template("payments.html",
            message="Account not found.", txns=[], stripe_enabled=STRIPE_ENABLED)
    if not authenticate(acc, pin):
        return render_template("payments.html",
            message="Invalid PIN.", txns=[], stripe_enabled=STRIPE_ENABLED)

    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "SELECT type, amount, balance_after, note, created_at "
        "FROM transactions WHERE account_number=? AND note LIKE '%Stripe%' "
        "ORDER BY id DESC LIMIT 50",
        (acc,)
    )
    rows = c.fetchall()
    conn.close()
    return render_template("payments.html",
        message=f"Showing {len(rows)} Stripe transactions for {acc}.",
        txns=rows, stripe_enabled=STRIPE_ENABLED)


if __name__ == "__main__":
    app.run(debug=True, port=5001, host="0.0.0.0")

