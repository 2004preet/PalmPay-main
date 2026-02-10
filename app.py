
from flask import Flask, render_template, request
import sqlite3
import cv2
import base64
import re
import numpy as np
from palm_recognition import get_palm_recognizer
import threading
import time

app = Flask(__name__)

DB = "palm_pay.db"

# Initialize palm recognizer
palm_recognizer = get_palm_recognizer()

# Global variable to track retraining status
retraining_status = {"in_progress": False, "last_retrained": None, "message": ""}

def retrain_model_background():
    """Background function to retrain the model with all stored palm data"""
    global retraining_status
    try:
        retraining_status["in_progress"] = True
        retraining_status["message"] = "Starting model retraining..."

        # Get all palm images from database
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("SELECT hand_image FROM users WHERE hand_image IS NOT NULL")
        rows = c.fetchall()
        conn.close()

        total_images = len(rows)
        if total_images < 5:
            retraining_status["message"] = f"Not enough data for retraining (need at least 5 images, have {total_images})"
            retraining_status["in_progress"] = False
            return

        retraining_status["message"] = f"Found {total_images} palm images for retraining"

        # Adaptive training parameters based on dataset size for SPEED optimization
        if total_images <= 10:
            epochs = 25
            batch_size = 4
            augment_factor = 8
        elif total_images <= 50:
            epochs = 20
            batch_size = 8
            augment_factor = 6
        elif total_images <= 100:
            epochs = 15
            batch_size = 12
            augment_factor = 4
        elif total_images <= 500:
            epochs = 12
            batch_size = 16
            augment_factor = 3
        else:  # 500+ users like 1000 users
            epochs = 8  # Very fast for large datasets
            batch_size = 24  # Larger batch size for speed
            augment_factor = 2  # Less augmentation for speed

        retraining_status["message"] = f"Adaptive training: {epochs} epochs, batch size {batch_size}, {total_images * augment_factor} total samples"

        # Create temporary training data
        import os
        import tempfile
        import shutil
        from advanced_train_model import create_arcface_model, load_images_with_labels, create_training_data
        from tensorflow import keras

        # Create temporary directory structure
        temp_dir = tempfile.mkdtemp()
        train_dir = os.path.join(temp_dir, "Train")
        os.makedirs(train_dir)

        # Save images to temporary files
        for i, (img_blob,) in enumerate(rows):
            try:
                nparr = np.frombuffer(img_blob, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    filename = f"IMG_{i+1:04d}.jpg"
                    filepath = os.path.join(train_dir, filename)
                    cv2.imwrite(filepath, img)
            except Exception as e:
                print(f"Error saving image {i}: {e}")
                continue

        retraining_status["message"] = f"Saved {len(os.listdir(train_dir))} images to temporary directory"

        # Temporarily modify the training script paths and parameters
        import advanced_train_model
        original_train_dir = advanced_train_model.TRAIN_DIR
        original_valid_dir = advanced_train_model.VALID_DIR
        original_model_path = advanced_train_model.MODEL_SAVE_PATH
        original_epochs = advanced_train_model.EPOCHS
        original_batch_size = advanced_train_model.BATCH_SIZE

        advanced_train_model.TRAIN_DIR = train_dir
        advanced_train_model.VALID_DIR = None  # No validation for retraining
        advanced_train_model.MODEL_SAVE_PATH = "palm_feature_extractor_advanced.h5"
        advanced_train_model.EPOCHS = epochs
        advanced_train_model.BATCH_SIZE = batch_size

        try:
            # Temporarily modify augmentation factor for speed
            import types
            original_create_training_data = advanced_train_model.create_training_data
            def fast_create_training_data(images, labels, augment_factor=augment_factor):
                return original_create_training_data(images, labels, augment_factor)
            advanced_train_model.create_training_data = fast_create_training_data

            retraining_status["message"] = f"🚀 Fast training: {epochs} epochs, batch size {batch_size}..."
            advanced_train_model.train_arcface_model()
            retraining_status["message"] = f"✅ Model retraining completed successfully! Trained on {total_images} users with {epochs} epochs in record time."
            retraining_status["last_retrained"] = time.strftime("%Y-%m-%d %H:%M:%S")

        except Exception as e:
            retraining_status["message"] = f"Retraining failed: {str(e)}"
        finally:
            # Restore original parameters
            advanced_train_model.TRAIN_DIR = original_train_dir
            advanced_train_model.VALID_DIR = original_valid_dir
            advanced_train_model.MODEL_SAVE_PATH = original_model_path
            advanced_train_model.EPOCHS = original_epochs
            advanced_train_model.BATCH_SIZE = original_batch_size
            advanced_train_model.create_training_data = original_create_training_data

            # Clean up temporary directory
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

def assess_image_quality(frame):
    """Assess image quality - check brightness, contrast, and blur"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Check brightness (mean intensity)
    brightness = np.mean(gray)
    
    # Check contrast (standard deviation)
    contrast = np.std(gray)
    
    # Check sharpness using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Return quality metrics
    quality_score = 0
    issues = []
    
    if brightness < 50:
        issues.append("too dark")
        quality_score += 20
    elif brightness > 200:
        issues.append("too bright")
        quality_score += 20
    else:
        quality_score += 40
    
    if contrast < 20:
        issues.append("low contrast")
        quality_score += 10
    else:
        quality_score += 30
    
    if laplacian_var < 50:
        issues.append("blurry")
        quality_score += 10
    else:
        quality_score += 30
    
    return quality_score, issues, {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': laplacian_var
    }

def capture_palm_image():
    """Capture high-quality palm image from camera with quality checks"""
    cap = cv2.VideoCapture(0)
    hand_image = None
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera. Check permissions and connections.")
        return None
    
    # Optimize camera settings for quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_FOCUS_MODE, 1)
    
    # Warm up camera
    for _ in range(10):
        cap.read()
    
    time.sleep(1)
    
    print("📷 Capturing palm image... Position your palm clearly in front of camera")
    
    gui_available = True
    start_time = time.time()
    capture_time = 4  # Seconds to wait
    best_frame = None
    best_quality = -1
    frames_evaluated = 0
    
    try:
        while time.time() - start_time < capture_time:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame from camera")
                break
            
            frames_evaluated += 1
            quality_score, issues, metrics = assess_image_quality(frame)
            
            # Keep the best quality frame
            if quality_score > best_quality:
                best_quality = quality_score
                best_frame = frame.copy()
            
            # Display frame with quality feedback
            if gui_available:
                try:
                    # Add quality info to frame
                    display_frame = frame.copy()
                    status_text = f"Quality: {quality_score}% | Brightness: {metrics['brightness']:.0f} | Contrast: {metrics['contrast']:.0f}"
                    cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if issues:
                        issue_text = "Issues: " + ", ".join(issues)
                        cv2.putText(display_frame, issue_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    elapsed = time.time() - start_time
                    remaining = int(capture_time - elapsed)
                    timer_text = f"Capturing in {remaining}s | Press 's' to save, 'q' to cancel"
                    cv2.putText(display_frame, timer_text, (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    cv2.imshow("PalmPay - Palm Registration", display_frame)
                    key = cv2.waitKey(100) & 0xFF
                    
                    if key == ord('s'):
                        hand_image = best_frame if best_frame is not None else frame
                        print(f"✅ Image captured (user pressed 's') - Quality: {best_quality}%")
                        break
                    elif key == ord('q'):
                        print("❌ Capture cancelled by user")
                        break
                except Exception as e:
                    gui_available = False
            
            time.sleep(0.05)
        
        # If auto-capture triggered, use best frame
        if hand_image is None and best_frame is not None:
            hand_image = best_frame
            print(f"✅ Auto-captured best frame - Quality: {best_quality}% after {frames_evaluated} frames")
        
        # Encode the image
        if hand_image is not None:
            success, encoded = cv2.imencode('.jpg', hand_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success:
                return encoded.tobytes()
            else:
                print("❌ Error: Failed to encode image")
                return None
    
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
    
    return None

def capture_multiple_palm_images(num_captures=3):
    """Capture multiple palm images for better feature extraction"""
    images = []
    print(f"Capturing {num_captures} palm images for registration...")
    
    for i in range(num_captures):
        print(f"\n--- Capture {i+1}/{num_captures} ---")
        print("Please position your palm and hold steady...")
        image = capture_palm_image()
        if image is not None:
            images.append(image)
            print(f"✅ Capture {i+1} successful")
            if i < num_captures - 1:
                print("Please adjust your palm position slightly for the next capture...")
                time.sleep(1)  # Brief pause between captures
        else:
            print(f"❌ Capture {i+1} failed")
    
    if len(images) == 0:
        return None
    elif len(images) < num_captures:
        print(f"Warning: Only captured {len(images)} out of {num_captures} images")
    
    return images

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

    # Capture multiple palm images for better accuracy
    hand_images = capture_multiple_palm_images(num_captures=3)

    if hand_images is None or len(hand_images) == 0:
        error_msg = "❌ No palm images captured. Troubleshooting: 1) Check camera permissions (System Preferences → Security & Privacy → Camera), 2) Ensure camera is not in use by another app, 3) Verify camera is connected and working."
        return render_template("register.html", message=error_msg)

    # Extract palm features from all images and average them
    try:
        # Use batch extraction for efficiency
        feature_list = palm_recognizer.extract_features_batch(hand_images, fast_mode=False)
        
        # Average the features for more robust representation
        palm_features = np.mean(feature_list, axis=0)
        # Re-normalize after averaging
        palm_features = palm_features / np.linalg.norm(palm_features)
        palm_features_bytes = palm_features.tobytes()
        
        # Store the first image as representative
        hand_image = hand_images[0]
        
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
        
        # Check total users for auto-retraining
        c.execute("SELECT COUNT(*) FROM users WHERE hand_image IS NOT NULL")
        total_users = c.fetchone()[0]
        conn.close()
        
        # Auto-retrain if we have enough users and retraining is not already in progress
        if total_users >= 5 and not retraining_status["in_progress"]:
            # Check if we haven't retrained recently (not in last hour)
            should_retrain = True
            if retraining_status["last_retrained"]:
                from datetime import datetime
                last_retrain_time = datetime.strptime(retraining_status["last_retrained"], "%Y-%m-%d %H:%M:%S")
                time_since_retrain = datetime.now() - last_retrain_time
                if time_since_retrain.total_seconds() < 3600:  # 1 hour
                    should_retrain = False
            
            if should_retrain:
                # Start automatic retraining in background
                thread = threading.Thread(target=retrain_model_background)
                thread.daemon = True
                thread.start()
                success_message = f"Registration successful! {len(hand_images)} palm images captured and features averaged for enhanced accuracy. Model is being automatically retrained in the background."
            else:
                success_message = f"Registration successful! {len(hand_images)} palm images captured and features averaged for enhanced accuracy."
        else:
            success_message = "Registration successful! Palm features extracted and stored."
        
        return render_template("register.html", message=success_message)
        
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

@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    global retraining_status
    if request.method == "GET":
        return render_template("retrain.html", status=retraining_status)
    
    if retraining_status["in_progress"]:
        return render_template("retrain.html", status=retraining_status, message="Retraining already in progress")
    
    # Start retraining in background thread
    thread = threading.Thread(target=retrain_model_background)
    thread.daemon = True
    thread.start()
    
    return render_template("retrain.html", status=retraining_status, message="Retraining started in background")

@app.route("/deposit", methods=["GET", "POST"])
def deposit():
    if request.method == "GET":
        return render_template("deposit.html", message=None)
    acc = request.form.get("account_number","").strip()
    pin = request.form.get("pin","").strip()
    amt = request.form.get("amount","0").strip()

    try:
        user = get_user(acc)
        if not user:
            return render_template("deposit.html", message="❌ Account not found. Please register first.")

        if not authenticate(acc, pin):
            return render_template("deposit.html", message="❌ Invalid PIN. Please try again.")

        # Palm authentication for UPI transaction
        print("📷 Please position your palm for verification...")
        palm_image = capture_palm_image()
        if palm_image is None:
            return render_template("deposit.html", message="❌ Palm image not captured. Please try again.")
        
        is_verified, similarity, error_msg = verify_palm_authentication(acc, palm_image)
        if not is_verified:
            return render_template("deposit.html", message=f"❌ Palm authentication failed (similarity: {similarity:.1%}). {error_msg}")

        try:
            amount = float(amt)
            if amount <= 0:
                return render_template("deposit.html", message="❌ Amount must be greater than 0.")
        except ValueError:
            return render_template("deposit.html", message="❌ Invalid amount format.")

        new_balance = float(user[7]) + amount
        set_balance(acc, new_balance)
        add_txn(acc, "deposit", amount, new_balance, "Cash deposit - Palm verified")
        return render_template("deposit.html", message=f"✅ Deposit successful! Palm verified (match: {similarity:.1%}). New balance: ₹{new_balance:.2f}")
    
    except Exception as e:
        print(f"Error in deposit: {e}")
        return render_template("deposit.html", message=f"❌ Error processing deposit: {str(e)}")


@app.route("/withdraw", methods=["GET", "POST"])
def withdraw():
    if request.method == "GET":
        return render_template("withdraw.html", message=None)
    acc = request.form.get("account_number","").strip()
    pin = request.form.get("pin","").strip()
    amt = request.form.get("amount","0").strip()

    try:
        user = get_user(acc)
        if not user:
            return render_template("withdraw.html", message="❌ Account not found. Please register first.")

        if not authenticate(acc, pin):
            return render_template("withdraw.html", message="❌ Invalid PIN. Please try again.")

        # Palm authentication for UPI transaction
        print("📷 Please position your palm for verification...")
        palm_image = capture_palm_image()
        if palm_image is None:
            return render_template("withdraw.html", message="❌ Palm image not captured. Please try again.")
        
        is_verified, similarity, error_msg = verify_palm_authentication(acc, palm_image)
        if not is_verified:
            return render_template("withdraw.html", message=f"❌ Palm authentication failed (similarity: {similarity:.1%}). {error_msg}")

        try:
            amount = float(amt)
            if amount <= 0:
                return render_template("withdraw.html", message="❌ Amount must be greater than 0.")
            if amount > float(user[7]):
                return render_template("withdraw.html", message=f"❌ Insufficient funds. Available balance: ₹{user[7]:.2f}")
        except ValueError:
            return render_template("withdraw.html", message="❌ Invalid amount format.")

        new_balance = float(user[7]) - amount
        set_balance(acc, new_balance)
        add_txn(acc, "withdraw", amount, new_balance, "Cash withdrawal - Palm verified")
        return render_template("withdraw.html", message=f"✅ Withdrawal successful! Palm verified (match: {similarity:.1%}). New balance: ₹{new_balance:.2f}")
    
    except Exception as e:
        print(f"Error in withdraw: {e}")
        return render_template("withdraw.html", message=f"❌ Error processing withdrawal: {str(e)}")

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
