# PalmPay - Complete Guide: Image Upload & Operations

## 🎯 Quick Answer: How to Upload Images and Proceed Operations

**Important**: PalmPay doesn't use file upload. It uses **real-time camera capture** from your webcam.

### How It Works:
1. **Camera opens automatically** when needed
2. **Position your palm** in front of camera
3. **Press 's' key** to capture
4. **Camera closes automatically**
5. **Image is processed immediately**

---

## 📊 Training Status

### Current Status: ✅ Training in Progress (Fine-tuning Phase)

- **Initial Training**: ✅ Completed (Early stopped at Epoch 25)
- **Fine-tuning**: 🔄 In Progress (Epoch 1/20)
- **Loss**: ✅ Decreasing (Good progress)
- **Model Saving**: ✅ Automatic
- **Estimated Time**: 5-10 more minutes

### Monitor Training:
```bash
# View live progress
tail -f training_live.log

# Check status
tail -20 training_live.log
```

---

## 🚀 Complete Step-by-Step Guide

### Step 1: Wait for Training to Complete

```bash
# Check if training is complete
tail -20 training_live.log | grep "Training completed"

# When you see "Training completed successfully!", proceed
```

### Step 2: Start the Application

```bash
# Start the Flask app
python3 app.py
```

Open browser: **http://localhost:5000**

### Step 3: Register a New User

#### 3.1: Fill Registration Form
1. Go to: **http://localhost:5000/register**
2. Fill in the form:
   ```
   Name: John Doe
   Account Number: ACC001
   Phone: 1234567890
   Address: 123 Main St
   Account Type: Savings
   PIN: 1234
   ```

#### 3.2: Capture Palm Image (Camera)
1. Click **"Register"** button
2. **Camera window opens automatically** (OpenCV window)
3. **Position your palm**:
   - Keep palm flat
   - Ensure good lighting
   - Keep palm still
   - Make sure palm is clearly visible
4. **Press 's' key** on your keyboard to save/capture
5. **Press 'q' key** to cancel (if needed)
6. Camera window closes automatically
7. System processes the image

#### 3.3: Registration Complete
- You'll see: "Registration successful! Palm features extracted and stored."
- Your palm features are stored in the database
- You can now perform transactions

### Step 4: Deposit Money

#### 4.1: Fill Deposit Form
1. Go to: **http://localhost:5000/deposit**
2. Enter:
   ```
   Account Number: ACC001
   PIN: 1234
   Amount: 1000
   ```

#### 4.2: Verify Palm (Camera)
1. Click **"Deposit"** button
2. **Camera window opens automatically**
3. **Position the SAME palm** you used during registration:
   - Must be the same hand (left or right)
   - Same positioning as registration
   - Good lighting
4. **Press 's' key** to capture
5. System verifies palm against stored features
6. Camera window closes automatically

#### 4.3: Transaction Result
- **If verified** (similarity >= 75%): 
  - "Deposit successful! Palm verified (similarity: 85%)"
  - New balance displayed
- **If not verified** (similarity < 75%): 
  - "Palm authentication failed. Similarity: 65%"
  - Transaction denied

### Step 5: Withdraw Money

1. Go to: **http://localhost:5000/withdraw**
2. Enter account number, PIN, and amount
3. Click **"Withdraw"**
4. **Camera opens** → Position **SAME palm** → **Press 's'**
5. System verifies palm
6. Transaction processed if verified

### Step 6: Transfer Money

#### 6.1: Register Recipient (if needed)
1. Register another user with Account Number: "ACC002"

#### 6.2: Transfer Money
1. Go to: **http://localhost:5000/transfer**
2. Enter:
   ```
   From Account: ACC001
   PIN: 1234
   To Account: ACC002
   Amount: 500
   ```
3. Click **"Transfer"**
4. **Camera opens** → Position **SAME palm** → **Press 's'**
5. System verifies sender's palm
6. Money transferred if verified

### Step 7: Check Balance

1. Go to: **http://localhost:5000/balance**
2. Enter account number and PIN
3. Click "Check Balance"
4. Current balance displayed (no palm verification needed)

### Step 8: View Transaction History

1. Go to: **http://localhost:5000/history**
2. Enter account number and PIN
3. Click "View History"
4. Last 100 transactions displayed

---

## 📸 Camera Capture Details

### How Camera Capture Works

1. **Automatic Opening**: Camera window opens when needed
2. **Live Preview**: You see your palm in real-time
3. **Keyboard Controls**:
   - **'s' key**: Save/capture the image
   - **'q' key**: Cancel and go back
4. **Automatic Closing**: Window closes after capture
5. **Image Processing**: Captured image is processed immediately

### Camera Window
```
┌─────────────────────────────────┐
│  PalmPay - Press 's' to save    │
│  palm image, 'q' to cancel      │
│                                 │
│  [Live Camera Feed]             │
│  [Your Palm Here]               │
│                                 │
└─────────────────────────────────┘
```

### Best Practices

1. **Use Same Palm**: Always use the same hand (left or right)
2. **Good Lighting**: Ensure adequate lighting
3. **Keep Still**: Keep palm still during capture
4. **Clear View**: Make sure palm is clearly visible
5. **Similar Position**: Use similar position as registration

---

## 🔄 Complete Flow Diagrams

### Registration Flow
```
User fills form
    ↓
Clicks "Register"
    ↓
Camera window opens
    ↓
User positions palm
    ↓
User presses 's' key
    ↓
Camera window closes
    ↓
Image captured
    ↓
Palm features extracted
    ↓
Features stored in database
    ↓
Registration complete
```

### Transaction Flow (Deposit/Withdraw/Transfer)
```
User fills form (account, PIN, amount)
    ↓
Clicks transaction button
    ↓
PIN verified
    ↓
Camera window opens
    ↓
User positions SAME palm
    ↓
User presses 's' key
    ↓
Camera window closes
    ↓
Image captured
    ↓
Palm features extracted
    ↓
Features compared with stored features
    ↓
Similarity calculated
    ↓
If similarity >= threshold (75%):
    → Transaction approved
    → Money transferred
    → Success message
If similarity < threshold:
    → Transaction denied
    → Error message
```

---

## 🎯 Key Points

### Image Upload
- ❌ **No file upload**: PalmPay doesn't use file upload
- ✅ **Camera capture**: Images captured from webcam in real-time
- ✅ **Automatic processing**: Images processed immediately
- ✅ **Secure**: Images stored securely in database

### Operations
- ✅ **Registration**: Capture palm during signup
- ✅ **Deposit**: Verify palm to deposit money
- ✅ **Withdraw**: Verify palm to withdraw money
- ✅ **Transfer**: Verify palm to transfer money
- ✅ **Balance**: Check balance (no palm needed)
- ✅ **History**: View transactions (no palm needed)

### Security
- ✅ **Dual authentication**: PIN + Palm verification
- ✅ **Similarity threshold**: 75% (adjustable)
- ✅ **Secure storage**: Features stored in database
- ✅ **High accuracy**: Trained model provides better matching

---

## 🛠️ Troubleshooting

### Camera Not Opening
1. **Check permissions**: System Preferences → Security → Camera
2. **Close other apps**: Close apps using camera
3. **Restart app**: Restart the Flask app
4. **Check camera**: Make sure camera is connected

### Palm Authentication Fails
1. **Use same palm**: Must use same hand as registration
2. **Check similarity**: Check similarity score in error message
3. **Good lighting**: Ensure adequate lighting
4. **Re-register**: Re-register if authentication fails repeatedly
5. **Adjust threshold**: Adjust threshold if similarity is close

### Training Issues
1. **Check progress**: `tail -f training_live.log`
2. **Wait for completion**: Training takes 30-60 minutes
3. **Check errors**: Check for errors in training log
4. **Use default model**: App works with default model (lower accuracy)

---

## 📋 Quick Reference

### URLs
- **Home**: http://localhost:5000
- **Register**: http://localhost:5000/register
- **Deposit**: http://localhost:5000/deposit
- **Withdraw**: http://localhost:5000/withdraw
- **Transfer**: http://localhost:5000/transfer
- **Balance**: http://localhost:5000/balance
- **History**: http://localhost:5000/history
- **Users**: http://localhost:5000/users

### Camera Controls
- **'s' key**: Save/capture image
- **'q' key**: Cancel
- **Window**: Closes automatically after capture

### Commands
```bash
# Start app
python3 app.py

# Monitor training
tail -f training_live.log

# Check training status
tail -20 training_live.log

# Check if training is running
ps aux | grep train_model.py
```

---

## ✅ Checklist

### Before Using App
- [ ] Training complete (or using default model)
- [ ] App running (`python3 app.py`)
- [ ] Camera permissions granted
- [ ] Camera working (test with other apps)

### Registration
- [ ] Fill all required fields
- [ ] Camera window opens
- [ ] Palm clearly visible
- [ ] Press 's' to capture
- [ ] Registration successful

### Transactions
- [ ] Enter correct account number
- [ ] Enter correct PIN
- [ ] Enter amount
- [ ] Use SAME palm as registration
- [ ] Press 's' to capture
- [ ] Transaction successful

---

## 🎉 Summary

### How to Upload Images
- **No file upload**: PalmPay uses camera capture
- **Camera opens**: Automatically when needed
- **Press 's'**: To capture image
- **Press 'q'**: To cancel

### How to Proceed Operations
1. **Register**: Fill form → Camera opens → Press 's' → Done
2. **Deposit**: Fill form → Camera opens → Press 's' → Done
3. **Withdraw**: Fill form → Camera opens → Press 's' → Done
4. **Transfer**: Fill form → Camera opens → Press 's' → Done
5. **Balance**: Fill form → Click button → Done (no camera)
6. **History**: Fill form → Click button → Done (no camera)

### Key Requirements
- ✅ **Same palm**: Always use same hand
- ✅ **Good lighting**: Ensure adequate lighting
- ✅ **Still palm**: Keep palm still during capture
- ✅ **Clear view**: Make sure palm is clearly visible

---

**You're ready to use PalmPay!** 🚀

For more details, see:
- **USER_GUIDE.md**: Complete user guide
- **QUICK_START.md**: Quick start guide
- **HOW_TO_USE.md**: Detailed usage instructions
- **TRAINING_GUIDE.md**: Training instructions

