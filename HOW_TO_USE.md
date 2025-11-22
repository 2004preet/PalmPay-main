# How to Upload Images and Proceed Operations in PalmPay

## 📸 Important: No File Upload - Camera Capture Only

**PalmPay uses camera capture, not file upload.** Images are captured in real-time from your webcam during registration and transactions.

## 🎯 Step-by-Step Guide

### 1. Monitor Training Progress (Currently Running)

```bash
# Check training status
tail -f training_live.log

# Quick status check
tail -10 training_live.log | grep -E "Epoch|Saved|Training completed"
```

**Current Status**: Training is in progress (Epoch 17/100)
- Loss is decreasing: ✅ Good progress
- Model saving automatically: ✅ Working
- Estimated time remaining: 10-20 minutes

### 2. Start the Application

```bash
# After training completes (or use default model)
python3 app.py
```

The app will start at: **http://localhost:5000**

### 3. Register a New User (Capture Palm Image)

#### Step 3.1: Fill Registration Form
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

#### Step 3.2: Capture Palm Image
1. Click **"Register"** button
2. **Camera window opens automatically**
3. **Position your palm** in front of the camera:
   - Keep palm flat
   - Ensure good lighting
   - Keep palm still
   - Make sure palm is clearly visible
4. **Press 's' key** on keyboard to save/capture
5. **Press 'q' key** to cancel (if needed)
6. Camera window closes automatically
7. Palm features are extracted and stored

#### Step 3.3: Registration Complete
- You'll see: "Registration successful! Palm features extracted and stored."
- Your palm is now registered in the system
- You can now perform transactions

### 4. Deposit Money (Verify Palm)

#### Step 4.1: Fill Deposit Form
1. Go to: **http://localhost:5000/deposit**
2. Enter:
   ```
   Account Number: ACC001
   PIN: 1234
   Amount: 1000
   ```

#### Step 4.2: Verify Palm
1. Click **"Deposit"** button
2. **Camera window opens automatically**
3. **Position the SAME palm** you used during registration:
   - Must be the same hand (left or right)
   - Same positioning as registration
   - Good lighting
4. **Press 's' key** to capture
5. System verifies palm against stored features
6. Camera window closes automatically

#### Step 4.3: Transaction Result
- **If verified**: "Deposit successful! Palm verified (similarity: 85%)"
- **If not verified**: "Palm authentication failed. Similarity: 65%"
- New balance displayed

### 5. Withdraw Money (Verify Palm)

1. Go to: **http://localhost:5000/withdraw**
2. Enter account number, PIN, and amount
3. Click **"Withdraw"**
4. **Camera opens** → Position **SAME palm** → **Press 's'**
5. Transaction processed if palm matches

### 6. Transfer Money (Verify Palm)

#### Step 6.1: Register Recipient (if needed)
1. Register another user with Account Number: "ACC002"

#### Step 6.2: Transfer Money
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

### 7. Check Balance

1. Go to: **http://localhost:5000/balance**
2. Enter account number and PIN
3. Click "Check Balance"
4. Current balance displayed (no palm verification needed)

### 8. View Transaction History

1. Go to: **http://localhost:5000/history**
2. Enter account number and PIN
3. Click "View History"
4. Last 100 transactions displayed

## 🎥 Camera Capture Details

### How Camera Capture Works

1. **Automatic Opening**: Camera window opens when needed
2. **Live Preview**: You see your palm in real-time
3. **Keyboard Controls**:
   - **'s' key**: Save/capture the image
   - **'q' key**: Cancel and go back
4. **Automatic Closing**: Window closes after capture
5. **Image Processing**: Captured image is processed immediately

### Camera Window Appearance

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

### Best Practices for Palm Capture

1. **Lighting**:
   - ✅ Use good, even lighting
   - ✅ Avoid shadows on palm
   - ✅ Avoid direct bright light
   - ❌ Don't use dim lighting

2. **Positioning**:
   - ✅ Keep palm flat
   - ✅ Position palm in center of frame
   - ✅ Keep palm still
   - ✅ Full palm visible
   - ❌ Don't move palm during capture

3. **Consistency**:
   - ✅ Always use same hand (left or right)
   - ✅ Similar positioning as registration
   - ✅ Similar distance from camera
   - ❌ Don't switch hands

4. **Image Quality**:
   - ✅ Clear, sharp image
   - ✅ Good contrast
   - ✅ Simple background
   - ❌ Avoid blurry images

## 🔄 Complete Operation Flow

### Registration Flow
```
1. Fill registration form
   ↓
2. Click "Register"
   ↓
3. Camera window opens
   ↓
4. Position palm → Press 's'
   ↓
5. Camera closes
   ↓
6. Palm features extracted
   ↓
7. Stored in database
   ↓
8. Registration complete
```

### Transaction Flow (Deposit/Withdraw/Transfer)
```
1. Fill transaction form (account, PIN, amount)
   ↓
2. Click transaction button
   ↓
3. PIN verified
   ↓
4. Camera window opens
   ↓
5. Position SAME palm → Press 's'
   ↓
6. Camera closes
   ↓
7. Palm features extracted
   ↓
8. Compared with stored features
   ↓
9. Similarity calculated
   ↓
10. If similarity >= threshold (75%):
    → Transaction approved
    → Money transferred
    → Success message
11. If similarity < threshold:
    → Transaction denied
    → Error message
```

## 📊 Understanding Similarity Scores

### What is Similarity?

- **Similarity Score**: 0-100% (0.0 to 1.0)
- **Threshold**: 75% (default, adjustable)
- **Higher is Better**: Closer to 100% = better match

### Similarity Ranges

- **90-100%**: Excellent match ✅
- **80-90%**: Very good match ✅
- **75-80%**: Good match ✅ (threshold)
- **70-75%**: Below threshold ⚠️
- **<70%**: Poor match ❌

### Why Authentication Fails

1. **Different Hand**: Using different hand than registration
2. **Poor Lighting**: Insufficient or uneven lighting
3. **Wrong Position**: Palm in different position
4. **Movement**: Palm moved during capture
5. **Blurry Image**: Camera not focused
6. **Low Threshold**: Similarity below 75%

## 🛠️ Troubleshooting

### Camera Not Opening

**Problem**: Camera window doesn't open

**Solutions**:
1. Check camera permissions:
   - macOS: System Preferences → Security & Privacy → Camera
   - Allow Python/Terminal to access camera
2. Close other apps using camera
3. Restart the app
4. Check if camera is connected

### Palm Authentication Fails

**Problem**: "Palm authentication failed" message

**Solutions**:
1. Use the same palm as registration
2. Check similarity score in error message
3. Ensure good lighting
4. Keep palm in similar position
5. Re-register if needed
6. Adjust threshold if similarity is close (e.g., 73%)

### Training Not Complete

**Problem**: Want to use app before training completes

**Solutions**:
1. App works with default model (lower accuracy)
2. Wait for training to complete (better accuracy)
3. Check training progress: `tail -f training_live.log`
4. Training usually takes 30-60 minutes

## 🎯 Quick Reference

### Registration
```
URL: http://localhost:5000/register
Steps: Fill form → Click Register → Camera opens → Press 's' → Done
```

### Deposit
```
URL: http://localhost:5000/deposit
Steps: Fill form → Click Deposit → Camera opens → Press 's' → Done
```

### Withdraw
```
URL: http://localhost:5000/withdraw
Steps: Fill form → Click Withdraw → Camera opens → Press 's' → Done
```

### Transfer
```
URL: http://localhost:5000/transfer
Steps: Fill form → Click Transfer → Camera opens → Press 's' → Done
```

### Camera Controls
```
's' key: Save/capture image
'q' key: Cancel
Window: Closes automatically after capture
```

## ✅ Checklist

Before using the app:
- [ ] Training complete (or using default model)
- [ ] App running (`python3 app.py`)
- [ ] Camera permissions granted
- [ ] Camera working (test with other apps)

For registration:
- [ ] Fill all required fields
- [ ] Camera window opens
- [ ] Palm clearly visible
- [ ] Press 's' to capture
- [ ] Registration successful

For transactions:
- [ ] Enter correct account number
- [ ] Enter correct PIN
- [ ] Enter amount
- [ ] Use SAME palm as registration
- [ ] Press 's' to capture
- [ ] Transaction successful

## 🎉 You're Ready!

Follow these steps to:
1. ✅ Register users with palm images
2. ✅ Deposit money with palm verification
3. ✅ Withdraw money with palm verification
4. ✅ Transfer money between users
5. ✅ Check balance and transaction history

**Remember**: Always use the **SAME palm** for registration and all transactions!

---

For detailed information, see:
- **USER_GUIDE.md**: Complete user guide
- **QUICK_START.md**: Quick start guide
- **TRAINING_GUIDE.md**: Training instructions

