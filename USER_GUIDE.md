# PalmPay - Complete User Guide

## 🎯 Overview

PalmPay is a secure payment system that uses palm recognition for authentication. Users register their palm during signup, and then use it to verify transactions (deposit, withdraw, transfer money).

## 📊 Monitoring Training Progress

### Check Training Status

The model is currently training in the background. You can monitor it with:

```bash
# View live training progress
tail -f training_live.log

# View last 30 lines
tail -30 training_live.log

# Check if training is still running
ps aux | grep "train_model.py" | grep -v grep
```

### Training Status (Current)
- ✅ **Training in Progress**: Epoch 17/100
- ✅ **Loss Decreasing**: 0.6978 → 0.1721 (good progress!)
- ✅ **Model Saving**: Best models saved automatically
- ✅ **Learning Rate**: Reduced to 0.0005 (adaptive)
- ⏱️ **Estimated Time**: 10-20 more minutes

### When Training Completes

You'll see:
1. Similarity statistics (same palm vs different palms)
2. Recommended threshold value
3. Model saved to `palm_feature_extractor.h5`
4. Ready to use message

## 🚀 How to Use the PalmPay App

### Step 1: Start the Application

```bash
# Make sure training is complete (or use default model)
python3 app.py
```

The app will start at: **http://localhost:5000**

### Step 2: Register a New User

1. **Open the Registration Page**
   - Go to: http://localhost:5000/register
   - Fill in the registration form:
     - **Name**: Your full name
     - **Account Number**: Unique account number (e.g., "ACC001")
     - **Phone**: Your phone number (optional)
     - **Address**: Your address (optional)
     - **Account Type**: Savings/Current (optional)
     - **PIN**: 4-digit PIN (e.g., "1234")

2. **Capture Palm Image**
   - Click "Register" button
   - **Camera window will open** showing your palm
   - **Position your palm** clearly in front of the camera
   - **Press 's' key** to save/capture the palm image
   - **Press 'q' key** to cancel if needed
   - Camera window will close automatically

3. **Registration Complete**
   - You'll see: "Registration successful! Palm features extracted and stored."
   - Your palm features are now stored in the database
   - You can now use Deposit, Withdraw, or Transfer

### Step 3: Deposit Money

1. **Go to Deposit Page**
   - Navigate to: http://localhost:5000/deposit
   - Enter your:
     - **Account Number**: (e.g., "ACC001")
     - **PIN**: Your 4-digit PIN
     - **Amount**: Amount to deposit (e.g., "1000")

2. **Authenticate with Palm**
   - Click "Deposit" button
   - **Camera window will open**
   - **Position the SAME palm** you used during registration
   - **Press 's' key** to capture
   - System verifies your palm against stored features

3. **Transaction Complete**
   - If palm matches: "Deposit successful! Palm verified (similarity: XX%)"
   - If palm doesn't match: "Palm authentication failed. Similarity: XX%"
   - New balance will be displayed

### Step 4: Withdraw Money

1. **Go to Withdraw Page**
   - Navigate to: http://localhost:5000/withdraw
   - Enter your:
     - **Account Number**: (e.g., "ACC001")
     - **PIN**: Your 4-digit PIN
     - **Amount**: Amount to withdraw

2. **Authenticate with Palm**
   - Click "Withdraw" button
   - **Camera window will open**
   - **Position the SAME palm** you used during registration
   - **Press 's' key** to capture
   - System verifies your palm

3. **Transaction Complete**
   - If verified: Withdrawal successful with new balance
   - If not verified: Authentication failed message

### Step 5: Transfer Money

1. **Go to Transfer Page**
   - Navigate to: http://localhost:5000/transfer
   - Enter:
     - **From Account**: Your account number
     - **PIN**: Your 4-digit PIN
     - **To Account**: Recipient's account number
     - **Amount**: Amount to transfer

2. **Authenticate with Palm**
   - Click "Transfer" button
   - **Camera window will open**
   - **Position the SAME palm** you used during registration
   - **Press 's' key** to capture
   - System verifies sender's palm (recipient doesn't need verification)

3. **Transaction Complete**
   - If verified: Transfer successful
   - Money is deducted from sender and added to recipient
   - New balance displayed

### Step 6: Check Balance

1. **Go to Balance Page**
   - Navigate to: http://localhost:5000/balance
   - Enter your:
     - **Account Number**
     - **PIN**
   - Click "Check Balance"
   - Current balance displayed (no palm verification needed)

### Step 7: View Transaction History

1. **Go to History Page**
   - Navigate to: http://localhost:5000/history
   - Enter your:
     - **Account Number**
     - **PIN**
   - Click "View History"
   - Last 100 transactions displayed

## 📸 Camera Capture Instructions

### Important Tips for Palm Capture

1. **Use the Same Palm**
   - Always use the **SAME hand** (left or right) for registration and transactions
   - Don't switch hands - it won't match!

2. **Positioning**
   - Place palm **flat** in front of camera
   - Keep palm **still** when capturing
   - Ensure **good lighting**
   - Avoid shadows on palm

3. **Camera Window Controls**
   - **'s' key**: Save/capture the palm image
   - **'q' key**: Cancel and go back
   - **Close window**: Automatically closes after capture

4. **Image Quality**
   - Ensure palm is **clearly visible**
   - Avoid blurry images
   - Keep background simple
   - Good contrast helps

## 🔒 Security Features

### Authentication Requirements

1. **PIN Verification**: 4-digit PIN required
2. **Palm Verification**: Palm must match stored features
3. **Similarity Threshold**: Default 75% (adjustable)
4. **Dual Authentication**: Both PIN and palm required

### Palm Matching

- **Similarity Score**: Shows how well palm matches (0-100%)
- **Threshold**: 75% similarity required (default)
- **High Accuracy**: Trained model provides better matching
- **False Rejection**: If similarity < threshold, transaction denied

## 🛠️ Troubleshooting

### Camera Not Opening

1. **Check Permissions**
   - macOS: System Preferences → Security & Privacy → Camera
   - Allow terminal/Python to access camera

2. **Check Camera Availability**
   - Make sure camera is not used by another app
   - Try closing other apps using camera

3. **Check Camera Index**
   - Default is camera 0
   - If multiple cameras, may need to change index in code

### Palm Authentication Fails

1. **Use Same Palm**
   - Must use the same hand as registration
   - Check if you're using left vs right hand

2. **Positioning**
   - Ensure palm is in similar position as registration
   - Good lighting is important
   - Keep palm still during capture

3. **Similarity Threshold**
   - Check similarity score in error message
   - If too low, may need to adjust threshold
   - Or re-register with better palm image

### Training Issues

1. **Training Not Starting**
   - Check if dependencies are installed
   - Verify images exist in Files/Train/
   - Check for errors in terminal

2. **Training Takes Too Long**
   - Normal: 30-60 minutes depending on hardware
   - Can reduce epochs in train_model.py
   - Can reduce batch size if out of memory

3. **Low Accuracy After Training**
   - Add more training images
   - Ensure images are clear and well-lit
   - Check similarity statistics after training
   - Adjust threshold if needed

## 📋 Quick Reference

### Registration Flow
```
1. Fill registration form
2. Camera opens → Position palm → Press 's'
3. Palm features extracted and stored
4. Registration complete
```

### Transaction Flow (Deposit/Withdraw/Transfer)
```
1. Enter account number and PIN
2. Enter amount (for deposit/withdraw/transfer)
3. Camera opens → Position SAME palm → Press 's'
4. Palm verified against stored features
5. If verified → Transaction processed
6. If not verified → Transaction denied
```

### Camera Controls
- **'s' key**: Save/capture image
- **'q' key**: Cancel
- **Window closes**: Automatically after capture

## 🎯 Best Practices

1. **Registration**
   - Use clear, well-lit palm image
   - Keep palm flat and still
   - Use the same hand consistently

2. **Transactions**
   - Always use the same palm as registration
   - Ensure good lighting
   - Keep palm in similar position

3. **Security**
   - Don't share your PIN
   - Don't share your account number
   - Use strong 4-digit PIN

4. **Accuracy**
   - Wait for training to complete for best accuracy
   - Use recommended threshold from training
   - Re-register if authentication fails repeatedly

## 📞 Support

### Check Training Progress
```bash
tail -f training_live.log
```

### Check App Logs
```bash
python3 app.py
# Check terminal output for errors
```

### View Users
- Go to: http://localhost:5000/users
- View all registered users and their balances

---

## 🚀 Getting Started Checklist

- [ ] Training complete (or using default model)
- [ ] App running (python3 app.py)
- [ ] Camera permissions granted
- [ ] Registered first user
- [ ] Tested deposit transaction
- [ ] Tested withdraw transaction
- [ ] Tested transfer transaction

---

**Note**: The app uses **camera capture** (not file upload). Images are captured in real-time from your webcam during registration and transactions.

