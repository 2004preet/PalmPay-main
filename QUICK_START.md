# PalmPay - Quick Start Guide

## 🚀 Quick Start (5 Minutes)

### 1. Wait for Training to Complete

```bash
# Check training status
tail -20 training_live.log

# When you see "Training completed successfully!", proceed
```

### 2. Start the App

```bash
python3 app.py
```

Open browser: **http://localhost:5000**

### 3. Register a User

1. Go to: **http://localhost:5000/register**
2. Fill in:
   - Name: "John Doe"
   - Account Number: "ACC001"
   - PIN: "1234"
   - (Other fields optional)
3. Click "Register"
4. **Camera opens** → Position your palm → **Press 's'** → Camera closes
5. Done! Registration successful

### 4. Deposit Money

1. Go to: **http://localhost:5000/deposit**
2. Enter:
   - Account Number: "ACC001"
   - PIN: "1234"
   - Amount: "1000"
3. Click "Deposit"
4. **Camera opens** → Position **SAME palm** → **Press 's'** → Camera closes
5. Done! Money deposited

### 5. Transfer Money

1. Register another user (Account: "ACC002")
2. Go to: **http://localhost:5000/transfer**
3. Enter:
   - From Account: "ACC001"
   - PIN: "1234"
   - To Account: "ACC002"
   - Amount: "500"
4. Click "Transfer"
5. **Camera opens** → Position **SAME palm** → **Press 's'** → Camera closes
6. Done! Money transferred

## 📸 Camera Capture

### How It Works

1. **Camera window opens automatically**
2. **Position your palm** in front of camera
3. **Press 's' key** to save/capture
4. **Press 'q' key** to cancel
5. **Window closes automatically** after capture

### Important

- ✅ Use **SAME palm** for registration and transactions
- ✅ **Good lighting** helps accuracy
- ✅ Keep palm **flat and still**
- ✅ **Press 's'** when palm is clearly visible

## 🔍 Monitor Training

```bash
# View live progress
tail -f training_live.log

# Check current epoch
tail -5 training_live.log | grep "Epoch"

# Check if training is running
ps aux | grep train_model.py
```

## 🎯 Key Points

1. **No File Upload**: Images are captured from camera in real-time
2. **Same Palm Required**: Must use same hand for all transactions
3. **Dual Authentication**: PIN + Palm verification required
4. **Training Improves Accuracy**: Wait for training to complete

## ❓ Common Issues

### Camera Not Opening
- Check camera permissions (macOS: System Preferences → Security → Camera)
- Close other apps using camera
- Restart the app

### Palm Authentication Fails
- Use the same palm as registration
- Ensure good lighting
- Check similarity score in error message
- Re-register if needed

### Training Taking Too Long
- Normal: 30-60 minutes
- Check progress with: `tail -f training_live.log`
- Can reduce epochs in train_model.py if needed

## 📋 Navigation

- **Home**: http://localhost:5000
- **Register**: http://localhost:5000/register
- **Deposit**: http://localhost:5000/deposit
- **Withdraw**: http://localhost:5000/withdraw
- **Transfer**: http://localhost:5000/transfer
- **Balance**: http://localhost:5000/balance
- **History**: http://localhost:5000/history
- **Users**: http://localhost:5000/users

---

**That's it! You're ready to use PalmPay.** 🎉

