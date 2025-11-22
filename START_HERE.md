# 🚀 PalmPay - Start Here

## ✅ Training Complete!

**Status**: ✅ Training completed successfully!
**Model**: `palm_feature_extractor.h5` (19MB)
**Status**: Ready to use!

---

## 🎯 Quick Start (3 Steps)

### Step 1: Start the App

```bash
python3 app.py
```

Open browser: **http://localhost:5000**

### Step 2: Register a User

1. Go to: **http://localhost:5000/register**
2. Fill form (Name, Account Number, PIN: 1234)
3. Click "Register"
4. **Camera opens** → Position palm → **Press 's'** → Done

### Step 3: Perform Operations

1. **Deposit**: Go to deposit page → Fill form → Camera opens → Press 's'
2. **Withdraw**: Go to withdraw page → Fill form → Camera opens → Press 's'
3. **Transfer**: Go to transfer page → Fill form → Camera opens → Press 's'

---

## 📸 How Image "Upload" Works

### Important: No File Upload!

**PalmPay uses camera capture, not file upload.**

### How It Works:

1. **Camera opens automatically** when you click a button
2. **Position your palm** in front of camera
3. **Press 's' key** on keyboard to capture
4. **Camera closes automatically**
5. **Image is processed immediately**

### Camera Controls:
- **'s' key**: Save/capture image
- **'q' key**: Cancel
- **Window**: Closes automatically after capture

---

## 🎯 Complete Operation Guide

### Registration (First Time)

```
1. Go to: http://localhost:5000/register
2. Fill form:
   - Name: John Doe
   - Account Number: ACC001
   - PIN: 1234
   - (Other fields optional)
3. Click "Register"
4. Camera opens → Position palm → Press 's'
5. Registration complete!
```

### Deposit Money

```
1. Go to: http://localhost:5000/deposit
2. Fill form:
   - Account Number: ACC001
   - PIN: 1234
   - Amount: 1000
3. Click "Deposit"
4. Camera opens → Position SAME palm → Press 's'
5. Deposit successful!
```

### Withdraw Money

```
1. Go to: http://localhost:5000/withdraw
2. Fill form:
   - Account Number: ACC001
   - PIN: 1234
   - Amount: 500
3. Click "Withdraw"
4. Camera opens → Position SAME palm → Press 's'
5. Withdrawal successful!
```

### Transfer Money

```
1. Register recipient (Account: ACC002)
2. Go to: http://localhost:5000/transfer
3. Fill form:
   - From Account: ACC001
   - PIN: 1234
   - To Account: ACC002
   - Amount: 500
4. Click "Transfer"
5. Camera opens → Position SAME palm → Press 's'
6. Transfer successful!
```

### Check Balance

```
1. Go to: http://localhost:5000/balance
2. Fill form:
   - Account Number: ACC001
   - PIN: 1234
3. Click "Check Balance"
4. Balance displayed (no camera needed)
```

### View History

```
1. Go to: http://localhost:5000/history
2. Fill form:
   - Account Number: ACC001
   - PIN: 1234
3. Click "View History"
4. Transactions displayed (no camera needed)
```

---

## 📋 Key Points

### Image Capture
- ✅ **Camera opens automatically**
- ✅ **Press 's' to capture**
- ✅ **Press 'q' to cancel**
- ✅ **Window closes automatically**

### Palm Requirements
- ✅ **Use SAME palm** for registration and transactions
- ✅ **Good lighting** helps accuracy
- ✅ **Keep palm still** during capture
- ✅ **Clear view** of palm

### Authentication
- ✅ **PIN required**: 4-digit PIN
- ✅ **Palm required**: Palm verification
- ✅ **Threshold**: 75% similarity (default)
- ✅ **Dual security**: PIN + Palm

---

## 🛠️ Troubleshooting

### Camera Not Opening
1. Check camera permissions (System Preferences → Security → Camera)
2. Close other apps using camera
3. Restart the app

### Palm Authentication Fails
1. Use the same palm as registration
2. Ensure good lighting
3. Keep palm in similar position
4. Check similarity score in error message

### Training Results
- Model saved: `palm_feature_extractor.h5`
- Default threshold: 0.75 (75%)
- Recommended threshold: 0.95 (may be too strict)
- **Use default threshold (0.75) for better results**

---

## 📊 Training Results

### Model Status
- ✅ **Training**: Completed successfully
- ✅ **Model**: `palm_feature_extractor.h5` (19MB)
- ✅ **Status**: Ready to use

### Threshold Settings
- **Default**: 0.75 (75%) - Recommended for use
- **Recommended**: 0.95 (95%) - May be too strict
- **Adjustment**: Can be changed in `palm_recognition.py`

### Usage
- **Default threshold (0.75)**: Good balance between security and usability
- **Higher threshold (0.95)**: More secure but may reject valid palms
- **Lower threshold (0.70)**: More lenient but may accept wrong palms

---

## 🎯 Navigation

### App URLs
- **Home**: http://localhost:5000
- **Register**: http://localhost:5000/register
- **Deposit**: http://localhost:5000/deposit
- **Withdraw**: http://localhost:5000/withdraw
- **Transfer**: http://localhost:5000/transfer
- **Balance**: http://localhost:5000/balance
- **History**: http://localhost:5000/history
- **Users**: http://localhost:5000/users

### Commands
```bash
# Start app
python3 app.py

# Monitor training (if needed)
tail -f training_live.log

# Check model
ls -lh palm_feature_extractor.h5
```

---

## ✅ Checklist

### Before Starting
- [x] Training complete ✅
- [x] Model saved ✅
- [ ] App running (`python3 app.py`)
- [ ] Camera permissions granted
- [ ] Browser open (http://localhost:5000)

### First Use
- [ ] Registered first user
- [ ] Tested deposit
- [ ] Tested withdraw
- [ ] Tested transfer
- [ ] Checked balance
- [ ] Viewed history

---

## 🎉 You're Ready!

### Next Steps:
1. **Start the app**: `python3 app.py`
2. **Register users**: Go to register page
3. **Perform transactions**: Deposit, withdraw, transfer
4. **Check balance**: View account balance
5. **View history**: See transaction history

### Remember:
- ✅ **Use SAME palm** for all operations
- ✅ **Press 's'** to capture palm image
- ✅ **Good lighting** helps accuracy
- ✅ **Keep palm still** during capture

---

## 📚 Documentation

For more details, see:
- **USER_GUIDE.md**: Complete user guide
- **QUICK_START.md**: Quick start guide
- **HOW_TO_USE.md**: Detailed usage instructions
- **COMPLETE_GUIDE.md**: Complete operation guide
- **TRAINING_GUIDE.md**: Training instructions

---

**Happy Banking with PalmPay!** 🎉

