# 💰 PalmPay - Transfer Money Guide

## ✅ Your Model is Already Trained!

The model `palm_feature_extractor.h5` is already trained and ready to use. You don't need to upload or train images - the system captures images from your camera automatically during operations.

## 🚀 How to Transfer Money Between Registered Users

### Step 1: Check Registered Users

Open your browser and go to:
```
http://localhost:5001/users
```

Or check via terminal:
```bash
cd /Users/macbook/Downloads/PalmPay-main
python3 -c "import sqlite3; conn = sqlite3.connect('palm_pay.db'); c = conn.cursor(); c.execute('SELECT name, account_number, balance FROM users'); [print(f'{row[0]}: {row[1]} - Balance: {row[2]}') for row in c.fetchall()]; conn.close()"
```

### Step 2: Deposit Money to Sender Account (if needed)

1. Go to: **http://localhost:5001/deposit**
2. Fill in:
   - **Account Number**: (sender's account, e.g., "ACC001")
   - **PIN**: (sender's 4-digit PIN)
   - **Amount**: (e.g., "1000")
3. Click "Deposit"
4. **Camera opens** → Position sender's palm → Wait 3 seconds (or press 's')
5. Money deposited successfully

### Step 3: Transfer Money

1. Go to: **http://localhost:5001/transfer**
2. Fill in the form:
   ```
   From Account: ACC001        (sender's account)
   PIN: 1234                   (sender's PIN)
   To Account: ACC002          (receiver's account)
   Amount: 500                 (amount to transfer)
   ```
3. Click "Transfer"
4. **Camera opens** → Position sender's palm → Wait 3 seconds (or press 's')
5. System verifies sender's palm
6. If verified: Money transferred successfully!
7. If not verified: Check similarity score in error message

## 📋 Complete Transfer Process

```
Sender Account (ACC001)          Receiver Account (ACC002)
────────────────────────          ────────────────────────
1. Has balance: $1000             1. Has balance: $500
2. Initiates transfer: $500       2. Receives: $500
3. Verifies palm                  3. No palm needed
4. New balance: $500              4. New balance: $1000
```

## 🎯 Quick Commands

### Check Users and Balances
```bash
cd /Users/macbook/Downloads/PalmPay-main
python3 -c "
import sqlite3
conn = sqlite3.connect('palm_pay.db')
c = conn.cursor()
c.execute('SELECT name, account_number, balance FROM users')
print('Registered Users:')
for name, acc, balance in c.fetchall():
    print(f'  {name} ({acc}): \${balance:.2f}')
conn.close()
"
```

### Deposit Money (Terminal)
Open browser: http://localhost:5001/deposit

### Transfer Money (Terminal)
Open browser: http://localhost:5001/transfer

### View Transaction History
Open browser: http://localhost:5001/history

## 📸 Palm Verification Process

### For Transfer Operations:

1. **Sender must verify palm**:
   - Camera opens automatically
   - Position the SAME palm used during registration
   - Wait 3 seconds OR press 's' key
   - System verifies similarity (must be >= 75%)
   - If verified: Transfer proceeds
   - If not verified: Transfer denied

2. **Receiver doesn't need to verify**:
   - Only sender needs palm verification
   - Receiver gets money automatically
   - No camera needed for receiver

## 🔍 Understanding Similarity Scores

- **75-100%**: ✅ Transfer approved
- **50-75%**: ⚠️ Below threshold (may need better lighting/position)
- **<50%**: ❌ Too different (wrong palm or different person)

## 🛠️ Troubleshooting

### Transfer Fails - Palm Not Verified

**Solution**:
1. Use the SAME palm as registration
2. Ensure good lighting
3. Keep palm in similar position
4. Check similarity score in error message
5. If similarity is close (e.g., 73%), try again with better lighting

### Insufficient Balance

**Solution**:
1. Deposit more money to sender account
2. Go to: http://localhost:5001/deposit
3. Verify palm and deposit amount
4. Try transfer again

### Receiver Account Not Found

**Solution**:
1. Make sure receiver is registered
2. Check account number is correct
3. Go to: http://localhost:5001/users to see all registered users

## ✅ Example Transfer Workflow

### Scenario: Transfer $500 from ACC001 to ACC002

1. **Check balances**:
   - Go to: http://localhost:5001/users
   - Note current balances

2. **Ensure sender has balance**:
   - If ACC001 has < $500, deposit money first
   - Go to: http://localhost:5001/deposit

3. **Perform transfer**:
   - Go to: http://localhost:5001/transfer
   - From Account: ACC001
   - PIN: 1234 (sender's PIN)
   - To Account: ACC002
   - Amount: 500
   - Click "Transfer"

4. **Verify palm**:
   - Camera opens
   - Position sender's palm (ACC001's palm)
   - Wait 3 seconds
   - System verifies

5. **Check result**:
   - If successful: See success message with new balances
   - Check: http://localhost:5001/users for updated balances
   - View history: http://localhost:5001/history

## 🎯 Quick Reference

### Transfer Money
```
URL: http://localhost:5001/transfer

Required:
- Sender Account Number
- Sender PIN (4 digits)
- Receiver Account Number
- Amount

Process:
1. Fill form → Click Transfer
2. Camera opens → Verify sender's palm
3. Transfer completes automatically
```

### Deposit Money
```
URL: http://localhost:5001/deposit

Required:
- Account Number
- PIN
- Amount

Process:
1. Fill form → Click Deposit
2. Camera opens → Verify palm
3. Money added to account
```

### Check Balance
```
URL: http://localhost:5001/balance

Required:
- Account Number
- PIN

No palm verification needed for balance check!
```

---

## 🚀 Ready to Transfer!

Your trained model is ready. Just:
1. Make sure both users are registered
2. Ensure sender has sufficient balance
3. Go to transfer page
4. Verify sender's palm
5. Transfer completes!

---

**Note**: The model `palm_feature_extractor.h5` is already trained and working. No need to upload or train images - it captures from camera automatically during operations!

