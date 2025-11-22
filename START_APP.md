# 🚀 Start PalmPay App in Terminal

## Quick Start

Open your terminal and run:

```bash
cd /Users/macbook/Downloads/PalmPay-main
python3 app.py
```

## What You'll See

When you run the app, you'll see output like this:

```
2025-11-12 23:13:15.507029: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized...
Loaded palm recognition model from palm_feature_extractor.h5
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5001
 * Running on http://192.168.x.x:5001
Press CTRL+C to quit
 * Restarting with stat
```

## Access the App

Once you see "Running on http://127.0.0.1:5001", open your browser and go to:

**http://localhost:5001**

## Available Pages

- **Home**: http://localhost:5001
- **Register**: http://localhost:5001/register
- **Deposit**: http://localhost:5001/deposit
- **Withdraw**: http://localhost:5001/withdraw
- **Transfer**: http://localhost:5001/transfer
- **Balance**: http://localhost:5001/balance
- **History**: http://localhost:5001/history

## Terminal Output

While the app is running, you'll see:
- Flask request logs
- Palm capture messages
- Debug information
- Error messages (if any)

Example output when using the app:
```
127.0.0.1 - - [12/Nov/2025 23:15:30] "GET /register HTTP/1.1" 200 -
Capturing palm image... Please position your palm in front of the camera.
Capturing in 4 seconds... Please position your palm.
Capturing in 3 seconds... Please position your palm.
Capturing in 2 seconds... Please position your palm.
Capturing in 1 seconds... Please position your palm.
Palm image captured automatically after 5.0 seconds
127.0.0.1 - - [12/Nov/2025 23:15:35] "POST /register HTTP/1.1" 200 -
```

## Stop the App

Press **Ctrl+C** in the terminal to stop the app.

## Alternative: Use the Script

You can also use the provided script:

```bash
cd /Users/macbook/Downloads/PalmPay-main
./run_app.sh
```

This script shows a nice header and then starts the app.

## Troubleshooting

### Port Already in Use
If you see "Address already in use", stop any existing app:
```bash
pkill -f "python3 app.py"
```

### Camera Not Working
- Check camera permissions in System Preferences
- Make sure no other app is using the camera
- Check that your camera is connected

### Model Not Found
If you see "Model not found", make sure `palm_feature_extractor.h5` exists:
```bash
ls -lh palm_feature_extractor.h5
```

## Enjoy!

Once the app is running, you can:
1. Register users with palm images
2. Perform transactions (deposit, withdraw, transfer)
3. Check balances
4. View transaction history

All operations will show output in your terminal!


