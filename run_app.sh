#!/bin/bash
# PalmPay - Run App in Terminal
# This script starts the Flask app and shows output in terminal

cd "$(dirname "$0")"

echo "=========================================="
echo "  PalmPay - Starting Application"
echo "=========================================="
echo ""
echo "🌐 App will be available at: http://localhost:5001"
echo ""
echo "📋 Available pages:"
echo "  - Home: http://localhost:5001"
echo "  - Register: http://localhost:5001/register"
echo "  - Deposit: http://localhost:5001/deposit"
echo "  - Withdraw: http://localhost:5001/withdraw"
echo "  - Transfer: http://localhost:5001/transfer"
echo "  - Balance: http://localhost:5001/balance"
echo "  - History: http://localhost:5001/history"
echo ""
echo "💡 Press Ctrl+C to stop the app"
echo ""
echo "=========================================="
echo ""

# Start the Flask app
python3 app.py


