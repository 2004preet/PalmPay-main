#!/bin/bash

# PalmPay Setup and Run Script for macOS

echo "=========================================="
echo "PalmPay - Setup and Run Script"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "Step 1: Installing dependencies..."
pip3 install -r requirements.txt

echo ""
echo "Step 2: Training the model (this may take a few minutes)..."
python3 train_model.py

echo ""
echo "Step 3: Starting the Flask application..."
echo "The app will be available at http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""
python3 app.py

