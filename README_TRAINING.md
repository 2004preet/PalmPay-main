# PalmPay - Palm Recognition Training Guide

## Overview
This PalmPay system uses deep learning for palm recognition to authenticate UPI transactions. The system extracts palm features during registration and verifies them during transactions (deposit, withdraw, transfer).

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional but Recommended)
The model can work with a pre-trained MobileNetV2, but training on your palm dataset will improve accuracy:

```bash
python train_model.py
```

This will:
- Load images from `Files/Train/` and `Files/Valid/`
- Train a feature extractor using transfer learning
- Save the model to `palm_feature_extractor.h5`

**Note:** If you don't train the model, the system will use a default pre-trained model (which still works, but may have lower accuracy).

### 3. Run the Application
```bash
python app.py
```

The Flask app will start on `http://localhost:5000`

## How It Works

### Registration
1. User fills in registration form (name, account number, PIN, etc.)
2. System captures palm image from camera
3. System extracts palm features using the trained model
4. Features are stored in the database along with user information

### Transactions (Deposit, Withdraw, Transfer)
1. User enters account number and PIN
2. System verifies PIN
3. System captures palm image from camera
4. System extracts features from the new palm image
5. System compares new features with stored features using cosine similarity
6. If similarity >= threshold (75%), transaction is approved
7. Transaction is processed

## Model Architecture
- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Feature Extraction:** 128-dimensional feature vector
- **Similarity Metric:** Cosine similarity
- **Threshold:** 75% (adjustable in `palm_recognition.py`)

## Improving Accuracy

1. **Train the Model:** Run `train_model.py` with your palm dataset
2. **More Training Data:** Add more palm images to `Files/Train/` and `Files/Valid/`
3. **Adjust Threshold:** Modify the threshold in `palm_recognition.py` (lower = more permissive, higher = more strict)
4. **Consistent Palm Position:** Ensure users place their palm in a similar position during registration and transactions
5. **Good Lighting:** Ensure adequate lighting when capturing palm images

## File Structure
```
PalmPay-main/
├── app.py                    # Main Flask application
├── palm_recognition.py       # Palm recognition module
├── train_model.py           # Training script
├── palm_feature_extractor.h5 # Trained model (created after training)
├── palm_pay.db              # SQLite database
├── requirements.txt         # Python dependencies
└── Files/
    ├── Train/               # Training images
    └── Valid/               # Validation images
```

## Troubleshooting

### Model Not Found
If you see "Model not found" message, the system will automatically create a default model. For better accuracy, train the model first using `train_model.py`.

### Low Accuracy
- Ensure you've trained the model with your dataset
- Check that palm images are clear and well-lit
- Try adjusting the threshold in `palm_recognition.py`
- Ensure users use the same palm (left/right) during registration and transactions

### Camera Issues
- Ensure your camera is connected and accessible
- On macOS, you may need to grant camera permissions
- Try using a different camera index if you have multiple cameras

## Notes
- The system requires both PIN and palm verification for transactions
- Palm features are stored as 128-dimensional vectors in the database
- The model uses transfer learning for efficient training
- The system works with or without a trained model (uses default if not trained)

