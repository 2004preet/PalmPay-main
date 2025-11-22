# PalmPay Setup Instructions

## Quick Start

### 1. Install Dependencies
```bash
cd PalmPay-main
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
```bash
python train_model.py
```
This creates the palm recognition model. If you skip this step, the app will automatically create a default model when you first run it.

### 3. Run the Application
```bash
python app.py
```

Open your browser to `http://localhost:5000`

## How It Works

### Registration Process
1. Fill in registration form (name, account number, 4-digit PIN, etc.)
2. When prompted, position your palm in front of the camera
3. Press 's' to save your palm image
4. System extracts and stores your palm features

### Transaction Process (Deposit/Withdraw/Transfer)
1. Enter your account number and PIN
2. Enter transaction details (amount, etc.)
3. When prompted, position your palm in front of the camera
4. Press 's' to verify your palm
5. System compares your palm with the stored features
6. If match is found (similarity >= 75%), transaction is processed

## Important Notes

- **Palm Consistency**: Use the same palm (left or right) during registration and transactions
- **Lighting**: Ensure good lighting when capturing palm images
- **Position**: Try to position your palm similarly each time
- **Threshold**: The similarity threshold is set to 75%. You can adjust it in `palm_recognition.py` if needed (lower = more permissive, higher = more strict)

## Model Details

- **Architecture**: MobileNetV2 (pre-trained on ImageNet) + custom feature extraction layers
- **Feature Size**: 128-dimensional normalized vectors
- **Similarity Metric**: Cosine similarity
- **Default Threshold**: 75%

## Troubleshooting

### Model Not Found
The app will automatically create a default model if `palm_feature_extractor.h5` is not found. For better accuracy, run `train_model.py` first.

### Low Recognition Accuracy
1. Ensure consistent palm positioning
2. Use good lighting
3. Use the same palm (left/right) consistently
4. Try adjusting the threshold in `palm_recognition.py`
5. Re-register with clearer palm images

### Camera Issues
- Grant camera permissions to your terminal/IDE
- Ensure camera is connected and working
- On macOS, you may need to grant permissions in System Preferences

## Files

- `app.py` - Main Flask application
- `palm_recognition.py` - Palm recognition module
- `train_model.py` - Model training script
- `palm_feature_extractor.h5` - Trained model (created after training)
- `palm_pay.db` - SQLite database (created automatically)
- `Files/Train/` - Training images
- `Files/Valid/` - Validation images

## Next Steps

1. Run `train_model.py` to create the model
2. Start the app with `python app.py`
3. Register users with their palm images
4. Perform transactions with palm verification

The system is now ready to use!

