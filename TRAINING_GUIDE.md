# Palm Recognition Model Training Guide

## Overview

This guide explains how to train the palm recognition model for accurate palm authentication in the PalmPay system. The improved training script uses **contrastive learning** with **data augmentation** to achieve high accuracy.

## Key Improvements

1. **Contrastive Learning**: Trains the model to distinguish between same and different palms
2. **Data Augmentation**: Improves robustness to lighting, rotation, and other variations
3. **Image Enhancement**: Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) for better feature extraction
4. **Better Architecture**: Uses EfficientNetB0 (or MobileNetV2 as fallback) with batch normalization
5. **Fine-tuning**: Two-stage training with fine-tuning for better accuracy
6. **Automatic Evaluation**: Calculates similarity metrics and recommends optimal threshold

## Training Steps

### 1. Install Dependencies

Make sure you have all required packages:

```bash
pip install -r requirements.txt
```

Required packages:
- TensorFlow >= 2.10.0
- OpenCV
- NumPy
- scikit-learn
- Flask

### 2. Prepare Training Data

Organize your palm images in the following structure:

```
Files/
├── Train/          # Training images (recommended: 20+ images)
│   ├── IMG_0371.JPG
│   ├── IMG_0372.JPG
│   └── ...
└── Valid/          # Validation images (recommended: 10+ images)
    ├── IMG_0378.JPG
    ├── IMG_0379.JPG
    └── ...
```

**Important Notes:**
- More training data = better accuracy
- Images should be clear and well-lit
- Consistent palm positioning helps
- Include images from different lighting conditions if possible

### 3. Test Training Setup

Before training, verify your setup:

```bash
python test_training.py
```

This will check:
- Required directories exist
- Images are found
- Dependencies are installed

### 4. Train the Model

Run the training script:

```bash
python train_model.py
```

**Training Process:**
1. Loads images from `Files/Train/` and `Files/Valid/`
2. Enhances images using CLAHE
3. Creates positive pairs (same image with augmentation) and negative pairs (different images)
4. Trains the model using contrastive learning
5. Fine-tunes the model with lower learning rate
6. Evaluates the model and recommends optimal threshold
7. Saves the model to `palm_feature_extractor.h5`

**Training Time:**
- Depends on number of images and hardware
- Typically 30-60 minutes on CPU
- Faster on GPU (if available)
- Training will stop early if validation loss doesn't improve

### 5. Verify Model

After training, the script will:
- Calculate similarity statistics
- Recommend optimal threshold
- Show separation between same and different palms

**Expected Output:**
```
Similarity Statistics:
  Same palm (with augmentation):
    Mean: 0.92 ± 0.05
    Min: 0.85
  Different palms:
    Mean: 0.45 ± 0.15
    Max: 0.70
  Separation: 0.47

✓ Good separation! Model should perform well.

Recommended threshold: 0.80
```

### 6. Adjust Threshold (if needed)

If the recommended threshold is different from the default (0.75), update it in `palm_recognition.py`:

```python
def get_palm_recognizer(threshold=0.80):  # Update threshold here
    ...
```

**Threshold Guidelines:**
- **Higher threshold (0.85-0.95)**: More strict, fewer false positives, but may reject valid palms
- **Lower threshold (0.70-0.80)**: More lenient, fewer false negatives, but may accept wrong palms
- **Optimal**: Balance between false positives and false negatives

## Model Architecture

- **Base Model**: EfficientNetB0 (or MobileNetV2 as fallback)
- **Feature Dimension**: 128
- **Normalization**: L2 normalization for cosine similarity
- **Loss Function**: Contrastive loss with margin = 1.0

## Data Augmentation

The training script applies the following augmentations:

1. **Rotation**: ±15 degrees
2. **Brightness**: ±20% adjustment
3. **Contrast**: ±10% adjustment
4. **Gaussian Noise**: Small amount of noise
5. **Zoom**: 95-105% scaling
6. **Horizontal Flip**: Small probability

## Troubleshooting

### Low Accuracy

1. **Add more training data**: More images improve accuracy
2. **Check image quality**: Ensure images are clear and well-lit
3. **Adjust threshold**: Try the recommended threshold from training
4. **Retrain with more epochs**: Increase EPOCHS in `train_model.py`
5. **Use more augmentation**: Increase `num_pairs_per_image` in training

### Training Errors

1. **Out of memory**: Reduce BATCH_SIZE in `train_model.py`
2. **No images found**: Check directory structure and file names
3. **TensorFlow errors**: Update TensorFlow: `pip install --upgrade tensorflow`
4. **Model not saving**: Check disk space and permissions

### Poor Separation

If similarity statistics show poor separation:
1. **More training data**: Add more diverse palm images
2. **Better quality images**: Ensure images are clear and consistent
3. **Longer training**: Increase EPOCHS or reduce early stopping patience
4. **Adjust margin**: Try different MARGIN values (0.5, 1.0, 1.5)

## Using the Trained Model

After training, the model is automatically used by `app.py`:

1. **Registration**: Palm features are extracted and stored
2. **Transactions**: Palm is verified against stored features
3. **Verification**: Uses cosine similarity with the trained threshold

## Performance Tips

1. **Consistent positioning**: Users should place palm in similar position
2. **Good lighting**: Ensure adequate lighting during capture
3. **Same hand**: Use the same hand (left/right) for registration and transactions
4. **Clean background**: Avoid cluttered backgrounds
5. **Regular retraining**: Retrain with more data as you add users

## Next Steps

1. Train the model with your palm dataset
2. Test with registered users
3. Adjust threshold based on results
4. Add more training data if needed
5. Retrain periodically for better accuracy

## Support

For issues or questions:
1. Check the training output for error messages
2. Verify image quality and quantity
3. Review similarity statistics
4. Adjust threshold and retrain if needed

---

**Note**: The model improves with more training data. Start with available images and add more as you collect palm images from users.

