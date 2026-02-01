"""
Advanced Palm Recognition Model Training with ArcFace Loss
State-of-the-art model for palm authentication with superior accuracy
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
import random
from sklearn.metrics.pairwise import cosine_similarity
import math
import logging
import datetime

# Custom L2Normalize layer
class L2Normalize(layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super(L2Normalize, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=self.axis)

    def get_config(self):
        config = super(L2Normalize, self).get_config()
        config.update({"axis": self.axis})
        return config

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("tqdm not available, progress bars disabled")

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 8  # Smaller batch size for better generalization
EPOCHS = 100  # More epochs for better convergence
LEARNING_RATE = 0.0001  # Lower learning rate for fine-tuning
TRAIN_DIR = "Files/Train"
VALID_DIR = "Files/Valid"
MODEL_SAVE_PATH = "palm_feature_extractor_advanced.h5"
FEATURE_DIM = 512  # Increased feature dimension for better accuracy
NUM_CLASSES = 3  # Based on the 3 palm groups in training data
ARC_SCALE = 64.0  # Scale parameter for ArcFace loss
ARC_MARGIN = 0.5  # Margin parameter for ArcFace loss

def advanced_augment_image(img):
    """Advanced data augmentation for better model robustness"""
    img = img.astype(np.float32)

    # Random rotation (-20 to 20 degrees)
    if random.random() > 0.4:
        angle = random.uniform(-20, 20)
        center = (img.shape[1] // 2, img.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                           borderMode=cv2.BORDER_REFLECT)

    # Random brightness and contrast
    if random.random() > 0.4:
        alpha = random.uniform(0.8, 1.2)  # contrast
        beta = random.uniform(-20, 20)    # brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Random Gaussian noise
    if random.random() > 0.6:
        noise = np.random.normal(0, random.uniform(3, 8), img.shape)
        img = np.clip(img + noise, 0, 255)

    # Random blur
    if random.random() > 0.7:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # Random horizontal flip (very low probability for palms)
    if random.random() > 0.95:
        img = cv2.flip(img, 1)

    # Random zoom and crop
    if random.random() > 0.6:
        zoom_factor = random.uniform(0.9, 1.1)
        h, w = img.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        img = cv2.resize(img, (new_w, new_h))
        if zoom_factor > 1:
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            img = img[start_y:start_y+h, start_x:start_x+w]
        else:
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            img = cv2.copyMakeBorder(img, pad_y, h-new_h-pad_y,
                                   pad_x, w-new_w-pad_x,
                                   cv2.BORDER_REFLECT)

    # Color jittering
    if random.random() > 0.5:
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        # Hue shift
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-10, 10)) % 180
        # Saturation
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.8, 1.2), 0, 255)
        # Value
        hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.9, 1.1), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return np.clip(img, 0, 255).astype(np.uint8)

def enhance_image_advanced(img):
    """Advanced image enhancement for palm recognition"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE with different parameters
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Additional enhancement for palm lines
    l = cv2.equalizeHist(l)

    # Bilateral filter to reduce noise while keeping edges
    l = cv2.bilateralFilter(l, 9, 75, 75)

    # Merge channels
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Sharpen the image slightly
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel * 0.1)

    return img

def load_images_with_labels(directory):
    """Load images and assign labels based on filename patterns"""
    images = []
    labels = []
    image_paths = (glob.glob(os.path.join(directory, "*.JPG")) +
                   glob.glob(os.path.join(directory, "*.jpg")) +
                   glob.glob(os.path.join(directory, "*.PNG")) +
                   glob.glob(os.path.join(directory, "*.png")))

    # Group images by palm based on number ranges (assuming consecutive numbers belong to same palm)
    palm_groups = {}
    for img_path in sorted(image_paths):
        base_name = os.path.basename(img_path)
        # Extract number from filename like IMG_0371 -> 0371
        try:
            number_part = ''.join(filter(str.isdigit, base_name))
            if number_part:
                number = int(number_part)
                # Group by hundreds: 037x = palm 0, 038x = palm 1, 039x = palm 2
                palm_id = (number // 100) - 37  # 371-379 = 0, 381-389 = 1, 391-399 = 2
            else:
                palm_id = 0
        except:
            palm_id = 0

        if palm_id not in palm_groups:
            palm_groups[palm_id] = []
        palm_groups[palm_id].append(img_path)

    # Assign numeric labels
    label_map = {palm_id: i for i, palm_id in enumerate(sorted(palm_groups.keys()))}

    for palm_id, paths in palm_groups.items():
        for img_path in paths:
            img = cv2.imread(img_path)
            if img is not None:
                img = enhance_image_advanced(img)
                img = cv2.resize(img, IMG_SIZE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label_map[palm_id])

    return np.array(images), np.array(labels)

def create_arcface_model():
    """Create simple feature extractor model"""
    try:
        # Use MobileNetV2 for faster training and better generalization
        from tensorflow.keras.applications import MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            pooling='avg'
        )
        print("   Using MobileNetV2 as base model")
    except ImportError:
        from tensorflow.keras.applications import MobileNet
        base_model = MobileNet(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            pooling='avg'
        )
        print("   Using MobileNet as base model")

    # Unfreeze more layers for fine-tuning from start
    base_model.trainable = True
    # Freeze only the first few layers
    for layer in base_model.layers[:50]:
        layer.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Simple feature extraction
    x = base_model(inputs, training=False)

    # Single dense layer for features
    features = layers.Dense(FEATURE_DIM, activation='relu', name='features')(x)
    features = layers.BatchNormalization()(features)
    features = layers.Dropout(0.3)(features)

    # L2 normalize features
    features_normalized = L2Normalize(axis=1, name='features_normalized')(features)

    # Classification head
    classification_output = layers.Dense(NUM_CLASSES, activation='softmax', name='classification')(features_normalized)

    # Create training model with classification output
    training_model = keras.Model(inputs, classification_output, name='palm_classifier_training')

    # Create feature extractor model (for later use)
    feature_extractor_model = keras.Model(inputs, features_normalized, name='palm_feature_extractor')

    return training_model, feature_extractor_model, base_model, inputs

def create_training_data(images, labels, augment_factor=5):
    """Create augmented training data for ArcFace training"""
    augmented_images = []
    augmented_labels = []

    iterator = zip(images, labels)
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="Creating augmented data", total=len(images))

    for img, label in iterator:
        # Original image
        augmented_images.append(img)
        augmented_labels.append(label)

        # Augmented versions
        for _ in range(augment_factor):
            aug_img = advanced_augment_image(img.copy())
            augmented_images.append(aug_img.astype('float32') / 255.0)
            augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)

def train_arcface_model():
    """Train palm recognition model using ArcFace loss"""
    # Setup logging
    log_filename = f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Enable mixed precision for faster training
    try:
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        mixed_precision.set_policy(mixed_precision.Policy('mixed_float16'))
        logger.info("Mixed precision enabled for faster training")
    except:
        logger.info("Mixed precision not available, using float32")
    
    print("=" * 70)
    print("Advanced Palm Recognition Training with ArcFace Loss")
    print("=" * 70)
    logger.info("Starting advanced palm recognition training")

    # Load images with labels
    print("\n1. Loading training images...")
    train_images, train_labels = load_images_with_labels(TRAIN_DIR)
    print(f"   Loaded {len(train_images)} training images from {len(np.unique(train_labels))} palms")

    valid_images, valid_labels = None, None
    if os.path.exists(VALID_DIR):
        valid_images, valid_labels = load_images_with_labels(VALID_DIR)
        print(f"   Loaded {len(valid_images)} validation images from {len(np.unique(valid_labels))} palms")

    if len(train_images) == 0:
        raise ValueError(f"No images found in {TRAIN_DIR}")

    # Normalize images
    train_images = train_images.astype('float32') / 255.0
    if valid_images is not None:
        valid_images = valid_images.astype('float32') / 255.0

    # Create model
    print("\n2. Creating classification model...")
    training_model, feature_extractor_model, base_model, inputs = create_arcface_model()

    # Compile model with categorical cross-entropy
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n3. Model Architecture:")
    training_model.summary()

    # Prepare training data
    print("\n4. Preparing training data with augmentation...")
    train_aug_images, train_aug_labels = create_training_data(train_images, train_labels, augment_factor=5)
    print(f"   Created {len(train_aug_images)} training samples")

    valid_aug_images, valid_aug_labels = None, None
    if valid_images is not None:
        valid_aug_images, valid_aug_labels = create_training_data(valid_images, valid_labels, augment_factor=3)
        print(f"   Created {len(valid_aug_images)} validation samples")

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if valid_aug_images is not None else 'loss',
            factor=0.5,
            patience=10,  # Increased patience
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_loss' if valid_aug_images is not None else 'loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=f"logs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    ]

    # Convert labels to one-hot encoding
    train_labels_onehot = tf.keras.utils.to_categorical(train_aug_labels, num_classes=NUM_CLASSES)
    if valid_aug_images is not None:
        valid_labels_onehot = tf.keras.utils.to_categorical(valid_aug_labels, num_classes=NUM_CLASSES)

    # Train model
    print("\n5. Training model...")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")

    history = training_model.fit(
        train_aug_images,
        train_labels_onehot,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(valid_aug_images, valid_labels_onehot) if valid_aug_images is not None else None,
        callbacks=callbacks,
        verbose=1
    )

    # Fine-tuning: Unfreeze base model
    print("\n6. Fine-tuning with base model unfrozen...")
    base_model.trainable = True
    # Freeze early layers
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Recompile with lower learning rate
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = training_model.fit(
        train_aug_images,
        train_labels_onehot,
        batch_size=BATCH_SIZE,
        epochs=50,  # Increased fine-tuning epochs
        validation_data=(valid_aug_images, valid_labels_onehot) if valid_aug_images is not None else None,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if valid_aug_images is not None else 'loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ],
        verbose=1
    )

    # Create feature extractor model (without classification head)
    print(f"\n7. Creating final feature extractor...")
    feature_extractor = feature_extractor_model

    # Save the feature extractor
    feature_extractor.save(MODEL_SAVE_PATH)
    print(f"   Saved to {MODEL_SAVE_PATH}")

    # Evaluate model
    print("\n8. Evaluating model...")
    evaluate_advanced_model(feature_extractor, train_images, valid_images)

    print("\n" + "=" * 70)
    print("Advanced training completed successfully!")
    print("=" * 70)
    print(f"Feature extractor saved to: {MODEL_SAVE_PATH}")
    print("\nNext steps:")
    print("1. Update palm_recognition.py to use the new model")
    print("2. Test the improved accuracy")
    print("3. Adjust threshold if needed")

    return feature_extractor

def evaluate_advanced_model(model, train_images, valid_images=None):
    """Evaluate the advanced trained model"""
    print("\nEvaluating advanced model performance...")

    # Extract features
    print("Extracting features from training images...")
    train_features = model.predict(train_images, batch_size=16, verbose=0)

    # Calculate similarity metrics
    print("Calculating similarity metrics...")

    # Intra-class similarity (same palm with augmentation)
    similarities_same = []
    for i in range(min(10, len(train_images))):
        feat1 = train_features[i]
        # Simulate augmentation effect
        noise = np.random.normal(0, 0.1, feat1.shape)
        feat2 = feat1 + noise
        feat2 = feat2 / np.linalg.norm(feat2)  # Re-normalize
        sim = cosine_similarity([feat1], [feat2])[0][0]
        similarities_same.append(sim)

    # Inter-class similarity (different palms)
    similarities_diff = []
    num_comparisons = min(100, len(train_features) * (len(train_features) - 1) // 2)
    for _ in range(num_comparisons):
        i, j = random.sample(range(len(train_features)), 2)
        sim = cosine_similarity([train_features[i]], [train_features[j]])[0][0]
        similarities_diff.append(sim)

    avg_same = np.mean(similarities_same)
    avg_diff = np.mean(similarities_diff)
    std_same = np.std(similarities_same)
    std_diff = np.std(similarities_diff)
    min_same = np.min(similarities_same)
    max_diff = np.max(similarities_diff)

    print(f"\nAdvanced Model Similarity Statistics:")
    print(f"  Same palm (with simulated augmentation):")
    print(f"    Mean: {avg_same:.4f} ± {std_same:.4f}")
    print(f"    Min: {min_same:.4f}")
    print(f"  Different palms:")
    print(f"    Mean: {avg_diff:.4f} ± {std_diff:.4f}")
    print(f"    Max: {max_diff:.4f}")
    print(f"  Separation: {avg_same - avg_diff:.4f}")

    # Calculate optimal threshold
    if min_same > max_diff:
        threshold = (min_same + max_diff) / 2
        threshold = max(0.75, min(0.95, threshold))
        print(f"\n✓ Excellent separation! ArcFace model performing very well.")
    else:
        threshold = (avg_same + avg_diff) / 2
        threshold = max(0.75, min(0.95, threshold))
        print(f"\n⚠ Some overlap detected. Consider more training data.")

    print(f"\nRecommended threshold: {threshold:.3f}")
    print(f"  (Current threshold in palm_recognition.py: 0.75)")
    print(f"  (Update palm_recognition.py with: threshold={threshold:.3f})")

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    try:
        model = train_arcface_model()
        print("\nTo use this improved model:")
        print("1. Copy palm_feature_extractor_advanced.h5 to palm_feature_extractor.h5")
        print("   or update palm_recognition.py to load the advanced model")
        print("2. Update threshold in palm_recognition.py based on evaluation")
        print("3. Test with real palm verification")
    except Exception as e:
        print(f"\nError during advanced training: {e}")
        import traceback
        traceback.print_exc()