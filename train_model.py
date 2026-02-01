"""
Advanced Training script for Palm Recognition Model
Trains a feature extractor using contrastive learning with data augmentation
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
try:
    from tensorflow.keras.applications import EfficientNetB0
    HAS_EFFICIENTNET = True
except ImportError:
    HAS_EFFICIENTNET = False
    print("EfficientNetB0 not available, will use MobileNetV2")
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import glob
import random
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
TRAIN_DIR = "Files/Train"
VALID_DIR = "Files/Valid"
MODEL_SAVE_PATH = "palm_feature_extractor.h5"
FEATURE_DIM = 128
MARGIN = 1.0  # Margin for contrastive loss (appropriate for normalized features)

def augment_image(img):
    """Apply data augmentation to improve model robustness"""
    # Convert to float
    img = img.astype(np.float32)
    
    # Random rotation (-15 to 15 degrees)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        center = (img.shape[1] // 2, img.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), 
                           borderMode=cv2.BORDER_REFLECT)
    
    # Random brightness adjustment
    if random.random() > 0.5:
        brightness = random.uniform(0.8, 1.2)
        img = np.clip(img * brightness, 0, 255)
    
    # Random contrast adjustment
    if random.random() > 0.5:
        contrast = random.uniform(0.9, 1.1)
        img = np.clip((img - 127.5) * contrast + 127.5, 0, 255)
    
    # Random Gaussian noise
    if random.random() > 0.7:
        noise = np.random.normal(0, 5, img.shape)
        img = np.clip(img + noise, 0, 255)
    
    # Random horizontal flip (small probability for palm)
    if random.random() > 0.9:
        img = cv2.flip(img, 1)
    
    # Random zoom
    if random.random() > 0.7:
        zoom_factor = random.uniform(0.95, 1.05)
        h, w = img.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        img = cv2.resize(img, (new_w, new_h))
        if zoom_factor > 1:
            # Crop center
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            img = img[start_y:start_y+h, start_x:start_x+w]
        else:
            # Pad
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            img = cv2.copyMakeBorder(img, pad_y, h-new_h-pad_y, 
                                   pad_x, w-new_w-pad_x, 
                                   cv2.BORDER_REFLECT)
    
    return img.astype(np.uint8)

def enhance_image(img):
    """Enhance image quality for better feature extraction"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge channels
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return img

def load_images_from_directory(directory):
    """Load all images from a directory with enhancement"""
    images = []
    image_paths = (glob.glob(os.path.join(directory, "*.JPG")) + 
                   glob.glob(os.path.join(directory, "*.jpg")) + 
                   glob.glob(os.path.join(directory, "*.PNG")) + 
                   glob.glob(os.path.join(directory, "*.png")) +
                   glob.glob(os.path.join(directory, "*.jpeg")) +
                   glob.glob(os.path.join(directory, "*.JPEG")))
    
    for img_path in sorted(image_paths):
        img = cv2.imread(img_path)
        if img is not None:
            # Enhance image
            img = enhance_image(img)
            # Resize
            img = cv2.resize(img, IMG_SIZE)
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return np.array(images)

def create_contrastive_model():
    """Create a feature extractor model using EfficientNet for better accuracy"""
    if HAS_EFFICIENTNET:
        # Use EfficientNetB0 for better feature extraction
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            pooling='avg'
        )
        print("   Using EfficientNetB0 as base model")
    else:
        # Fall back to MobileNetV2 if EfficientNet is not available
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            pooling='avg'
        )
        print("   Using MobileNetV2 as base model")
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom layers for feature extraction
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    
    # Add dense layers with batch normalization
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, activation='relu', name='fc2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Final feature vector
    features = layers.Dense(FEATURE_DIM, activation='linear', name='features')(x)
    
    # Normalize features for cosine similarity
    normalized_features = layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1), 
        output_shape=(FEATURE_DIM,),
        name='normalized_features'
    )(features)
    
    model = keras.Model(inputs, normalized_features, name='palm_feature_extractor')
    
    return model, base_model

def contrastive_loss(margin=1.0):
    """Contrastive loss function for training"""
    def loss(y_true, y_pred):
        # y_pred is the distance between features
        # y_true is 1 for same images, 0 for different images
        # Convert y_true to float32 to match y_pred
        y_true = tf.cast(y_true, tf.float32)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    return loss

def create_pairs(images, labels=None, num_pairs_per_image=3):
    """Create positive and negative pairs for contrastive learning"""
    pairs = []
    pair_labels = []
    
    n = len(images)
    
    # Create positive pairs (same image with augmentation)
    for i in range(n):
        for _ in range(num_pairs_per_image):
            # Original image
            img1 = images[i].copy()
            # Convert to uint8 for augmentation, then back to float
            img_uint8 = (img1 * 255).astype(np.uint8)
            img_aug = augment_image(img_uint8)
            img2 = img_aug.astype('float32') / 255.0
            pairs.append([img1, img2])
            pair_labels.append(1)  # Same image
    
    # Create negative pairs (different images)
    num_negative = len(pairs)
    for _ in range(num_negative):
        i, j = random.sample(range(n), 2)
        pairs.append([images[i], images[j]])
        pair_labels.append(0)  # Different images
    
    return np.array(pairs), np.array(pair_labels)

def create_triplets(images, num_triplets_per_image=2):
    """Create triplets for triplet loss training"""
    triplets = []
    n = len(images)
    
    for i in range(n):
        for _ in range(num_triplets_per_image):
            # Anchor
            anchor = images[i]
            # Positive (augmented version)
            positive = augment_image(images[i].copy())
            # Negative (different image)
            negative_idx = random.choice([j for j in range(n) if j != i])
            negative = images[negative_idx]
            
            triplets.append([anchor, positive, negative])
    
    return np.array(triplets)

def train_with_contrastive_learning():
    """Train feature extractor using contrastive learning"""
    print("=" * 60)
    print("Palm Recognition Model Training with Contrastive Learning")
    print("=" * 60)
    
    # Load images
    print("\n1. Loading training images...")
    train_images = load_images_from_directory(TRAIN_DIR)
    print(f"   Loaded {len(train_images)} training images")
    
    valid_images = []
    if os.path.exists(VALID_DIR):
        valid_images = load_images_from_directory(VALID_DIR)
        print(f"   Loaded {len(valid_images)} validation images")
    
    if len(train_images) == 0:
        raise ValueError(f"No images found in {TRAIN_DIR}")
    
    # Normalize images
    train_images = train_images.astype('float32') / 255.0
    if len(valid_images) > 0:
        valid_images = valid_images.astype('float32') / 255.0
    
    # Create model
    print("\n2. Creating feature extractor model...")
    feature_extractor, base_model = create_contrastive_model()
    
    # Create Siamese network for contrastive learning
    input_a = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name='input_a')
    input_b = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name='input_b')
    
    # Extract features
    features_a = feature_extractor(input_a)
    features_b = feature_extractor(input_b)
    
    # Compute distance (Euclidean distance)
    distance = layers.Lambda(
        lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True)),
        name='distance'
    )([features_a, features_b])
    
    siamese_model = keras.Model([input_a, input_b], distance, name='siamese_network')
    
    # Compile model
    siamese_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=contrastive_loss(margin=MARGIN),
        metrics=['accuracy']
    )
    
    print("\n3. Model Architecture:")
    feature_extractor.summary()
    
    # Prepare training data
    print("\n4. Preparing training pairs...")
    train_pairs, train_labels = create_pairs(train_images, num_pairs_per_image=2)
    print(f"   Created {len(train_pairs)} training pairs")
    
    valid_pairs, valid_labels = None, None
    if len(valid_images) > 0:
        valid_pairs, valid_labels = create_pairs(valid_images, num_pairs_per_image=1)
        print(f"   Created {len(valid_pairs)} validation pairs")
    
    # Callbacks - save feature extractor, not siamese model
    monitor_metric = 'val_loss' if valid_pairs is not None else 'loss'
    
    class SaveFeatureExtractorCallback(keras.callbacks.Callback):
        def __init__(self, feature_extractor, filepath, monitor='loss'):
            super().__init__()
            self.feature_extractor = feature_extractor
            self.filepath = filepath
            self.monitor = monitor
            self.best_score = float('inf')
        
        def on_epoch_end(self, epoch, logs=None):
            current_score = logs.get(self.monitor)
            if current_score is None:
                return
            if current_score < self.best_score:
                self.best_score = current_score
                self.feature_extractor.save(self.filepath)
                print(f"\n   Saved best feature extractor (score: {current_score:.4f})")
    
    save_feature_callback = SaveFeatureExtractorCallback(
        feature_extractor, MODEL_SAVE_PATH, monitor=monitor_metric
    )
    
    callbacks = [
        save_feature_callback,
        EarlyStopping(
            monitor=monitor_metric,
            patience=15,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\n5. Training model...")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Margin: {MARGIN}")
    
    history = siamese_model.fit(
        [train_pairs[:, 0], train_pairs[:, 1]],
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(
            [valid_pairs[:, 0], valid_pairs[:, 1]],
            valid_labels
        ) if valid_pairs is not None else None,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning: Unfreeze some layers
    print("\n6. Fine-tuning model...")
    base_model.trainable = True
    # Freeze early layers, unfreeze later layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    siamese_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.1),
        loss=contrastive_loss(margin=MARGIN),
        metrics=['accuracy']
    )
    
    print("   Fine-tuning with lower learning rate...")
    
    # Create a new callback for fine-tuning
    save_feature_callback_fine = SaveFeatureExtractorCallback(
        feature_extractor, MODEL_SAVE_PATH, monitor=monitor_metric
    )
    
    history_fine = siamese_model.fit(
        [train_pairs[:, 0], train_pairs[:, 1]],
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=20,
        validation_data=(
            [valid_pairs[:, 0], valid_pairs[:, 1]],
            valid_labels
        ) if valid_pairs is not None else None,
        callbacks=[
            save_feature_callback_fine,
            EarlyStopping(
                monitor=monitor_metric,
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ],
        verbose=1
    )
    
    # Save the feature extractor (final version)
    print(f"\n7. Saving feature extractor to {MODEL_SAVE_PATH}...")
    feature_extractor.save(MODEL_SAVE_PATH)
    
    # Evaluate model
    print("\n8. Evaluating model...")
    evaluate_model(feature_extractor, train_images, valid_images if len(valid_images) > 0 else None)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Feature extractor saved to: {MODEL_SAVE_PATH}")
    print("\nThe model is ready to use for palm recognition in app.py")
    
    return feature_extractor

def evaluate_model(model, train_images, valid_images=None):
    """Evaluate the trained model"""
    print("\nEvaluating model performance...")
    
    # Extract features for all training images
    print("Extracting features from training images...")
    train_features = []
    batch_size = 32
    for i in range(0, len(train_images), batch_size):
        batch = train_images[i:i+batch_size]
        features = model.predict(batch, verbose=0)
        train_features.extend(features)
    train_features = np.array(train_features)
    
    # Calculate intra-class similarity (same images with augmentation)
    print("Calculating similarity metrics...")
    similarities_same = []
    for i in range(min(10, len(train_images))):
        img = train_images[i]
        # Get original features
        feat1 = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        # Get augmented features
        img_uint8 = (img * 255).astype(np.uint8)
        aug_img = augment_image(img_uint8)
        aug_img = aug_img.astype('float32') / 255.0
        feat2 = model.predict(np.expand_dims(aug_img, axis=0), verbose=0)[0]
        # Calculate similarity (cosine similarity for normalized features)
        sim = cosine_similarity([feat1], [feat2])[0][0]
        similarities_same.append(sim)
    
    # Calculate inter-class similarity (different images)
    similarities_diff = []
    num_comparisons = min(50, len(train_images) * (len(train_images) - 1) // 2)
    for _ in range(num_comparisons):
        i, j = random.sample(range(len(train_images)), 2)
        sim = cosine_similarity([train_features[i]], [train_features[j]])[0][0]
        similarities_diff.append(sim)
    
    avg_same = np.mean(similarities_same)
    avg_diff = np.mean(similarities_diff)
    std_same = np.std(similarities_same)
    std_diff = np.std(similarities_diff)
    min_same = np.min(similarities_same)
    max_diff = np.max(similarities_diff)
    
    print(f"\nSimilarity Statistics:")
    print(f"  Same palm (with augmentation):")
    print(f"    Mean: {avg_same:.4f} ± {std_same:.4f}")
    print(f"    Min: {min_same:.4f}")
    print(f"  Different palms:")
    print(f"    Mean: {avg_diff:.4f} ± {std_diff:.4f}")
    print(f"    Max: {max_diff:.4f}")
    print(f"  Separation: {avg_same - avg_diff:.4f}")
    
    # Recommended threshold - use the point where separation is maximized
    # Threshold should be between max_diff and min_same
    if min_same > max_diff:
        threshold = (min_same + max_diff) / 2
        threshold = max(0.70, min(0.95, threshold))
        print(f"\n✓ Good separation! Model should perform well.")
    else:
        threshold = (avg_same + avg_diff) / 2
        threshold = max(0.70, min(0.95, threshold))
        print(f"\n⚠ Some overlap detected. Consider more training or data augmentation.")
    
    print(f"\nRecommended threshold: {threshold:.3f}")
    print(f"  (Current threshold in palm_recognition.py: 0.75)")
    print(f"  (Adjust in palm_recognition.py if needed)")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    
    try:
        model = train_with_contrastive_learning()
        print("\nNext steps:")
        print("1. Test the model accuracy")
        print("2. Adjust threshold in palm_recognition.py if needed (default: 0.75)")
        print("3. Run the Flask app: python app.py")
        print("4. Register users with their palm images")
        print("5. Perform transactions with palm verification")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
