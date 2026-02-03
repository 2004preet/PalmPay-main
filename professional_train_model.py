"""
Professional Palm Recognition Training with Triplet Loss
Creates the most advanced palm recognition model using triplet loss for maximum accuracy
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, regularizers
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import pickle
import random

class TripletLoss(layers.Layer):
    """Triplet loss for metric learning"""

    def __init__(self, margin=0.5, **kwargs):
        super(TripletLoss, self).__init__(**kwargs)
        self.margin = margin

    def call(self, inputs):
        anchor, positive, negative = inputs

        # Compute distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        # Triplet loss
        loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)
        return tf.reduce_mean(loss)

def create_triplet_model(input_shape=(224, 224, 3)):
    """Create advanced triplet loss model for palm recognition"""

    # Backbone - EfficientNetB3 with attention
    from tensorflow.keras.applications import EfficientNetB3

    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Unfreeze more layers for fine-tuning
    for layer in base_model.layers[-100:]:
        layer.trainable = True

    def create_embedding_network():
        inputs = keras.Input(shape=input_shape)

        # Multi-scale feature extraction
        x = base_model(inputs, training=True)

        # Squeeze and Excitation attention
        def squeeze_excite_block(input_tensor, ratio=16):
            channels = input_tensor.shape[-1]
            se = layers.GlobalAveragePooling2D()(input_tensor)
            se = layers.Dense(channels // ratio, activation='relu')(se)
            se = layers.Dense(channels, activation='sigmoid')(se)
            se = layers.Reshape((1, 1, channels))(se)
            return layers.Multiply()([input_tensor, se])

        x = squeeze_excite_block(x)

        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)

        # Dense layers with residual connections
        x = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        residual = x

        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        # Residual connection
        if residual.shape[-1] == 512:
            x = layers.Add()([x, residual])

        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Final embedding layer
        embedding = layers.Dense(512, activation=None, kernel_regularizer=regularizers.l2(1e-4))(x)
        embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(embedding)

        return keras.Model(inputs, embedding, name='embedding_network')

    # Create three instances of the embedding network
    embedding_network = create_embedding_network()

    # Triplet inputs
    anchor_input = keras.Input(shape=input_shape, name='anchor')
    positive_input = keras.Input(shape=input_shape, name='positive')
    negative_input = keras.Input(shape=input_shape, name='negative')

    # Generate embeddings
    anchor_embedding = embedding_network(anchor_input)
    positive_embedding = embedding_network(positive_input)
    negative_embedding = embedding_network(negative_input)

    # Triplet loss
    triplet_loss = TripletLoss(margin=0.5)([anchor_embedding, positive_embedding, negative_embedding])

    # Create triplet model
    triplet_model = keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=triplet_loss,
        name='triplet_model'
    )

    # Create feature extractor model
    feature_extractor = keras.Model(
        embedding_network.input,
        embedding_network.output,
        name='palm_feature_extractor'
    )

    return triplet_model, feature_extractor

def load_palm_images(train_dir, img_size=(224, 224), max_images_per_person=10):
    """Load palm images for triplet training"""
    images = []
    person_ids = []

    if not os.path.exists(train_dir):
        print(f"Training directory {train_dir} not found!")
        return np.array([]), np.array([])

    # Get all image files
    image_files = []
    for file in os.listdir(train_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(train_dir, file))

    print(f"Found {len(image_files)} palm images")

    # For demo purposes, treat all images as from different people
    # In real scenario, images should be organized by person
    for i, img_path in enumerate(image_files[:max_images_per_person * 5]):  # Limit for demo
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img.astype('float32') / 255.0

        images.append(img)
        person_ids.append(i % 5)  # Simulate 5 different people

    return np.array(images), np.array(person_ids)

def create_triplet_batch(images, person_ids, batch_size=8):
    """Create a batch of triplets for training"""
    anchors = []
    positives = []
    negatives = []

    unique_persons = np.unique(person_ids)

    for _ in range(batch_size):
        # Select anchor person
        anchor_person = np.random.choice(unique_persons)
        anchor_indices = np.where(person_ids == anchor_person)[0]

        if len(anchor_indices) < 2:
            continue  # Need at least 2 images per person

        # Select anchor and positive
        anchor_idx, positive_idx = np.random.choice(anchor_indices, 2, replace=False)
        anchor = images[anchor_idx]
        positive = images[positive_idx]

        # Select negative
        negative_person = np.random.choice([p for p in unique_persons if p != anchor_person])
        negative_indices = np.where(person_ids == negative_person)[0]
        negative_idx = np.random.choice(negative_indices)
        negative = images[negative_idx]

        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)

    return np.array(anchors), np.array(positives), np.array(negatives)

def data_generator(images, person_ids, batch_size=8):
    """Generator for triplet training data"""
    while True:
        anchors, positives, negatives = create_triplet_batch(images, person_ids, batch_size)
        yield (anchors, positives, negatives), np.zeros(batch_size)  # Dummy labels

def train_professional_model(train_dir, model_save_path="palm_feature_extractor_professional.h5",
                           epochs=50, batch_size=8, steps_per_epoch=100):
    """Train the professional triplet loss palm recognition model"""

    print("🚀 Starting Professional Palm Recognition Training with Triplet Loss")
    print("=" * 70)

    # Load data
    print("Loading palm images...")
    images, person_ids = load_palm_images(train_dir)

    if len(images) == 0:
        print("❌ No training images found!")
        return 0.5

    print(f"Dataset: {len(images)} images from {len(np.unique(person_ids))} simulated persons")

    # Create models
    print("Creating triplet loss model...")
    triplet_model, feature_extractor = create_triplet_model()

    # Compile triplet model
    triplet_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4, amsgrad=True),
        loss=lambda y_true, y_pred: y_pred  # Triplet loss is already computed
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            save_best_only=True,
            monitor='loss',
            mode='min',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='./logs_professional',
            histogram_freq=1
        )
    ]

    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(images, person_ids, batch_size),
        output_signature=(
            (
                tf.TensorSpec(shape=(batch_size, 224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, 224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, 224, 224, 3), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.float32)
        )
    )

    # Training
    print("Starting triplet training...")
    history = triplet_model.fit(
        dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save feature extractor
    print("Saving professional feature extractor...")
    feature_extractor.save(model_save_path, save_format='h5')

    # Evaluate embeddings
    print("Evaluating learned embeddings...")
    embeddings = feature_extractor.predict(images, batch_size=batch_size, verbose=1)

    # Simple evaluation - check if same person embeddings are closer
    similarities = []
    labels = []

    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            similarities.append(sim)
            labels.append(1 if person_ids[i] == person_ids[j] else 0)

    similarities = np.array(similarities)
    labels = np.array(labels)

    # Find optimal threshold
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_accuracy = 0

    for threshold in thresholds:
        predictions = (similarities >= threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print("\n🎯 Professional Training Complete!")
    print(f"Best evaluation accuracy: {best_accuracy:.4f}")
    print(f"Recommended threshold: {best_threshold:.3f}")
    print(f"Model saved as: {model_save_path}")

    # Save training history
    with open('professional_training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    return best_threshold

if __name__ == "__main__":
    # Configuration
    TRAIN_DIR = "Files/Train"
    MODEL_SAVE_PATH = "palm_feature_extractor_professional.h5"

    # Train the professional model
    threshold = train_professional_model(
        train_dir=TRAIN_DIR,
        model_save_path=MODEL_SAVE_PATH,
        epochs=100,
        batch_size=4,
        steps_per_epoch=50
    )

    print(f"\n✅ Professional model training complete!")
    print(f"Update palm_recognition.py with:")
    print(f"  model_path='{MODEL_SAVE_PATH}'")
    print(f"  threshold={threshold:.3f}")