#!/usr/bin/env python3
"""
Quick test of the advanced training model
"""

from advanced_train_model import load_images_with_labels, create_training_data, create_arcface_model, arcface_loss
import numpy as np
import tensorflow as tf

print('Loading data...')
train_images, train_labels = load_images_with_labels('Files/Train')
print(f'Loaded {len(train_images)} images')

# Normalize
train_images = train_images.astype('float32') / 255.0

# Create augmented data
print('Creating augmented data...')
train_aug_images, train_aug_labels = create_training_data(train_images, train_labels, augment_factor=2)
print(f'Augmented to {len(train_aug_images)} samples')

# Create model
print('Creating model...')
model, base = create_arcface_model()

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'arc_face_layer': arcface_loss(20, margin=0.5, scale=64.0),
        'normalized_features': None
    }
)

print('Testing one training step...')
try:
    # Test with one batch
    batch_size = 4
    batch_images = train_aug_images[:batch_size]
    batch_labels = train_aug_labels[:batch_size]

    result = model.train_on_batch([batch_images, batch_labels],
                                 {'arc_face_layer': batch_labels, 'normalized_features': np.zeros((batch_size, 512))})
    print('✓ Training step successful!')
    print('Loss:', result)
except Exception as e:
    print('✗ Training failed:', e)
    import traceback
    traceback.print_exc()