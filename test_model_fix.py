#!/usr/bin/env python3
"""
Test the fixed advanced training model
"""

from advanced_train_model import create_arcface_model, arcface_loss
import tensorflow as tf
import numpy as np

print('Testing model compilation...')
try:
    model, base = create_arcface_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'arc_face_layer': arcface_loss(20, margin=0.5, scale=64.0),
            'normalized_features': None
        },
        metrics={
            'arc_face_layer': ['accuracy'],
            'normalized_features': []
        }
    )
    print('✓ Model compiled successfully!')

    # Test with dummy data
    dummy_images = np.random.random((4, 224, 224, 3)).astype(np.float32)
    dummy_labels = np.array([0, 1, 2, 0], dtype=np.int32)

    result = model.train_on_batch([dummy_images, dummy_labels],
                                 {'arc_face_layer': dummy_labels, 'normalized_features': np.zeros((4, 512))})
    print('✓ Training step successful!')
    print('Loss:', result)

except Exception as e:
    print('✗ Error:', e)
    import traceback
    traceback.print_exc()