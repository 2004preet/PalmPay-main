#!/usr/bin/env python3
"""
Test the updated PalmRecognizer with advanced model
"""

from palm_recognition import PalmRecognizer
import numpy as np

print('Testing updated PalmRecognizer with advanced model...')
try:
    # Test with advanced model (should fallback gracefully)
    recognizer = PalmRecognizer()
    print('✓ PalmRecognizer initialized successfully!')

    # Test feature extraction
    dummy_image = np.random.random((224, 224, 3)).astype(np.float32)
    features = recognizer.extract_features(dummy_image)
    print('✓ Feature extraction successful!')
    print(f'  Features shape: {features.shape}')
    print(f'  Features range: [{features.min():.4f}, {features.max():.4f}]')

    # Test similarity comparison
    features1 = recognizer.extract_features(np.random.random((224, 224, 3)).astype(np.float32))
    features2 = recognizer.extract_features(np.random.random((224, 224, 3)).astype(np.float32))
    similarity = recognizer.compare_features(features1, features2)
    print('✓ Similarity comparison successful!')
    print(f'  Random similarity: {similarity:.4f}')

    print('🎉 Advanced Palm Recognition is ready!')

except Exception as e:
    print('✗ Error:', e)
    import traceback
    traceback.print_exc()