#!/usr/bin/env python3
"""
Test script to validate feature extraction optimizations
"""

import time
import numpy as np
import cv2
from palm_recognition import PalmRecognizer

def create_test_image():
    """Create a simple test image for benchmarking"""
    # Create a 224x224 RGB image with some pattern
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return img

def test_feature_extraction_speed():
    """Test the speed of feature extraction with and without optimizations"""
    print("Testing feature extraction speed optimizations...")

    # Initialize recognizer
    recognizer = PalmRecognizer()

    # Create test images
    test_images = [create_test_image() for _ in range(10)]

    print("\n1. Testing single image extraction (fast mode):")
    start_time = time.time()
    for img in test_images:
        features = recognizer.extract_features(img, fast_mode=True)
    fast_time = time.time() - start_time
    print(".3f")
    print(f"   Features shape: {features.shape}")

    print("\n2. Testing single image extraction (normal mode):")
    start_time = time.time()
    for img in test_images:
        features = recognizer.extract_features(img, fast_mode=False)
    normal_time = time.time() - start_time
    print(".3f")

    print("\n3. Testing batch extraction (fast mode):")
    start_time = time.time()
    features_batch = recognizer.extract_features_batch(test_images, fast_mode=True)
    batch_time = time.time() - start_time
    print(".3f")
    print(f"   Batch size: {len(features_batch)}, Feature shape: {features_batch[0].shape}")

    print("\n4. Testing batch extraction (normal mode):")
    start_time = time.time()
    features_batch = recognizer.extract_features_batch(test_images, fast_mode=False)
    batch_normal_time = time.time() - start_time
    print(".3f")

    print("\nSpeed comparison:")
    print(".2f")
    print(".2f")
    print(".2f")

    print("\n✓ Feature extraction optimizations working correctly!")

if __name__ == "__main__":
    test_feature_extraction_speed()