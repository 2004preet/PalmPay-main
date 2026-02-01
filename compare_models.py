#!/usr/bin/env python3
"""
Model Comparison Script
Compare basic and advanced palm recognition models
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from palm_recognition import PalmRecognizer
import glob
from sklearn.metrics.pairwise import cosine_similarity

def load_test_images(directory):
    """Load test images for comparison"""
    images = []
    image_paths = (glob.glob(os.path.join(directory, "*.JPG")) +
                   glob.glob(os.path.join(directory, "*.jpg")) +
                   glob.glob(os.path.join(directory, "*.PNG")) +
                   glob.glob(os.path.join(directory, "*.png")))

    for img_path in sorted(image_paths)[:10]:  # Test with first 10 images
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

    return np.array(images)

def compare_models():
    """Compare basic and advanced models"""
    print("=" * 60)
    print("Palm Recognition Model Comparison")
    print("=" * 60)

    # Load test images
    print("\n1. Loading test images...")
    test_images = load_test_images("Files/Train")
    if len(test_images) == 0:
        print("No test images found!")
        return

    print(f"   Loaded {len(test_images)} test images")

    # Test basic model
    print("\n2. Testing basic model...")
    basic_recognizer = PalmRecognizer(model_path="palm_feature_extractor.h5", threshold=0.75)

    basic_features = []
    for i, img in enumerate(test_images):
        features = basic_recognizer.extract_features(img)
        basic_features.append(features)
        if i % 5 == 0:
            print(f"   Processed {i+1}/{len(test_images)} images")

    basic_features = np.array(basic_features)

    # Test advanced model if available
    advanced_features = None
    if os.path.exists("palm_feature_extractor_advanced.h5"):
        print("\n3. Testing advanced model...")
        advanced_recognizer = PalmRecognizer(model_path="palm_feature_extractor_advanced.h5", threshold=0.85)

        advanced_features = []
        for i, img in enumerate(test_images):
            features = advanced_recognizer.extract_features(img)
            advanced_features.append(features)
            if i % 5 == 0:
                print(f"   Processed {i+1}/{len(test_images)} images")

        advanced_features = np.array(advanced_features)
    else:
        print("\n3. Advanced model not found (palm_feature_extractor_advanced.h5)")

    # Compare feature dimensions
    print("\n4. Feature Analysis:")
    print(f"   Basic model features: {basic_features.shape[1]} dimensions")
    if advanced_features is not None:
        print(f"   Advanced model features: {advanced_features.shape[1]} dimensions")

    # Calculate similarities
    print("\n5. Similarity Analysis:")

    # Basic model similarities
    basic_similarities = []
    for i in range(len(basic_features)):
        for j in range(i+1, len(basic_features)):
            sim = cosine_similarity([basic_features[i]], [basic_features[j]])[0][0]
            basic_similarities.append(sim)

    basic_avg_sim = np.mean(basic_similarities)
    basic_std_sim = np.std(basic_similarities)

    print(f"   Basic Model - Average similarity: {basic_avg_sim:.4f} ± {basic_std_sim:.4f}")

    if advanced_features is not None:
        # Advanced model similarities
        advanced_similarities = []
        for i in range(len(advanced_features)):
            for j in range(i+1, len(advanced_features)):
                sim = cosine_similarity([advanced_features[i]], [advanced_features[j]])[0][0]
                advanced_similarities.append(sim)

        advanced_avg_sim = np.mean(advanced_similarities)
        advanced_std_sim = np.std(advanced_similarities)

        print(f"   Advanced Model - Average similarity: {advanced_avg_sim:.4f} ± {advanced_std_sim:.4f}")

        # Compare
        diff = advanced_avg_sim - basic_avg_sim
        if diff > 0:
            print(f"   ✓ Advanced model shows {diff:.4f} higher average similarity")
        else:
            print(f"   ⚠ Advanced model shows {abs(diff):.4f} lower average similarity")

    # Feature variance analysis
    print("\n6. Feature Variance Analysis:")
    basic_variance = np.mean(np.var(basic_features, axis=0))
    print(f"   Basic model feature variance: {basic_variance:.6f}")

    if advanced_features is not None:
        advanced_variance = np.mean(np.var(advanced_features, axis=0))
        print(f"   Advanced model feature variance: {advanced_variance:.6f}")

        var_ratio = advanced_variance / basic_variance
        print(f"   Variance ratio (Advanced/Basic): {var_ratio:.2f}")
        if var_ratio > 1:
            print("   ✓ Advanced model has higher feature variance (potentially better discrimination)")
        else:
            print("   ⚠ Advanced model has lower feature variance")

    print("\n" + "=" * 60)
    print("Comparison completed!")
    print("=" * 60)

    if advanced_features is not None:
        print("\nRecommendations:")
        print("• Use advanced model for better accuracy")
        print("• Consider threshold around 0.85 for advanced model")
        print("• Retrain advanced model with more data for best results")
    else:
        print("\nNext steps:")
        print("• Train advanced model: python advanced_train_model.py")
        print("• Then re-run this comparison")

if __name__ == "__main__":
    try:
        compare_models()
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()