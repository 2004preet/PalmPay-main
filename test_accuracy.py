"""
Test Palm Recognition Accuracy
Evaluates the trained model's performance on test data
"""

import os
import cv2
import numpy as np
from palm_recognition import PalmRecognizer
from sklearn.metrics.pairwise import cosine_similarity
import glob

def test_accuracy():
    """Test the accuracy of palm recognition"""
    print("Testing Palm Recognition Accuracy")
    print("=" * 50)

    # Initialize recognizer
    recognizer = PalmRecognizer(threshold=0.75)

    # Load test images (assuming same structure as train/valid)
    test_dir = "Files/Valid"  # Or create a separate test directory
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found. Using Train for testing.")
        test_dir = "Files/Train"

    image_paths = (glob.glob(os.path.join(test_dir, "*.JPG")) +
                   glob.glob(os.path.join(test_dir, "*.jpg")) +
                   glob.glob(os.path.join(test_dir, "*.PNG")) +
                   glob.glob(os.path.join(test_dir, "*.png")))

    if len(image_paths) == 0:
        print("No test images found!")
        return

    # Group by palm
    palm_groups = {}
    for img_path in sorted(image_paths):
        base_name = os.path.basename(img_path)
        try:
            number_part = ''.join(filter(str.isdigit, base_name))
            if number_part:
                number = int(number_part)
                palm_id = (number // 100) - 37
            else:
                palm_id = 0
        except:
            palm_id = 0

        if palm_id not in palm_groups:
            palm_groups[palm_id] = []
        palm_groups[palm_id].append(img_path)

    print(f"Found {len(palm_groups)} different palms with {len(image_paths)} total images")

    # Extract features for all images
    features_dict = {}
    for palm_id, paths in palm_groups.items():
        features_dict[palm_id] = []
        for path in paths:
            img = cv2.imread(path)
            if img is not None:
                features = recognizer.extract_features(img)
                features_dict[palm_id].append(features)

    # Test accuracy
    correct = 0
    total = 0

    for palm_id, features_list in features_dict.items():
        for i, features in enumerate(features_list):
            # Compare with all other palms
            best_match = None
            best_score = -1

            for other_palm_id, other_features_list in features_dict.items():
                for other_features in other_features_list:
                    if other_palm_id == palm_id and other_features is features:
                        continue  # Skip self
                    score = recognizer.compare_features(features, other_features)
                    if score > best_score:
                        best_score = score
                        best_match = other_palm_id

            if best_match == palm_id:
                correct += 1
            total += 1

    accuracy = correct / total * 100
    print(".2f")
    print(f"Correct identifications: {correct}/{total}")

    return accuracy

if __name__ == "__main__":
    test_accuracy()