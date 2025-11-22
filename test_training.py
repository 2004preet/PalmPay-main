#!/usr/bin/env python3
"""
Quick test script to verify training setup
"""

import os
import sys

# Check if required directories exist
print("Checking training setup...")
if not os.path.exists("Files/Train"):
    print("ERROR: Files/Train directory not found!")
    sys.exit(1)

if not os.path.exists("Files/Valid"):
    print("WARNING: Files/Valid directory not found! Will use part of training data for validation.")

# Check for images
import glob
train_images = glob.glob("Files/Train/*.JPG") + glob.glob("Files/Train/*.jpg") + \
               glob.glob("Files/Train/*.PNG") + glob.glob("Files/Train/*.png")
valid_images = glob.glob("Files/Valid/*.JPG") + glob.glob("Files/Valid/*.jpg") + \
               glob.glob("Files/Valid/*.PNG") + glob.glob("Files/Valid/*.png")

print(f"Found {len(train_images)} training images")
print(f"Found {len(valid_images)} validation images")

if len(train_images) == 0:
    print("ERROR: No training images found!")
    sys.exit(1)

# Check if TensorFlow is available
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("ERROR: TensorFlow not installed!")
    print("Please install: pip install tensorflow>=2.10.0")
    sys.exit(1)

# Check if OpenCV is available
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError:
    print("ERROR: OpenCV not installed!")
    print("Please install: pip install opencv-python")
    sys.exit(1)

# Check if scikit-learn is available
try:
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("ERROR: scikit-learn not installed!")
    print("Please install: pip install scikit-learn")
    sys.exit(1)

print("\n✓ All dependencies are available!")
print("\nYou can now run: python train_model.py")
print("(This may take a while depending on your hardware)")

