"""
Palm Recognition Module
Handles feature extraction and comparison for palm authentication
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class PalmRecognizer:
    """Palm recognition using deep learning feature extraction"""
    
    def __init__(self, model_path="palm_feature_extractor.h5", threshold=0.75):
        """
        Initialize the palm recognizer
        
        Args:
            model_path: Path to the trained feature extractor model
            threshold: Similarity threshold for authentication (0-1)
        """
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self.img_size = (224, 224)
        self.load_model()
    
    def load_model(self):
        """Load the trained feature extractor model"""
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path, compile=False)
                print(f"Loaded palm recognition model from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating a new model...")
                self.model = self._create_default_model()
        else:
            print(f"Model not found at {self.model_path}")
            print("Creating a default model (accuracy may be lower)...")
            self.model = self._create_default_model()
    
    def _create_default_model(self):
        """Create a default feature extractor if trained model is not available"""
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras import layers
        
        # Try EfficientNetB0 first, fall back to MobileNetV2
        try:
            from tensorflow.keras.applications import EfficientNetB0
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size[0], self.img_size[1], 3),
                pooling='avg'
            )
            print("   Using EfficientNetB0 as base model")
        except (ImportError, Exception):
            # Fall back to MobileNetV2 if EfficientNet is not available
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size[0], self.img_size[1], 3),
                pooling='avg'
            )
            print("   Using MobileNetV2 as base model")
        
        base_model.trainable = False
        
        inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 3))
        x = base_model(inputs, training=False)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        features = layers.Dense(128, activation='linear')(x)
        normalized_features = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(features)
        
        model = keras.Model(inputs, normalized_features, name='palm_feature_extractor')
        return model
    
    def enhance_image(self, img):
        """Enhance image quality for better feature extraction"""
        # Convert to LAB color space for better enhancement
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img
    
    def preprocess_image(self, image):
        """
        Preprocess image for feature extraction with enhancement
        
        Args:
            image: Input image (numpy array, bytes, or file path)
        
        Returns:
            Preprocessed image array
        """
        # Handle different input types
        if isinstance(image, bytes):
            # Convert bytes to numpy array
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image, str):
            # Load from file path
            img = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            img = image.copy()
            # If image is RGB, convert to BGR for OpenCV processing
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Check if it's already BGR or RGB
                # If values are in [0, 1] range, assume RGB from model
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        if img is None:
            raise ValueError("Could not decode image")
        
        # Enhance image quality
        img = self.enhance_image(img)
        
        # Convert BGR to RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img = cv2.resize(img, self.img_size)
        
        # Normalize to [0, 1]
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def extract_features(self, image):
        """
        Extract features from a palm image
        
        Args:
            image: Input image (numpy array, bytes, or file path)
        
        Returns:
            Feature vector (numpy array of shape (128,))
        """
        try:
            processed_img = self.preprocess_image(image)
            features = self.model.predict(processed_img, verbose=0)
            # Remove batch dimension
            features = features[0]
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            raise
    
    def compare_features(self, features1, features2):
        """
        Compare two feature vectors using cosine similarity
        
        Args:
            features1: First feature vector
            features2: Second feature vector
        
        Returns:
            Similarity score (0-1)
        """
        # Ensure features are 2D arrays
        if len(features1.shape) == 1:
            features1 = features1.reshape(1, -1)
        if len(features2.shape) == 1:
            features2 = features2.reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(features1, features2)[0][0]
        return float(similarity)
    
    def verify_palm(self, stored_features, new_image):
        """
        Verify if a new palm image matches stored features
        
        Args:
            stored_features: Feature vector stored during registration (numpy array or bytes)
            new_image: New palm image to verify (numpy array, bytes, or file path)
        
        Returns:
            tuple: (is_verified: bool, similarity_score: float)
        """
        try:
            # Handle stored_features if it's bytes
            if isinstance(stored_features, bytes):
                stored_features = np.frombuffer(stored_features, dtype=np.float32)
            
            # Extract features from new image
            new_features = self.extract_features(new_image)
            
            # Compare features
            similarity = self.compare_features(stored_features, new_features)
            
            # Check if similarity exceeds threshold
            is_verified = similarity >= self.threshold
            
            return is_verified, similarity
        except Exception as e:
            print(f"Error during palm verification: {e}")
            return False, 0.0
    
    def save_features(self, features, file_path):
        """Save features to a file"""
        if isinstance(features, np.ndarray):
            features_bytes = features.tobytes()
            with open(file_path, 'wb') as f:
                f.write(features_bytes)
        else:
            with open(file_path, 'wb') as f:
                f.write(features)
    
    def load_features(self, file_path):
        """Load features from a file"""
        with open(file_path, 'rb') as f:
            features_bytes = f.read()
        features = np.frombuffer(features_bytes, dtype=np.float32)
        return features

# Global instance
_palm_recognizer = None

def get_palm_recognizer(threshold=0.75):
    """Get or create the global palm recognizer instance"""
    global _palm_recognizer
    if _palm_recognizer is None:
        _palm_recognizer = PalmRecognizer(threshold=threshold)
    return _palm_recognizer

