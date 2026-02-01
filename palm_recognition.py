"""
Palm Recognition Module
Handles feature extraction and comparison for palm authentication
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as _layers
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class PalmRecognizer:
    """Palm recognition using deep learning feature extraction"""
    
    def __init__(self, model_path="palm_feature_extractor_advanced.h5", threshold=0.75):
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
        print("Loading advanced trained palm recognition model...")
        try:
            # Custom L2Normalize layer to handle Lambda layer with tf.nn.l2_normalize
            class L2Normalize(_layers.Layer):
                def __init__(self, axis=1, **kwargs):
                    super(L2Normalize, self).__init__(**kwargs)
                    self.axis = axis

                def call(self, inputs):
                    return tf.nn.l2_normalize(inputs, axis=self.axis)

                def get_config(self):
                    config = super(L2Normalize, self).get_config()
                    config.update({"axis": self.axis})
                    return config

            # Provide compatibility for models saved from different TF/Keras versions
            # Custom DepthwiseConv2D wrapper to accept 'groups' argument if present
            class DepthwiseConv2DFixed(_layers.DepthwiseConv2D):
                def __init__(self, *args, groups=None, **kwargs):
                    if 'groups' in kwargs:
                        kwargs.pop('groups')
                    super().__init__(*args, **kwargs)

                @classmethod
                def from_config(cls, config):
                    # Remove unsupported 'groups' from config if present
                    config.pop('groups', None)
                    return super().from_config(config)

            custom_objects = {'DepthwiseConv2D': DepthwiseConv2DFixed, 'DepthwiseConv2DFixed': DepthwiseConv2DFixed, 'L2Normalize': L2Normalize}
            self.model = keras.models.load_model(self.model_path, compile=False, custom_objects=custom_objects)
            
            # Check if the model has Lambda layers and rebuild without them
            has_lambda = any(isinstance(layer, keras.layers.Lambda) for layer in self.model.layers)
            if has_lambda:
                print("  Model contains Lambda layers, rebuilding without them...")
                # Find the last non-Lambda layer
                last_layer = None
                for layer in reversed(self.model.layers):
                    if not isinstance(layer, keras.layers.Lambda):
                        last_layer = layer
                        break
                if last_layer is not None:
                    # Create new model with output from the last non-Lambda layer
                    new_model = keras.Model(inputs=self.model.input, outputs=last_layer.output)
                    self.model = new_model
                    print("  ✓ Rebuilt model without Lambda layers")
            
            print(f"✓ Loaded trained model from {self.model_path}")
            print("  Features: 512D ArcFace embeddings")
            print(f"  Threshold: {self.threshold}")
        except Exception as e:
            print(f"⚠️ Error loading trained model: {e}")
            print("  Using default model...")
            self.model = self._create_default_model()
    
    def _create_default_model(self):
        """Create a default feature extractor if trained model is not available"""
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras import layers
        
        # Try EfficientNetB3 first, fall back to B0, then MobileNetV2
        try:
            from tensorflow.keras.applications import EfficientNetB3
            base_model = EfficientNetB3(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size[0], self.img_size[1], 3),
                pooling='avg'
            )
            print("   Using EfficientNetB3 as base model")
        except (ImportError, Exception):
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
        x = layers.Dense(512, activation='relu', name='fc1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu', name='fc2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        features = layers.Dense(512, activation='linear', name='features')(x)  # Increased to 512 for better accuracy
        
        model = keras.Model(inputs, features, name='palm_feature_extractor')
        return model
    
    def enhance_image(self, img, fast_mode=False):
        """Enhance image quality for better feature extraction"""
        if fast_mode:
            # Skip enhancement for speed in registration
            return img
            
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
    
    def preprocess_image(self, image, fast_mode=False):
        """
        Preprocess image for feature extraction with enhancement
        
        Args:
            image: Input image (numpy array, bytes, or file path)
            fast_mode: If True, skip image enhancement for speed
        
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
        
        # Enhance image quality (skip in fast mode for registration)
        img = self.enhance_image(img, fast_mode=fast_mode)
        
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
    
    def extract_features(self, image, fast_mode=True):
        """
        Extract features from a palm image
        
        Args:
            image: Input image (numpy array, bytes, or file path)
            fast_mode: If True, skip image enhancement for speed
        
        Returns:
            Feature vector (numpy array of shape (512,))
        """
        try:
            processed_img = self.preprocess_image(image, fast_mode=fast_mode)
            features = self.model.predict(processed_img, verbose=0)
            # Remove batch dimension
            features = features[0]
            # L2 normalize the features
            features = features / np.linalg.norm(features)
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            raise
    
    def extract_features_batch(self, images, fast_mode=True):
        """
        Extract features from multiple palm images in batch for faster processing
        
        Args:
            images: List of input images (numpy arrays, bytes, or file paths)
            fast_mode: If True, skip image enhancement for speed
        
        Returns:
            List of feature vectors (each numpy array of shape (512,))
        """
        try:
            processed_images = []
            for image in images:
                processed_img = self.preprocess_image(image, fast_mode=fast_mode)
                processed_images.append(processed_img[0])  # Remove batch dim added by preprocess
            
            # Stack into batch
            batch_images = np.stack(processed_images, axis=0)
            
            # Predict in batch
            features_batch = self.model.predict(batch_images, verbose=0)
            
            # L2 normalize each feature vector
            features_list = []
            for features in features_batch:
                normalized_features = features / np.linalg.norm(features)
                features_list.append(normalized_features)
            
            return features_list
        except Exception as e:
            print(f"Error extracting features in batch: {e}")
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

