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
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class PalmRecognizer:
    """Palm recognition using deep learning feature extraction"""
    
    def __init__(self, model_path="palm_feature_extractor_professional_features.h5", threshold=0.65):
        """
        Initialize the professional palm recognizer with ArcFace and attention mechanisms
        
        Args:
            model_path: Path to the trained professional feature extractor model
            threshold: Similarity threshold for authentication (0.65 for professional accuracy)
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self.img_size = (224, 224)
        self.load_model()
    
    def load_model(self):
        """Load the professional trained model or fall back to advanced/created model"""
        print("Loading professional palm recognition model...")
        
        # Try professional model first
        professional_path = "palm_feature_extractor_professional_features.h5"
        if os.path.exists(professional_path):
            self.model_path = professional_path
        else:
            # Fall back to advanced model
            advanced_path = "palm_feature_extractor_advanced.h5"
            if os.path.exists(advanced_path):
                self.model_path = advanced_path
        
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
            
            model_type = "Professional" if "professional" in self.model_path else "Advanced"
            print(f"✓ Loaded {model_type} trained model from {self.model_path}")
            print("  Features: 512D ArcFace embeddings with attention")
            print(f"  Threshold: {self.threshold}")
        except Exception as e:
            print(f"⚠️ Error loading trained model: {e}")
            print("  Creating professional model...")
            self.model = self._create_advanced_model()
    
    def _create_advanced_model(self):
        """Create a professional-grade palm recognition model with ArcFace and attention mechanisms"""
        from tensorflow.keras.applications import EfficientNetB3
        from tensorflow.keras import layers, regularizers
        
        print("Creating advanced palm recognition model with ArcFace and attention...")
        
        # Use EfficientNetB3 as backbone for better feature extraction
        try:
            base_model = EfficientNetB3(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size[0], self.img_size[1], 3)
            )
            print("   Using EfficientNetB3 backbone")
        except Exception:
            # Fallback to EfficientNetB0
            from tensorflow.keras.applications import EfficientNetB0
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size[0], self.img_size[1], 3)
            )
            print("   Using EfficientNetB0 backbone")
        
        # Freeze the backbone initially
        base_model.trainable = False
        
        inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 3))
        
        # Multi-scale feature extraction
        # Get features at different scales
        base_features = base_model(inputs, training=False)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(base_features)
        
        # Attention mechanism - Squeeze and Excitation block
        def squeeze_excite_block(input_tensor, ratio=16):
            """Squeeze and Excitation attention block"""
            channels = input_tensor.shape[-1]
            se = layers.GlobalAveragePooling2D()(input_tensor)
            se = layers.Dense(channels // ratio, activation='relu')(se)
            se = layers.Dense(channels, activation='sigmoid')(se)
            se = layers.Reshape((1, 1, channels))(se)
            return layers.Multiply()([input_tensor, se])
        
        # Apply attention to base features before pooling
        base_features = layers.Reshape((1, 1, base_features.shape[-1]))(base_features)
        attended_features = squeeze_excite_block(base_features)
        attended_features = layers.Flatten()(attended_features)
        
        # Combine global and attended features
        x = layers.Concatenate()([x, attended_features])
        
        # Advanced dense layers with better regularization
        x = layers.Dense(1024, activation='relu', 
                        kernel_regularizer=regularizers.l2(1e-4),
                        name='fc1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Residual connection
        residual = x
        
        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(1e-4),
                        name='fc2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # Add residual connection
        if residual.shape[-1] == 512:
            x = layers.Add()([x, residual])
        
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(1e-4),
                        name='fc3')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # ArcFace-inspired final layer
        features = layers.Dense(512, activation=None, 
                              kernel_regularizer=regularizers.l2(1e-4),
                              name='features')(x)
        
        # L2 normalization for ArcFace
        features = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1), name='l2_normalize')(features)
        
        model = keras.Model(inputs, features, name='advanced_palm_recognizer')
        
        print("✓ Created advanced model with:")
        print("  - EfficientNetB3 backbone")
        print("  - Squeeze-and-Excitation attention")
        print("  - Multi-scale features")
        print("  - ArcFace-style L2 normalization")
        print("  - Advanced regularization")
        
        return model
    
    def enhance_image(self, img, fast_mode=False):
        """Enhance image quality for better feature extraction with advanced techniques"""
        if fast_mode:
            # Minimal enhancement for speed
            return img
            
        # Step 1: Denoising
        img = cv2.bilateralFilter(img, 9, 75, 75)  # Bilateral filter for noise reduction while preserving edges
        
        # Step 2: Convert to LAB color space for better enhancement
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) with optimized parameters
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Enhance color channels slightly
            a = cv2.addWeighted(a, 1.1, cv2.GaussianBlur(a, (0, 0), 1), -0.1, 0)
            b = cv2.addWeighted(b, 1.1, cv2.GaussianBlur(b, (0, 0), 1), -0.1, 0)
            
            # Merge channels
            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Step 3: Sharpening for better edge detection
        kernel = np.array([[-1,-1,-1], 
                          [-1, 9,-1],
                          [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        
        # Step 4: Adaptive gamma correction for better contrast
        quality = self.assess_image_quality(img)
        if quality['brightness'] < 100:
            gamma = 0.8  # Brighten dark images
        elif quality['brightness'] > 150:
            gamma = 1.4  # Darken bright images
        else:
            gamma = 1.0
        
        if gamma != 1.0:
            lookUpTable = np.empty((1,256), np.uint8)
            for i in range(256):
                lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            img = cv2.LUT(img, lookUpTable)
        
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
        
    def assess_image_quality(self, img):
        """Assess image quality metrics for adaptive processing"""
        if len(img.shape) == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Brightness (mean intensity)
        brightness = np.mean(gray)
        
        # Contrast (standard deviation)
        contrast = np.std(gray)
        
        # Sharpness (variance of Laplacian)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness
        }
    
    def adaptive_enhance_image(self, img):
        """Adaptive image enhancement based on quality assessment"""
        quality = self.assess_image_quality(img)
        
        # Adaptive gamma correction based on brightness
        if quality['brightness'] < 100:
            gamma = 0.8  # Brighten dark images
        elif quality['brightness'] > 150:
            gamma = 1.4  # Darken bright images
        else:
            gamma = 1.0
        
        if gamma != 1.0:
            lookUpTable = np.empty((1,256), np.uint8)
            for i in range(256):
                lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            img = cv2.LUT(img, lookUpTable)
        
        # Adaptive CLAHE based on contrast
        if quality['contrast'] < 30:
            clip_limit = 4.0  # More aggressive for low contrast
        else:
            clip_limit = 2.0
        
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Adaptive sharpening based on sharpness
        if quality['sharpness'] < 100:
            # More sharpening for blurry images
            kernel = np.array([[-1,-1,-1,-1,-1],
                              [-1, 1, 2, 1,-1],
                              [-1, 2, 4, 2,-1],
                              [-1, 1, 2, 1,-1],
                              [-1,-1,-1,-1,-1]]) / 8.0
            img = cv2.filter2D(img, -1, kernel)
        
        return img
        """
        Extract features from a palm image with enhanced preprocessing
        
        Args:
            image: Input image (numpy array, bytes, or file path)
            fast_mode: If True, use minimal enhancement for speed (default False for accuracy)
        
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
    
    def extract_features_batch(self, images, fast_mode=False):
        """
        Extract features from multiple palm images in batch for faster processing
        
        Args:
            images: List of input images (numpy arrays, bytes, or file paths)
            fast_mode: If True, use minimal enhancement for speed (default False for accuracy)
        
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

def get_palm_recognizer(threshold=0.65):
    """Get or create the global palm recognizer instance with enhanced accuracy"""
    global _palm_recognizer
    if _palm_recognizer is None:
        _palm_recognizer = PalmRecognizer(threshold=threshold)
    return _palm_recognizer

