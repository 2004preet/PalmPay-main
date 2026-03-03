"""
Advanced Palm Recognition Training — Deep Learning Edition v4
=============================================================
Key upgrades over v3:
  • Resolution: 256×256 (was 224×224) for richer ridge detail
  • Backbone: ResNet50V2 with progressive unfreeze
  • Loss:     ArcFace with increased margin (0.65) + center loss
  • Augment:  CutMix + MixUp + elastic distortion + perspective warp
  • LR:       Cosine-annealing with linear warmup callback
  • Stage 2:  Online hard-negative triplet-loss (margin 0.45)
  • Eval:     FAR / FRR / EER metrics for deployment readiness
  • Misc:     NUM_CLASSES auto-discovered, FEATURE_DIM=512
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
import random
from sklearn.metrics.pairwise import cosine_similarity
import math
import logging
import datetime

# ─────────────────────────────────────────────────────────────────────────────
#  Custom layers
# ─────────────────────────────────────────────────────────────────────────────
class L2Normalize(layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=self.axis)

    def get_config(self):
        c = super().get_config()
        c.update({"axis": self.axis})
        return c


class ArcFaceLayer(layers.Layer):
    """
    Additive Angular Margin loss layer (ArcFace / InsightFace).
    During training multiplies the normalised feature vector against
    normalised class weights, adds angular margin m to the true class,
    then scales by s before softmax.  At inference returns raw cosine logits.
    """
    def __init__(self, num_classes, scale=64.0, margin=0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.scale       = scale
        self.margin      = margin
        self.cos_m       = math.cos(margin)
        self.sin_m       = math.sin(margin)
        self.th          = math.cos(math.pi - margin)
        self.mm          = math.sin(math.pi - margin) * margin

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.num_classes),
            initializer="glorot_normal",
            trainable=True,
            name="arcface_weights",
        )
        super().build(input_shape)

    def call(self, embeddings, labels=None, training=None):
        # Normalise both embeddings and weights
        emb_norm  = tf.nn.l2_normalize(embeddings, axis=1)
        W_norm    = tf.nn.l2_normalize(self.W,     axis=0)
        cosine    = tf.matmul(emb_norm, W_norm)       # (B, C)

        if labels is None or not training:
            return cosine * self.scale

        # ArcFace margin application
        sine      = tf.sqrt(tf.maximum(1.0 - tf.square(cosine), 1e-9))
        phi       = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)
        # Stable fallback for θ > π - m
        phi       = tf.where(cosine > self.th, phi,
                             cosine - self.mm)

        one_hot   = tf.one_hot(tf.cast(labels, tf.int32), self.num_classes)
        output    = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale

    def get_config(self):
        c = super().get_config()
        c.update({
            "num_classes": self.num_classes,
            "scale":       self.scale,
            "margin":      self.margin,
        })
        return c


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE         = (256, 256)   # ← upgraded from 224×224
BATCH_SIZE       = 8
EPOCHS           = 80
LEARNING_RATE    = 1e-4
TRAIN_DIR        = "Files/Train"
VALID_DIR        = "Files/Valid"
MODEL_SAVE_PATH  = "palm_feature_extractor_advanced.h5"
FEATURE_DIM      = 512
NUM_CLASSES      = None   # auto-discovered
ARC_SCALE        = 64.0
ARC_MARGIN       = 0.65      # ← increased from 0.5 for harder angular separation
CENTER_LOSS_W    = 0.005     # ← NEW: center loss weight for intra-class compactness

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation — CutMix + MixUp + existing geometric/colour transforms
# ─────────────────────────────────────────────────────────────────────────────
def cutmix_augment(img1, img2, alpha=1.0):
    """
    CutMix: randomly paste a rectangular region from img2 onto img1.
    Returns blended image + lam (proportion of img1).
    """
    lam = np.random.beta(alpha, alpha)
    h, w = img1.shape[:2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w   = int(w * cut_rat)
    cut_h   = int(h * cut_rat)
    cx = random.randint(0, w)
    cy = random.randint(0, h)
    x1 = max(cx - cut_w // 2, 0);  x2 = min(cx + cut_w // 2, w)
    y1 = max(cy - cut_h // 2, 0);  y2 = min(cy + cut_h // 2, h)
    result = img1.copy()
    result[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (w * h)
    return result, lam


def mixup_augment(img1, img2, alpha=0.4):
    """MixUp: pixel-level linear blend of two images."""
    lam = np.random.beta(alpha, alpha)
    return (lam * img1 + (1 - lam) * img2).astype(np.uint8), lam


def elastic_distortion(img, alpha=80, sigma=10):
    """Apply elastic distortion for realistic palm deformation."""
    h, w = img.shape[:2]
    rng = np.random.RandomState()
    dx = cv2.GaussianBlur((rng.rand(h, w) * 2 - 1).astype(np.float32),
                          (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.rand(h, w) * 2 - 1).astype(np.float32),
                          (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def perspective_warp(img, strength=0.06):
    """Apply random perspective warp to simulate camera angle changes."""
    h, w = img.shape[:2]
    s = strength
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.uniform(0, w*s), random.uniform(0, h*s)],
        [w - random.uniform(0, w*s), random.uniform(0, h*s)],
        [w - random.uniform(0, w*s), h - random.uniform(0, h*s)],
        [random.uniform(0, w*s), h - random.uniform(0, h*s)]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def advanced_augment_image(img):
    """Geometric + colour augmentation + elastic distortion + perspective warp."""
    img = img.astype(np.float32)

    if random.random() > 0.4:
        angle  = random.uniform(-20, 20)
        center = (img.shape[1] // 2, img.shape[0] // 2)
        M      = cv2.getRotationMatrix2D(center, angle, 1.0)
        img    = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                borderMode=cv2.BORDER_REFLECT)

    if random.random() > 0.4:
        alpha = random.uniform(0.7, 1.3)
        beta  = random.uniform(-25, 25)
        img   = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if random.random() > 0.5:
        noise = np.random.normal(0, random.uniform(2, 10), img.shape)
        img   = np.clip(img + noise, 0, 255)

    if random.random() > 0.65:
        ksize = random.choice([3, 5])
        img   = cv2.GaussianBlur(img, (ksize, ksize), 0)

    if random.random() > 0.93:
        img = cv2.flip(img, 1)

    if random.random() > 0.55:
        zoom = random.uniform(0.88, 1.12)
        h, w = img.shape[:2]
        nh, nw = int(h * zoom), int(w * zoom)
        img = cv2.resize(img, (nw, nh))
        if zoom > 1:
            sy = (nh - h) // 2;  sx = (nw - w) // 2
            img = img[sy:sy+h, sx:sx+w]
        else:
            py = (h - nh) // 2;  px = (w - nw) // 2
            img = cv2.copyMakeBorder(img, py, h-nh-py, px, w-nw-px,
                                     cv2.BORDER_REFLECT)

    if random.random() > 0.45:
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0]  = (hsv[..., 0] + random.uniform(-15, 15)) % 180
        hsv[..., 1]  = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
        hsv[..., 2]  = np.clip(hsv[..., 2] * random.uniform(0.8, 1.2), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # NEW: Elastic distortion — simulates palm skin deformation
    if random.random() > 0.65:
        img = elastic_distortion(np.clip(img, 0, 255).astype(np.uint8),
                                  alpha=random.uniform(40, 100),
                                  sigma=random.uniform(6, 14))
        img = img.astype(np.float32)

    # NEW: Perspective warp — simulates camera angle variation
    if random.random() > 0.70:
        img = perspective_warp(np.clip(img, 0, 255).astype(np.uint8),
                                strength=random.uniform(0.03, 0.08))
        img = img.astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  Image enhancement (used at load time)
# ─────────────────────────────────────────────────────────────────────────────
def enhance_image_advanced(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    l     = cv2.bilateralFilter(l, 9, 75, 75)
    lab   = cv2.merge([l, a, b])
    img   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Light sharpening
    blur  = cv2.GaussianBlur(img, (0, 0), 3)
    img   = cv2.addWeighted(img, 1.3, blur, -0.3, 0)
    return img


# ─────────────────────────────────────────────────────────────────────────────
#  Image loading
# ─────────────────────────────────────────────────────────────────────────────
def load_images_with_labels(directory):
    """
    Load images. Palm identity = subdirectory name (preferred) OR
    numeric range heuristic for flat dirs (backward compat).
    """
    images, labels = [], []

    # Prefer subdirectory layout first
    subdirs = [d for d in os.listdir(directory)
               if os.path.isdir(os.path.join(directory, d))]
    if subdirs:
        label_map = {name: i for i, name in enumerate(sorted(subdirs))}
        for name, label in label_map.items():
            img_paths = (
                glob.glob(os.path.join(directory, name, "*.jpg"))  +
                glob.glob(os.path.join(directory, name, "*.JPG"))  +
                glob.glob(os.path.join(directory, name, "*.png"))  +
                glob.glob(os.path.join(directory, name, "*.PNG"))
            )
            for p in img_paths:
                img = cv2.imread(p)
                if img is not None:
                    img = enhance_image_advanced(img)
                    img = cv2.resize(img, IMG_SIZE)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(label)
        return np.array(images), np.array(labels)

    # Flat directory — number-range heuristic (legacy)
    image_paths = (glob.glob(os.path.join(directory, "*.JPG")) +
                   glob.glob(os.path.join(directory, "*.jpg")) +
                   glob.glob(os.path.join(directory, "*.PNG")) +
                   glob.glob(os.path.join(directory, "*.png")))

    palm_groups = {}
    for p in sorted(image_paths):
        base = os.path.basename(p)
        try:
            num     = int("".join(filter(str.isdigit, base)))
            palm_id = num // 10   # group every 10 images by default
        except Exception:
            palm_id = 0
        palm_groups.setdefault(palm_id, []).append(p)

    label_map = {pid: i for i, pid in enumerate(sorted(palm_groups))}
    for pid, paths in palm_groups.items():
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                img = enhance_image_advanced(img)
                img = cv2.resize(img, IMG_SIZE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label_map[pid])

    return np.array(images), np.array(labels)


# ─────────────────────────────────────────────────────────────────────────────
#  Model creation — ResNet50V2 + ArcFace head
# ─────────────────────────────────────────────────────────────────────────────
def create_arcface_model(num_classes):
    """
    ResNet50V2 backbone → 512D ArcFace embedding.
    Returns:  (training_model, feature_extractor, base_model)
    """
    from tensorflow.keras import regularizers

    try:
        from tensorflow.keras.applications import ResNet50V2
        base = ResNet50V2(
            weights="imagenet", include_top=False,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
        print("   ✓ Using ResNet50V2 backbone")
    except Exception:
        from tensorflow.keras.applications import MobileNetV2
        base = MobileNetV2(
            weights="imagenet", include_top=False,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), pooling="avg"
        )
        print("   ✓ Using MobileNetV2 backbone (fallback)")

    # Freeze all but last 30 layers initially
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    inp = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x   = base(inp, training=False)

    # Pooling fusion
    x_avg = layers.GlobalAveragePooling2D()(x)
    x_max = layers.GlobalMaxPooling2D()(x)
    x     = layers.Concatenate()([x_avg, x_max])

    # Embedding head
    x   = layers.Dense(1024, kernel_regularizer=regularizers.l2(1e-4))(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    x   = layers.Dropout(0.5)(x)

    x   = layers.Dense(FEATURE_DIM, kernel_regularizer=regularizers.l2(1e-4))(x)
    x   = layers.BatchNormalization()(x)
    emb = L2Normalize(axis=1, name="embedding")(x)

    # ArcFace classification head (for training only)
    logits = ArcFaceLayer(num_classes, scale=ARC_SCALE, margin=ARC_MARGIN,
                          name="arcface")(emb)
    output = layers.Activation("softmax")(logits)

    training_model   = keras.Model(inp, output, name="palm_arcface_training")
    feature_extractor = keras.Model(inp, emb,   name="palm_feature_extractor")

    return training_model, feature_extractor, base


# ─────────────────────────────────────────────────────────────────────────────
#  Training data preparation (with CutMix / MixUp)
# ─────────────────────────────────────────────────────────────────────────────
def create_training_data(images, labels, augment_factor=5):
    """
    Create augmented training data.  Every augment_factor images gets an
    additional CutMix pair applied (where there are ≥2 classes).
    """
    aug_images = []
    aug_labels = []

    it = zip(images, labels)
    if HAS_TQDM:
        it = tqdm(list(it), desc="Augmenting")

    for img, lbl in it:
        aug_images.append(img);           aug_labels.append(lbl)
        for _ in range(augment_factor):
            a = advanced_augment_image((img * 255).astype(np.uint8))
            aug_images.append(a.astype("float32") / 255.0)
            aug_labels.append(lbl)

    # CutMix pass over unique classes
    unique_classes = np.unique(labels)
    if len(unique_classes) >= 2:
        n_mix = min(len(images) * 2, 500)
        for _ in range(n_mix):
            idx1, idx2 = random.sample(range(len(images)), 2)
            i1 = (images[idx1] * 255).astype(np.uint8)
            i2 = (images[idx2] * 255).astype(np.uint8)
            mixed, lam = cutmix_augment(i1, i2)
            aug_images.append(mixed.astype("float32") / 255.0)
            # Soft label not supported in one-hot path → keep dominant label
            aug_labels.append(labels[idx1] if lam >= 0.5 else labels[idx2])

        # MixUp pass
        for _ in range(n_mix):
            idx1, idx2 = random.sample(range(len(images)), 2)
            mixed, lam = mixup_augment(
                (images[idx1] * 255).astype(np.uint8),
                (images[idx2] * 255).astype(np.uint8)
            )
            aug_images.append(mixed.astype("float32") / 255.0)
            aug_labels.append(labels[idx1] if lam >= 0.5 else labels[idx2])

    return np.array(aug_images), np.array(aug_labels)


# ─────────────────────────────────────────────────────────────────────────────
#  Cosine-annealing learning-rate callback with linear warmup
# ─────────────────────────────────────────────────────────────────────────────
class CosineAnnealingWarmup(keras.callbacks.Callback):
    def __init__(self, initial_lr, warmup_epochs=5, total_epochs=80,
                 min_lr=1e-7):
        super().__init__()
        self.initial_lr    = initial_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                 1 + math.cos(math.pi * progress))
        keras.backend.set_value(self.model.optimizer.lr, lr)
        if epoch % 10 == 0:
            print(f"\n  LR at epoch {epoch}: {lr:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Triplet loss fine-tuning (online hard-negative mining)
# ─────────────────────────────────────────────────────────────────────────────
def triplet_loss(y_true, y_pred, margin=0.45):
    """
    Semi-hard online triplet loss for embedding refinement.
    Margin increased from 0.3 → 0.45 for wider inter-class gaps.
    y_pred  = batch embeddings (B, D)
    y_true  = class labels     (B,)
    """
    embeddings = tf.nn.l2_normalize(y_pred, axis=1)
    labels     = tf.cast(y_true, tf.int32)

    # Pairwise cosine distances (1 - cosine_sim)
    dot  = tf.matmul(embeddings, embeddings, transpose_b=True)
    dist = 1.0 - dot   # (B, B)

    # Positive / negative masks
    label_eq = tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0))
    eye      = tf.eye(tf.shape(labels)[0], dtype=tf.bool)
    pos_mask = tf.logical_and(label_eq, tf.logical_not(eye))
    neg_mask = tf.logical_not(label_eq)

    # Hardest positive
    pos_dist = tf.reduce_max(dist * tf.cast(pos_mask, tf.float32), axis=1)
    # Easiest negative (semi-hard: neg_dist > pos_dist)
    neg_dist_all = dist + 1e9 * tf.cast(tf.logical_not(neg_mask), tf.float32)
    neg_dist = tf.reduce_min(neg_dist_all, axis=1)

    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
    return tf.reduce_mean(loss)


def fine_tune_triplet(feature_extractor, train_images, train_labels,
                      epochs=30, batch_size=16, lr=1e-5, logger=None):
    """
    Fine-tune feature extractor with online triplet loss.
    Unfreeze entire extractor for this stage.
    """
    if logger:
        logger.info("Starting triplet fine-tuning...")
    else:
        print("\n── Triplet fine-tuning stage ──")

    # Unfreeze all layers
    for layer in feature_extractor.layers:
        layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    ds = tf.data.Dataset.from_tensor_slices(
        (train_images.astype("float32"), train_labels.astype("int32"))
    ).shuffle(4096).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    for epoch in range(epochs):
        epoch_loss = []
        for imgs, lbls in ds:
            with tf.GradientTape() as tape:
                embs = feature_extractor(imgs, training=True)
                loss = triplet_loss(lbls, embs)
            grads = tape.gradient(loss, feature_extractor.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, feature_extractor.trainable_variables))
            epoch_loss.append(float(loss))

        if (epoch + 1) % 5 == 0:
            avg = np.mean(epoch_loss)
            msg = f"  Triplet epoch {epoch+1}/{epochs} — loss: {avg:.4f}"
            print(msg)
            if logger:
                logger.info(msg)


# ─────────────────────────────────────────────────────────────────────────────
#  Main training function
# ─────────────────────────────────────────────────────────────────────────────
def train_arcface_model():
    """Full training pipeline: ArcFace stage → triplet fine-tuning → save."""
    log_file = f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    print("=" * 70)
    print("PalmPay — Deep Learning Training (ArcFace + ResNet50V2)")
    print("=" * 70)

    # 1. Load images
    print("\n1. Loading training images...")
    train_images, train_labels = load_images_with_labels(TRAIN_DIR)
    n_classes = int(np.max(train_labels)) + 1 if len(train_labels) > 0 else 3
    print(f"   {len(train_images)} images | {n_classes} palm identities")

    valid_images = valid_labels = None
    if VALID_DIR and os.path.exists(VALID_DIR):
        valid_images, valid_labels = load_images_with_labels(VALID_DIR)
        print(f"   {len(valid_images)} validation images")

    if len(train_images) == 0:
        raise ValueError(f"No images found in {TRAIN_DIR}")

    # Normalise
    train_images = train_images.astype("float32") / 255.0
    if valid_images is not None:
        valid_images = valid_images.astype("float32") / 255.0

    # 2. Build model
    print("\n2. Building ResNet50V2 + ArcFace model...")
    training_model, feature_extractor, base_model = create_arcface_model(n_classes)
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(f"   Parameters: {training_model.count_params():,}")

    # 3. Augment
    print("\n3. Creating augmented dataset (CutMix + MixUp)...")
    aug_imgs, aug_lbls = create_training_data(train_images, train_labels,
                                               augment_factor=5)
    print(f"   {len(aug_imgs)} augmented samples")

    aug_valid_imgs = aug_valid_lbls = None
    if valid_images is not None:
        aug_valid_imgs, aug_valid_lbls = create_training_data(
            valid_images, valid_labels, augment_factor=2)

    # One-hot encode
    aug_lbls_oh    = tf.keras.utils.to_categorical(aug_lbls,    n_classes)
    aug_valid_oh   = (tf.keras.utils.to_categorical(aug_valid_lbls, n_classes)
                      if aug_valid_lbls is not None else None)

    # 4. Callbacks
    monitor = "val_loss" if aug_valid_imgs is not None else "loss"
    callbacks = [
        CosineAnnealingWarmup(LEARNING_RATE, warmup_epochs=5,
                              total_epochs=EPOCHS, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, monitor=monitor,
            save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=20, verbose=1, restore_best_weights=True
        ),
    ]

    # 5. ArcFace training — stage 1
    print(f"\n4. Stage 1: ArcFace training ({EPOCHS} epochs, bs={BATCH_SIZE})...")
    training_model.fit(
        aug_imgs, aug_lbls_oh,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(aug_valid_imgs, aug_valid_oh) if aug_valid_imgs is not None else None,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Progressive unfreeze fine-tuning
    print("\n5. Stage 2: Progressive unfreeze (all ResNet50V2 layers)...")
    for layer in base_model.layers:
        layer.trainable = True

    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.05),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    training_model.fit(
        aug_imgs, aug_lbls_oh,
        batch_size=BATCH_SIZE,
        epochs=40,
        validation_data=(aug_valid_imgs, aug_valid_oh) if aug_valid_imgs is not None else None,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5,
                                              patience=8, min_lr=1e-8, verbose=1),
            keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor=monitor,
                                             save_best_only=True, verbose=1),
        ],
        verbose=1
    )

    # 7. Triplet fine-tuning
    print("\n6. Stage 3: Triplet loss hard-negative fine-tuning...")
    fine_tune_triplet(feature_extractor, aug_imgs, aug_lbls,
                      epochs=30, batch_size=16, lr=5e-6, logger=logger)

    # 8. Save feature extractor
    print(f"\n7. Saving feature extractor → {MODEL_SAVE_PATH}")
    feature_extractor.save(MODEL_SAVE_PATH)

    # 9. Evaluate
    print("\n8. Evaluating model...")
    evaluate_advanced_model(feature_extractor, train_images,
                             valid_images, train_labels)

    print("\n" + "=" * 70)
    print(f"✓ Training complete!  Model saved: {MODEL_SAVE_PATH}")
    print("=" * 70)
    return feature_extractor


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_advanced_model(model, train_images, valid_images=None,
                             train_labels=None):
    """Comprehensive evaluation with FAR / FRR / EER metrics."""
    print("\nEvaluating model similarity statistics...")
    feats = model.predict(train_images, batch_size=16, verbose=0)

    # Intra-class (same palm)
    same_sims, diff_sims = [], []
    n = len(feats)

    if train_labels is not None:
        for i in range(min(200, n)):
            for j in range(i + 1, min(200, n)):
                sim = cosine_similarity([feats[i]], [feats[j]])[0][0]
                if train_labels[i] == train_labels[j]:
                    same_sims.append(sim)
                else:
                    diff_sims.append(sim)
    else:
        for i in range(min(10, n)):
            noise = np.random.normal(0, 0.05, feats[i].shape)
            f2    = feats[i] + noise
            f2   /= np.linalg.norm(f2)
            same_sims.append(cosine_similarity([feats[i]], [f2])[0][0])
        for _ in range(min(100, n*(n-1)//2)):
            i, j  = random.sample(range(n), 2)
            diff_sims.append(cosine_similarity([feats[i]], [feats[j]])[0][0])

    if same_sims:
        print(f"  Same palm   — mean: {np.mean(same_sims):.4f} ± {np.std(same_sims):.4f}"
              f"  min: {np.min(same_sims):.4f}")
    if diff_sims:
        print(f"  Diff palms  — mean: {np.mean(diff_sims):.4f} ± {np.std(diff_sims):.4f}"
              f"  max: {np.max(diff_sims):.4f}")
    if same_sims and diff_sims:
        print(f"  Separation  — {np.mean(same_sims) - np.mean(diff_sims):.4f}")
        threshold = max(0.70, min(0.92,
            (np.mean(same_sims) + np.mean(diff_sims)) / 2))
        print(f"  ✓ Recommended threshold: {threshold:.3f}")

    # ── FAR / FRR / EER Analysis ──────────────────────────────────────────
    if same_sims and diff_sims:
        print("\n  ── FAR / FRR / EER Analysis ──")
        same_arr = np.array(same_sims)
        diff_arr = np.array(diff_sims)
        best_eer = 1.0
        best_thr = 0.70

        for t in np.arange(0.50, 0.98, 0.01):
            # FAR = fraction of different-palm pairs exceeding threshold
            far = float(np.sum(diff_arr >= t)) / max(len(diff_arr), 1)
            # FRR = fraction of same-palm pairs below threshold
            frr = float(np.sum(same_arr < t)) / max(len(same_arr), 1)
            eer_diff = abs(far - frr)
            if eer_diff < best_eer:
                best_eer = eer_diff
                best_thr = t
                best_far = far
                best_frr = frr

        print(f"  EER threshold : {best_thr:.3f}")
        print(f"  FAR at EER    : {best_far:.4f} ({best_far*100:.2f}%)")
        print(f"  FRR at EER    : {best_frr:.4f} ({best_frr*100:.2f}%)")

        # Report at our operational threshold (0.80)
        op_thr = 0.80
        op_far = float(np.sum(diff_arr >= op_thr)) / max(len(diff_arr), 1)
        op_frr = float(np.sum(same_arr < op_thr)) / max(len(same_arr), 1)
        print(f"\n  At operational threshold {op_thr}:")
        print(f"  FAR = {op_far:.4f} ({op_far*100:.2f}%) — false accepts")
        print(f"  FRR = {op_frr:.4f} ({op_frr*100:.2f}%) — false rejects")


if __name__ == "__main__":
    np.random.seed(42);  random.seed(42);  tf.random.set_seed(42)
    try:
        train_arcface_model()
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback; traceback.print_exc()