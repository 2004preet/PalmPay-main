"""
PalmPay — Professional Palm Recognition Training v7
====================================================
PURPOSE: Train a palm-embedding model that actually integrates with the app.

Key fixes over previous scripts:
  1. Proper identity labeling from flat image directory (groups of ~10)
  2. Right-sized architecture: EfficientNetB0 (5M params) for 400 images
  3. Single-scale 224×224 (fast on CPU, enough resolution for palm ridges)
  4. ArcFace with sensible margin (0.35) for small datasets
  5. Strong augmentation pipeline: rotation, scale, CLAHE, noise, elastic
  6. 3-stage training: frozen backbone → gradual unfreeze → triplet fine-tune
  7. Cosine warmup LR schedule
  8. Full biometric evaluation: EER, FAR/FRR, TAR@FAR=0.1%
  9. Output model is directly usable by app.py via PalmRecognizerPro class

Usage:
    python train_palm_pro.py
"""

import os, sys, cv2, math, glob, random, datetime, warnings, logging
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 16
EPOCHS_STAGE1   = 60          # Frozen backbone — fast
EPOCHS_STAGE2   = 80          # Fine-tune — main training
EPOCHS_STAGE3   = 40          # Triplet refinement
LEARNING_RATE   = 3e-4
MIN_LR          = 1e-7
WARMUP_EPOCHS   = 5
FEATURE_DIM     = 256         # Compact but discriminative
ARC_SCALE       = 30.0
ARC_MARGIN      = 0.35        # Gentle margin — better for small datasets
AUGMENT_FACTOR  = 8           # 8× augmentation to combat small dataset
IMGS_PER_PALM   = 10          # Expected images per palm identity

TRAIN_DIR       = "Files/Train"
VALID_DIR       = "Files/Valid"
MODEL_PATH      = "palm_model_v7.h5"

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────
log_file = f"train_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
log = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE ENHANCEMENT
# ─────────────────────────────────────────────────────────────────────────────
def enhance_palm(img_bgr, size=IMG_SIZE):
    """
    Biometric-grade palm image enhancement:
      1. Skin-based ROI crop (fast)
      2. CLAHE adaptive contrast
      3. Light denoise
      4. Unsharp mask
      5. Gamma normalize
    """
    try:
        # --- Fast center-weighted crop (avoid slow contour detection) ---
        h_img, w_img = img_bgr.shape[:2]
        # Crop central 70% (palm is usually centered in capture)
        margin_h, margin_w = int(h_img * 0.15), int(w_img * 0.15)
        crop = img_bgr[margin_h:h_img - margin_h, margin_w:w_img - margin_w]
        if crop.shape[0] > 50 and crop.shape[1] > 50:
            img_bgr = crop

        # --- CLAHE ---
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # --- Light denoise ---
        img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)

        # --- Unsharp mask ---
        blur = cv2.GaussianBlur(img_bgr, (0, 0), 2.0)
        img_bgr = cv2.addWeighted(img_bgr, 1.4, blur, -0.4, 0)

        # --- Gamma correction ---
        gray_mean = np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
        if gray_mean > 1:
            gamma = np.log(128.0 / (gray_mean + 1e-6)) / np.log(2.0)
            gamma = float(np.clip(gamma, 0.5, 2.5))
            lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                            for i in range(256)], dtype=np.uint8)
            img_bgr = cv2.LUT(img_bgr, lut)
    except Exception as e:
        pass  # Fallback: just resize

    img_bgr = cv2.resize(img_bgr, size)
    return img_bgr


# ─────────────────────────────────────────────────────────────────────────────
#  AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def augment(img):
    """Strong augmentation pipeline designed for palm biometrics."""
    img = img.copy()

    # Rotation (-20° to +20°)
    if random.random() > 0.3:
        angle = random.uniform(-20, 20)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Scale (85% to 115%)
    if random.random() > 0.4:
        scale = random.uniform(0.85, 1.15)
        h, w = img.shape[:2]
        nh, nw = int(h * scale), int(w * scale)
        img = cv2.resize(img, (nw, nh))
        if scale > 1:
            sy, sx = (nh - h) // 2, (nw - w) // 2
            img = img[sy:sy + h, sx:sx + w]
        else:
            py, px = (h - nh) // 2, (w - nw) // 2
            img = cv2.copyMakeBorder(img, py, h - nh - py, px, w - nw - px,
                                     cv2.BORDER_REFLECT)

    # Translation
    if random.random() > 0.5:
        tx = random.uniform(-0.06, 0.06) * img.shape[1]
        ty = random.uniform(-0.06, 0.06) * img.shape[0]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             borderMode=cv2.BORDER_REFLECT)

    # Brightness / contrast
    if random.random() > 0.3:
        alpha = random.uniform(0.7, 1.3)
        beta = random.uniform(-25, 25)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # HSV jitter
    if random.random() > 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-15, 15)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.8, 1.2), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Gaussian noise
    if random.random() > 0.4:
        noise = np.random.normal(0, random.uniform(3, 12), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Gaussian blur
    if random.random() > 0.6:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Elastic distortion
    if random.random() > 0.6:
        h, w = img.shape[:2]
        alpha_e = random.uniform(30, 80)
        sigma_e = random.uniform(5, 12)
        dx = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma_e
        ) * alpha_e
        dy = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma_e
        ) * alpha_e
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        img = cv2.remap(img, (x + dx).astype(np.float32),
                        (y + dy).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Perspective warp
    if random.random() > 0.65:
        h, w = img.shape[:2]
        s = random.uniform(0.02, 0.06)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [random.uniform(0, w * s), random.uniform(0, h * s)],
            [w - random.uniform(0, w * s), random.uniform(0, h * s)],
            [w - random.uniform(0, w * s), h - random.uniform(0, h * s)],
            [random.uniform(0, w * s), h - random.uniform(0, h * s)]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random erasing
    if random.random() > 0.6:
        h, w = img.shape[:2]
        eh = random.randint(10, h // 4)
        ew = random.randint(10, w // 4)
        ex = random.randint(0, w - ew)
        ey = random.randint(0, h - eh)
        img[ey:ey + eh, ex:ex + ew] = np.random.randint(0, 256, (eh, ew, 3), dtype=np.uint8)

    return img


# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM LAYERS
# ─────────────────────────────────────────────────────────────────────────────
class L2Norm(layers.Layer):
    """L2-normalize embeddings."""
    def call(self, x):
        return tf.nn.l2_normalize(x, axis=1)
    def get_config(self):
        return super().get_config()


class ArcFaceHead(layers.Layer):
    """ArcFace additive angular margin classification head."""
    def __init__(self, num_classes, scale=30.0, margin=0.35, **kw):
        super().__init__(**kw)
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.num_classes),
            initializer="glorot_normal", trainable=True, name="arc_W"
        )

    def call(self, emb, labels=None, training=None):
        cosine = tf.matmul(
            tf.nn.l2_normalize(emb, 1), tf.nn.l2_normalize(self.W, 0)
        )
        if labels is None or not training:
            return cosine * self.scale
        sine = tf.sqrt(tf.maximum(1.0 - cosine ** 2, 1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        oh = tf.one_hot(tf.cast(labels, tf.int32), self.num_classes)
        return (oh * phi + (1 - oh) * cosine) * self.scale

    def get_config(self):
        c = super().get_config()
        c.update(dict(num_classes=self.num_classes, scale=self.scale,
                      margin=self.margin))
        return c


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING — proper identity grouping
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(directory, imgs_per_palm=IMGS_PER_PALM):
    """
    Load palm images and assign identity labels.

    Strategy:
      - If subdirectories exist → each subdir = one identity
      - If flat directory → group sequential images by `imgs_per_palm`
        (e.g., IMG_0001–0010 = palm 0, IMG_0011–0020 = palm 1, ...)
    """
    images, labels = [], []

    subdirs = sorted([d for d in os.listdir(directory)
                      if os.path.isdir(os.path.join(directory, d))])

    if subdirs:
        # ── Subdirectory mode
        log.info(f"  Found {len(subdirs)} identity folders")
        for label_id, name in enumerate(subdirs):
            folder = os.path.join(directory, name)
            paths = sorted(glob.glob(os.path.join(folder, "*.[jJpPbB][pPnNmM][gGpP]")))
            for p in paths:
                raw = cv2.imread(p)
                if raw is None:
                    continue
                enh = enhance_palm(raw, IMG_SIZE)
                images.append(cv2.cvtColor(enh, cv2.COLOR_BGR2RGB))
                labels.append(label_id)
            log.info(f"    {name}: {sum(1 for l in labels if l == label_id)} images")
    else:
        # ── Flat directory mode — group by sequential chunks
        exts = ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG", "bmp", "BMP"]
        all_paths = []
        for ext in exts:
            all_paths.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
        all_paths = sorted(set(all_paths))

        n_imgs = len(all_paths)
        n_identities = max(1, n_imgs // imgs_per_palm)

        log.info(f"  Flat directory: {n_imgs} images → {n_identities} identities "
                 f"({imgs_per_palm} imgs/palm)")

        for i, p in enumerate(all_paths):
            raw = cv2.imread(p)
            if raw is None:
                continue
            enh = enhance_palm(raw, IMG_SIZE)
            images.append(cv2.cvtColor(enh, cv2.COLOR_BGR2RGB))
            labels.append(i // imgs_per_palm)

    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int32)

    n_classes = len(np.unique(labels))
    log.info(f"  Loaded: {len(images)} images, {n_classes} identities")
    return images, labels, n_classes


# ─────────────────────────────────────────────────────────────────────────────
#  AUGMENTED DATASET
# ─────────────────────────────────────────────────────────────────────────────
def create_augmented_set(images, labels, factor=AUGMENT_FACTOR):
    """Create augmented copies of the dataset."""
    aug_imgs, aug_lbls = [], []

    for img, lbl in zip(images, labels):
        # Original
        aug_imgs.append(img)
        aug_lbls.append(lbl)
        # Augmented copies
        for _ in range(factor):
            a = augment((img * 255).astype(np.uint8))
            aug_imgs.append(a.astype(np.float32) / 255.0)
            aug_lbls.append(lbl)

    # Shuffle
    idx = list(range(len(aug_imgs)))
    random.shuffle(idx)
    aug_imgs = np.array([aug_imgs[i] for i in idx])
    aug_lbls = np.array([aug_lbls[i] for i in idx])

    log.info(f"  Augmented: {len(aug_imgs)} samples "
             f"(original {len(images)} × {factor + 1})")
    return aug_imgs, aug_lbls


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes):
    """
    EfficientNetB0 + lightweight attention + ArcFace.
    ~5M params — right-sized for 400-image dataset.
    Returns: (training_model, feature_extractor, backbone)
    """
    inp = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input")

    # Backbone — EfficientNetB0 (5.3M params, 224×224 native)
    try:
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_tensor=inp
        )
        backbone_name = "EfficientNetB0"
    except Exception:
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_tensor=inp
        )
        backbone_name = "MobileNetV2"

    log.info(f"  Backbone: {backbone_name} ({backbone.count_params():,} params)")

    # Freeze backbone initially
    backbone.trainable = False

    x = backbone.output

    # Lightweight channel attention (SE-like)
    gap = layers.GlobalAveragePooling2D(name="gap")(x)
    gmp = layers.GlobalMaxPooling2D(name="gmp")(x)
    pool = layers.Concatenate(name="pool_cat")([gap, gmp])

    # Embedding head
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4), name="fc1")(pool)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.PReLU(name="prelu1")(x)
    x = layers.Dropout(0.4, name="drop1")(x)

    x = layers.Dense(FEATURE_DIM, kernel_regularizer=regularizers.l2(1e-4), name="fc2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    emb = L2Norm(name="l2_norm")(x)

    # ArcFace head
    logits = ArcFaceHead(num_classes, scale=ARC_SCALE, margin=ARC_MARGIN,
                         name="arcface")(emb)
    out = layers.Activation("softmax", name="softmax")(logits)

    train_model = keras.Model(inp, out, name="palm_train")
    feat_model = keras.Model(inp, emb, name="palm_feat")

    return train_model, feat_model, backbone


# ─────────────────────────────────────────────────────────────────────────────
#  LR SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
class CosineWarmupScheduler(keras.callbacks.Callback):
    def __init__(self, base_lr, warmup_epochs, total_epochs, min_lr=1e-7):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress))
        self.model.optimizer.learning_rate.assign(lr)


# ─────────────────────────────────────────────────────────────────────────────
#  TRIPLET FINE-TUNING (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────
def online_triplet_loss(labels, embeddings, margin=0.3):
    """Semi-hard online triplet loss with cosine distance."""
    emb = tf.nn.l2_normalize(embeddings, axis=1)
    # Cosine similarity matrix
    sim = tf.matmul(emb, emb, transpose_b=True)
    dist = 1.0 - sim  # cosine distance

    labels = tf.cast(labels, tf.int32)
    eq = tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0))
    eye = tf.eye(tf.shape(labels)[0], dtype=tf.bool)

    pos_mask = tf.logical_and(eq, tf.logical_not(eye))
    neg_mask = tf.logical_not(eq)

    # Hardest positive (furthest same-class)
    pos_dist = dist * tf.cast(pos_mask, tf.float32)
    hardest_pos = tf.reduce_max(pos_dist, axis=1)

    # Hardest negative (closest different-class)
    neg_dist = dist + 1e9 * tf.cast(tf.logical_not(neg_mask), tf.float32)
    hardest_neg = tf.reduce_min(neg_dist, axis=1)

    loss = tf.maximum(hardest_pos - hardest_neg + margin, 0.0)
    return tf.reduce_mean(loss)


def triplet_finetune(feat_model, images, labels, epochs=EPOCHS_STAGE3,
                     batch_size=32, lr=5e-6):
    """Stage 3: triplet fine-tuning on the feature extractor."""
    log.info(f"\n{'─' * 60}")
    log.info(f"STAGE 3: Triplet fine-tuning ({epochs} epochs)")
    log.info(f"{'─' * 60}")

    for layer in feat_model.layers:
        layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    ds = tf.data.Dataset.from_tensor_slices(
        (images, labels)
    ).shuffle(4096).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    best_loss = float("inf")
    patience_counter = 0
    max_patience = 12

    for epoch in range(epochs):
        losses = []
        for batch_imgs, batch_lbls in ds:
            with tf.GradientTape() as tape:
                embs = feat_model(batch_imgs, training=True)
                loss = online_triplet_loss(batch_lbls, embs)
            grads = tape.gradient(loss, feat_model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, feat_model.trainable_variables))
            losses.append(float(loss))

        avg_loss = np.mean(losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
            feat_model.save(MODEL_PATH)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            log.info(f"  Epoch {epoch + 1:3d}/{epochs}  loss={avg_loss:.5f}  "
                     f"best={best_loss:.5f}")

    # Reload best
    feat_model.load_weights(MODEL_PATH)
    log.info(f"  Triplet best loss: {best_loss:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(feat_model, images, labels, dataset_name="Eval"):
    """Full biometric evaluation: same/diff stats, FAR/FRR, EER, TAR@FAR."""
    log.info(f"\n{'═' * 60}")
    log.info(f"BIOMETRIC EVALUATION — {dataset_name}")
    log.info(f"{'═' * 60}")

    feats = feat_model.predict(images, batch_size=32, verbose=0)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)

    n = len(feats)
    same_sims, diff_sims = [], []

    cap = min(n, 500)
    for i in range(cap):
        for j in range(i + 1, cap):
            sim = float(np.dot(feats[i], feats[j]))
            if labels[i] == labels[j]:
                same_sims.append(sim)
            else:
                diff_sims.append(sim)

    if not same_sims or not diff_sims:
        log.info("  Not enough pairs for evaluation")
        return 0.80

    same_arr = np.array(same_sims)
    diff_arr = np.array(diff_sims)

    log.info(f"  Same-palm pairs  : {len(same_arr):,}")
    log.info(f"  Diff-palm pairs  : {len(diff_arr):,}")
    log.info(f"  Same sim  (mean±std): {same_arr.mean():.4f} ± {same_arr.std():.4f}  "
             f"min={same_arr.min():.4f}")
    log.info(f"  Diff sim  (mean±std): {diff_arr.mean():.4f} ± {diff_arr.std():.4f}  "
             f"max={diff_arr.max():.4f}")
    log.info(f"  Separation gap      : {same_arr.mean() - diff_arr.mean():.4f}")

    # FAR / FRR / EER
    best_eer_diff = 1.0
    best_thr = 0.70
    best_far = best_frr = 0.0

    log.info(f"\n  {'Threshold':>10} {'FAR':>8} {'FRR':>8} {'|FAR-FRR|':>10}")
    for t in np.arange(0.30, 0.99, 0.01):
        far = float(np.sum(diff_arr >= t)) / len(diff_arr)
        frr = float(np.sum(same_arr < t)) / len(same_arr)
        d = abs(far - frr)
        if t * 100 % 5 == 0:
            log.info(f"  {t:>10.2f} {far:>8.4f} {frr:>8.4f} {d:>10.4f}")
        if d < best_eer_diff:
            best_eer_diff = d
            best_thr = t
            best_far = far
            best_frr = frr

    eer = (best_far + best_frr) / 2
    log.info(f"\n  ★ EER threshold : {best_thr:.3f}")
    log.info(f"    FAR @ EER     : {best_far:.4f} ({best_far * 100:.2f}%)")
    log.info(f"    FRR @ EER     : {best_frr:.4f} ({best_frr * 100:.2f}%)")
    log.info(f"    ≈ EER         : {eer * 100:.2f}%")

    # TAR @ FAR = 0.1%
    best_tar = 0.0
    for t in np.arange(0.50, 0.999, 0.001):
        far = float(np.sum(diff_arr >= t)) / len(diff_arr)
        if far <= 0.001:
            tar = float(np.sum(same_arr >= t)) / len(same_arr)
            if tar > best_tar:
                best_tar = tar
    log.info(f"    TAR @ FAR=0.1%: {best_tar * 100:.2f}%")

    # Operational threshold (halfway between same mean and diff max)
    op_thr = float(np.clip(
        (same_arr.mean() + diff_arr.max()) / 2.0,
        max(0.60, best_thr - 0.05),
        min(0.95, best_thr + 0.05)
    ))
    op_far = float(np.sum(diff_arr >= op_thr)) / len(diff_arr)
    op_frr = float(np.sum(same_arr < op_thr)) / len(same_arr)
    log.info(f"\n  ★ Recommended threshold: {op_thr:.3f}")
    log.info(f"    FAR={op_far * 100:.2f}%  FRR={op_frr * 100:.2f}%")
    log.info(f"{'═' * 60}")

    return op_thr


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train():
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    log.info("=" * 60)
    log.info("PalmPay — Professional Training v7")
    log.info("=" * 60)

    # ── 1. Load data
    log.info(f"\n1. Loading data from {TRAIN_DIR} ...")
    train_imgs, train_lbls, n_classes = load_dataset(TRAIN_DIR)

    val_imgs = val_lbls = None
    if os.path.exists(VALID_DIR):
        val_imgs, val_lbls, _ = load_dataset(VALID_DIR)

    if len(train_imgs) < 10:
        raise ValueError(f"Need at least 10 images, found {len(train_imgs)}")

    # ── 2. Augment
    log.info(f"\n2. Creating augmented dataset (×{AUGMENT_FACTOR + 1}) ...")
    aug_imgs, aug_lbls = create_augmented_set(train_imgs, train_lbls, AUGMENT_FACTOR)
    aug_lbls_oh = keras.utils.to_categorical(aug_lbls, n_classes)

    val_data = None
    if val_imgs is not None and len(val_imgs) > 0:
        val_aug, val_aug_l = create_augmented_set(val_imgs, val_lbls, 3)
        val_data = (val_aug, keras.utils.to_categorical(val_aug_l, n_classes))

    # ── 3. Build model
    log.info(f"\n3. Building model (classes={n_classes}, dim={FEATURE_DIM}) ...")
    train_model, feat_model, backbone = build_model(n_classes)
    train_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    log.info(f"  Total params: {train_model.count_params():,}")

    monitor = "val_loss" if val_data else "loss"

    # ── 4. Stage 1: Frozen backbone
    log.info(f"\n{'─' * 60}")
    log.info(f"STAGE 1: Train head only ({EPOCHS_STAGE1} epochs, backbone frozen)")
    log.info(f"{'─' * 60}")

    callbacks_s1 = [
        CosineWarmupScheduler(LEARNING_RATE, WARMUP_EPOCHS, EPOCHS_STAGE1, MIN_LR),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor=monitor, save_best_only=True, verbose=1),
    ]

    train_model.fit(
        aug_imgs, aug_lbls_oh,
        batch_size=BATCH_SIZE, epochs=EPOCHS_STAGE1,
        validation_data=val_data,
        callbacks=callbacks_s1, verbose=1
    )

    # ── 5. Stage 2: Progressive unfreeze
    log.info(f"\n{'─' * 60}")
    log.info(f"STAGE 2: Fine-tune backbone ({EPOCHS_STAGE2} epochs)")
    log.info(f"{'─' * 60}")

    # Unfreeze last 30% of backbone layers
    n_layers = len(backbone.layers)
    unfreeze_from = int(n_layers * 0.7)
    for layer in backbone.layers[unfreeze_from:]:
        layer.trainable = True
    log.info(f"  Unfroze layers {unfreeze_from}–{n_layers} "
             f"({n_layers - unfreeze_from} layers)")

    train_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.1),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_s2 = [
        CosineWarmupScheduler(LEARNING_RATE * 0.1, 3, EPOCHS_STAGE2, MIN_LR),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=8, min_lr=1e-8, verbose=1),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor=monitor, save_best_only=True, verbose=1),
    ]

    # Re-augment for more variety
    aug_imgs2, aug_lbls2 = create_augmented_set(train_imgs, train_lbls, AUGMENT_FACTOR)
    aug_lbls2_oh = keras.utils.to_categorical(aug_lbls2, n_classes)

    train_model.fit(
        aug_imgs2, aug_lbls2_oh,
        batch_size=BATCH_SIZE, epochs=EPOCHS_STAGE2,
        validation_data=val_data,
        callbacks=callbacks_s2, verbose=1
    )

    # ── 6. Stage 3: Triplet fine-tuning
    triplet_finetune(feat_model, train_imgs, train_lbls)

    # ── 7. Save final
    log.info(f"\n7. Saving model → {MODEL_PATH}")
    feat_model.save(MODEL_PATH)

    # ── 8. Evaluate
    op_thr = evaluate(feat_model, train_imgs, train_lbls, "Train Set")
    if val_imgs is not None and len(val_imgs) > 0:
        op_thr_val = evaluate(feat_model, val_imgs, val_lbls, "Validation Set")
        op_thr = op_thr_val  # Use val threshold

    log.info(f"\n{'=' * 60}")
    log.info(f"✓ Training complete!")
    log.info(f"  Model        : {MODEL_PATH}")
    log.info(f"  Feature dim  : {FEATURE_DIM}")
    log.info(f"  Threshold    : {op_thr:.3f}")
    log.info(f"  Log          : {log_file}")
    log.info(f"{'=' * 60}")

    return feat_model, op_thr


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE CLASS — drop-in replacement for app.py
# ─────────────────────────────────────────────────────────────────────────────
class PalmRecognizerPro:
    """
    Production palm recognizer that uses the trained v7 model.
    Drop-in replacement for the old PalmRecognizer / MockPalmVerifier.
    
    Usage in app.py:
        from train_palm_pro import PalmRecognizerPro
        palm_recognizer = PalmRecognizerPro("palm_model_v7.h5", threshold=0.75)
    """

    def __init__(self, model_path=MODEL_PATH, threshold=0.75):
        self.threshold = threshold
        self.img_size = IMG_SIZE
        self.model = None
        self.model_path = model_path
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            log.warning(f"Model not found: {self.model_path}")
            return
        try:
            self.model = keras.models.load_model(
                self.model_path,
                custom_objects={"L2Norm": L2Norm, "ArcFaceHead": ArcFaceHead}
            )
            log.info(f"✓ Loaded palm model: {self.model_path} "
                     f"(dim={self.model.output_shape[-1]})")
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            self.model = None

    def _preprocess(self, img_bgr):
        """Enhance + resize to model input."""
        enh = enhance_palm(img_bgr, self.img_size)
        rgb = cv2.cvtColor(enh, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb[np.newaxis]

    def enroll(self, img_bgr):
        """Get embedding for registration. Returns 256D L2-normalized vector."""
        if self.model is None:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        x = self._preprocess(img_bgr)
        emb = self.model.predict(x, verbose=0)[0]
        return (emb / (np.linalg.norm(emb) + 1e-9)).astype(np.float32)

    def verify(self, probe_bgr, enrolled_emb):
        """Verify a probe against enrolled embedding.
        Returns (match: bool, similarity: float)."""
        if self.model is None:
            return False, 0.0
        emb = self.enroll(probe_bgr)
        sim = float(np.dot(emb, enrolled_emb))
        return sim >= self.threshold, sim

    def multi_frame_verify(self, stored_features, image_list):
        """Multi-frame ensemble verification.
        Returns (match: bool, best_similarity: float)."""
        stored = (np.frombuffer(stored_features, dtype=np.float32)
                  if isinstance(stored_features, bytes) else stored_features)

        if not image_list:
            return False, 0.0

        sims = []
        for img_bytes in image_list:
            try:
                if isinstance(img_bytes, (bytes, bytearray)):
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                elif isinstance(img_bytes, np.ndarray):
                    img = img_bytes
                else:
                    continue

                if img is None:
                    continue

                emb = self.enroll(img)
                sim = float(np.dot(emb, stored))
                sims.append(sim)
            except Exception as e:
                log.warning(f"Frame error: {e}")

        if not sims:
            return False, 0.0

        # Use mean of top-2 similarities (balance security & usability)
        sims_sorted = sorted(sims, reverse=True)
        score = np.mean(sims_sorted[:min(2, len(sims_sorted))])

        # Check consistency
        if len(sims) >= 2 and np.std(sims) > 0.15:
            log.warning(f"Inconsistent frames: std={np.std(sims):.3f}")
            return False, float(score)

        return score >= self.threshold, float(score)


if __name__ == "__main__":
    train()
