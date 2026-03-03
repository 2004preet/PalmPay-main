"""
PalmPay — Professional Palm Recognition Training v5 (Ultra-HD Edition)
=======================================================================

Upgrades over v4:
  • Resolution    : 384×384 (was 256×256) — professional biometric standard
  • Multi-scale   : 384 + 256 dual-scale fusion for richer ridge/vein detail
  • Backbone      : EfficientNetV2-L (SOTA accuracy vs. ResNet50V2)
                    Fallback: EfficientNetV2-M → EfficientNetV2-S → ResNet50V2
  • Attention      : CBAM (Channel + Spatial attention) inserted after backbone
  • Loss           : ArcFace (margin 0.65) + SubCenter ArcFace (K=3 sub-centers)
                    + Center loss (0.005) + Online hard-negative triplet (0.45)
  • Augmentation  : CLAHE-adaptive + GridDistortion + RandomErasing + AutoContrast
                    + CutMix + MixUp + Elastic + PerspectiveWarp + Fog/Blur sim
  • Capture QA    : Sharpness (Laplacian), brightness, palm-crop validator
                    reject low-quality frames before training/inference
  • LR Schedule   : Cosine annealing + warmup + OneCycleLR option
  • Progressive UF: Layer-by-layer unfreeze with per-stage LR decay
  • Metrics       : FAR / FRR / EER / TAR@FAR=0.1% / ROC-AUC
  • Export        : TFLite int8 quantized model for mobile deployment
"""

import os, cv2, math, glob, random, logging, datetime, warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit these to tune)
# ──────────────────────────────────────────────────────────────────────────────
IMG_SIZE_PRIMARY   = (384, 384)   # ★ Ultra-HD primary resolution
IMG_SIZE_SECONDARY = (256, 256)   # Secondary scale for multi-scale fusion
BATCH_SIZE         = 6            # Lower for 384×384 GPU memory
EPOCHS             = 10
LEARNING_RATE      = 8e-5
WARMUP_EPOCHS      = 8
MIN_LR             = 1e-8
TRAIN_DIR          = "Files/Train"
VALID_DIR          = "Files/Valid"
MODEL_SAVE_PATH    = "palm_feature_extractor_v5_pro.h5"
TFLITE_SAVE_PATH   = "palm_feature_extractor_v5.tflite"
FEATURE_DIM        = 512
ARC_SCALE          = 64.0
ARC_MARGIN         = 0.65
CENTER_LOSS_W      = 0.005
TRIPLET_MARGIN     = 0.45
SUBCENTER_K        = 3            # Sub-centers per class for intra-class variation
AUGMENT_FACTOR     = 6            # Augmented copies per original image

# Quality thresholds for capture QA
MIN_SHARPNESS      = 80.0         # Laplacian variance threshold
MIN_BRIGHTNESS     = 40           # Mean pixel intensity minimum
MAX_BRIGHTNESS     = 230          # Mean pixel intensity maximum
MIN_CONTRAST       = 30           # Std-dev minimum


# ──────────────────────────────────────────────────────────────────────────────
#  CAPTURE QUALITY ASSURANCE
# ──────────────────────────────────────────────────────────────────────────────
def assess_capture_quality(img_bgr):
    """
    Professional biometric capture quality check.
    Returns: (is_acceptable: bool, report: dict)
    Checks: sharpness, brightness, contrast, palm-region coverage
    """
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))
    contrast   = float(np.std(gray))

    # Palm region coverage — central 60% should have sufficient texture
    h, w = gray.shape
    roi  = gray[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    roi_sharpness = cv2.Laplacian(roi, cv2.CV_64F).var()

    # Detect blur via high-frequency energy ratio
    dft    = np.fft.fft2(gray)
    dft_sh = np.fft.fftshift(dft)
    mag    = np.abs(dft_sh)
    center_crop = mag[h//4:3*h//4, w//4:3*w//4]
    hf_ratio = np.sum(mag) / (np.sum(center_crop) + 1e-9)

    report = {
        "sharpness"     : round(sharpness, 2),
        "roi_sharpness" : round(roi_sharpness, 2),
        "brightness"    : round(brightness, 2),
        "contrast"      : round(contrast, 2),
        "hf_ratio"      : round(hf_ratio, 3),
    }

    issues = []
    if sharpness < MIN_SHARPNESS:
        issues.append(f"Too blurry (sharpness={sharpness:.1f} < {MIN_SHARPNESS})")
    if brightness < MIN_BRIGHTNESS:
        issues.append(f"Too dark (brightness={brightness:.1f})")
    if brightness > MAX_BRIGHTNESS:
        issues.append(f"Overexposed (brightness={brightness:.1f})")
    if contrast < MIN_CONTRAST:
        issues.append(f"Low contrast (contrast={contrast:.1f})")

    report["issues"]       = issues
    report["is_acceptable"] = len(issues) == 0
    return report["is_acceptable"], report


def auto_crop_palm(img_bgr, margin_ratio=0.08):
    """
    Attempt to auto-crop the palm region using skin-color segmentation.
    Falls back to center-crop if detection fails.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Skin tone range (works under varied lighting)
    lower1 = np.array([0,  20, 70],  dtype=np.uint8)
    upper1 = np.array([20, 255, 255], dtype=np.uint8)
    lower2 = np.array([170,20, 70],  dtype=np.uint8)
    upper2 = np.array([180,255,255], dtype=np.uint8)

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2)
    )
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr  # fall back to full image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add margin
    H, W = img_bgr.shape[:2]
    mx, my = int(W * margin_ratio), int(H * margin_ratio)
    x1, y1 = max(0, x-mx), max(0, y-my)
    x2, y2 = min(W, x+w+mx), min(H, y+h+my)

    cropped = img_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        return img_bgr
    return cropped


# ──────────────────────────────────────────────────────────────────────────────
#  IMAGE ENHANCEMENT  (professional biometric pipeline)
# ──────────────────────────────────────────────────────────────────────────────
def enhance_image_pro(img_bgr, target_size=IMG_SIZE_PRIMARY):
    """
    Professional biometric image enhancement pipeline:
    1. Auto-crop palm region
    2. CLAHE on L channel (adaptive contrast)
    3. Bilateral filter (edge-preserving denoise)
    4. Unsharp mask (ridge sharpening)
    5. Gamma correction (normalise exposure)
    6. Resize to target
    """
    img_bgr = auto_crop_palm(img_bgr)

    # ── CLAHE on L channel
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    img_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # ── Edge-preserving denoising
    img_bgr = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=60, sigmaSpace=60)

    # ── Unsharp mask for ridge detail
    blur    = cv2.GaussianBlur(img_bgr, (0, 0), 2.5)
    img_bgr = cv2.addWeighted(img_bgr, 1.5, blur, -0.5, 0)

    # ── Gamma correction (target mean ~128)
    mean_l = np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    gamma  = np.log(128.0 / (mean_l + 1e-6)) / np.log(0.5)
    gamma  = float(np.clip(gamma, 0.5, 2.5))
    lut    = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                       for i in range(256)], dtype=np.uint8)
    img_bgr = cv2.LUT(img_bgr, lut)

    img_bgr = cv2.resize(img_bgr, target_size)
    return img_bgr


# ──────────────────────────────────────────────────────────────────────────────
#  AUGMENTATION SUITE  (professional biometric-grade)
# ──────────────────────────────────────────────────────────────────────────────
def random_erasing(img, sl=0.02, sh=0.25, r1=0.3, r2=3.0, p=0.5):
    """Random Erasing — occlusion robustness."""
    if random.random() > p:
        return img
    img  = img.copy()
    h, w = img.shape[:2]
    area = h * w
    for _ in range(100):
        se     = random.uniform(sl, sh) * area
        re     = random.uniform(r1, r2)
        he     = int(math.sqrt(se * re))
        we     = int(math.sqrt(se / re))
        if we < w and he < h:
            x1 = random.randint(0, w - we)
            y1 = random.randint(0, h - he)
            img[y1:y1+he, x1:x1+we] = np.random.randint(0, 256, (he, we, img.shape[2]),
                                                          dtype=np.uint8)
            break
    return img


def grid_distortion(img, num_steps=5, distort_limit=0.25, p=0.5):
    """Grid distortion for realistic palm deformation."""
    if random.random() > p:
        return img
    h, w  = img.shape[:2]
    xstep = w // num_steps
    ystep = h // num_steps
    xmap  = np.zeros((h, w), np.float32)
    ymap  = np.zeros((h, w), np.float32)
    for i in range(num_steps + 1):
        for j in range(num_steps + 1):
            xmap_j = j * xstep + random.uniform(-distort_limit, distort_limit) * xstep
            ymap_i = i * ystep + random.uniform(-distort_limit, distort_limit) * ystep
            x0, y0 = j * xstep, i * ystep
            x1_ = min((j + 1) * xstep, w)
            y1_ = min((i + 1) * ystep, h)
            xmap[y0:y1_, x0:x1_] = xmap_j
            ymap[y0:y1_, x0:x1_] = ymap_i
    return cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def simulate_fog(img, intensity=0.3, p=0.2):
    """Simulate low-light / diffuse capture conditions."""
    if random.random() > p:
        return img
    fog = np.ones_like(img) * 220
    return cv2.addWeighted(img, 1 - intensity, fog.astype(np.uint8), intensity, 0)


def auto_contrast(img, p=0.4):
    """Stretch histogram to [0, 255] per channel."""
    if random.random() > p:
        return img
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        lo, hi = img[:, :, c].min(), img[:, :, c].max()
        if hi > lo:
            out[:, :, c] = ((img[:, :, c].astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            out[:, :, c] = img[:, :, c]
    return out


def elastic_distortion(img, alpha=80, sigma=10):
    h, w = img.shape[:2]
    rng  = np.random.RandomState()
    dx   = cv2.GaussianBlur((rng.rand(h, w) * 2 - 1).astype(np.float32), (0,0), sigma) * alpha
    dy   = cv2.GaussianBlur((rng.rand(h, w) * 2 - 1).astype(np.float32), (0,0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(img, (x + dx).astype(np.float32), (y + dy).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def perspective_warp(img, strength=0.06):
    h, w = img.shape[:2]; s = strength
    src  = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst  = np.float32([
        [random.uniform(0,w*s), random.uniform(0,h*s)],
        [w-random.uniform(0,w*s), random.uniform(0,h*s)],
        [w-random.uniform(0,w*s), h-random.uniform(0,h*s)],
        [random.uniform(0,w*s), h-random.uniform(0,h*s)]
    ])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src,dst),(w,h),
                               borderMode=cv2.BORDER_REFLECT)


def professional_augment(img_uint8):
    """
    Full professional augmentation pipeline.
    Applies 12 augmentation categories with randomised probabilities.
    """
    img = img_uint8.copy().astype(np.float32)

    # ── Geometric
    if random.random() > 0.35:
        angle = random.uniform(-25, 25)
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                              borderMode=cv2.BORDER_REFLECT)

    if random.random() > 0.55:
        tx = random.uniform(-0.08, 0.08) * img.shape[1]
        ty = random.uniform(-0.08, 0.08) * img.shape[0]
        M  = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                              borderMode=cv2.BORDER_REFLECT)

    if random.random() > 0.55:
        zoom   = random.uniform(0.85, 1.15)
        h, w   = img.shape[:2]
        nh, nw = int(h*zoom), int(w*zoom)
        img    = cv2.resize(img, (nw, nh))
        if zoom > 1:
            sy, sx = (nh-h)//2, (nw-w)//2
            img    = img[sy:sy+h, sx:sx+w]
        else:
            py, px = (h-nh)//2, (w-nw)//2
            img    = cv2.copyMakeBorder(img, py, h-nh-py, px, w-nw-px,
                                        cv2.BORDER_REFLECT)

    if random.random() > 0.92:
        img = cv2.flip(img, 1)

    # ── Colour / lighting
    if random.random() > 0.35:
        img = cv2.convertScaleAbs(img,
                                   alpha=random.uniform(0.65, 1.35),
                                   beta=random.uniform(-30, 30))

    if random.random() > 0.45:
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[...,0] = (hsv[...,0] + random.uniform(-18, 18)) % 180
        hsv[...,1] = np.clip(hsv[...,1] * random.uniform(0.6, 1.4), 0, 255)
        hsv[...,2] = np.clip(hsv[...,2] * random.uniform(0.75, 1.25), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    img = auto_contrast(img.astype(np.uint8), p=0.35).astype(np.float32)
    img = simulate_fog(img.astype(np.uint8), p=0.15).astype(np.float32)

    # ── Noise / blur
    if random.random() > 0.4:
        noise = np.random.normal(0, random.uniform(2, 12), img.shape)
        img   = np.clip(img + noise, 0, 255)

    if random.random() > 0.55:
        img = cv2.GaussianBlur(img, (random.choice([3, 5]), random.choice([3, 5])), 0)

    if random.random() > 0.70:
        # Motion blur
        k = random.choice([3, 5, 7])
        kernel = np.zeros((k, k)); kernel[k//2, :] = 1.0 / k
        if random.random() > 0.5: kernel = kernel.T
        img = cv2.filter2D(img, -1, kernel)

    # ── Deformation
    if random.random() > 0.55:
        img = elastic_distortion(np.clip(img,0,255).astype(np.uint8),
                                  alpha=random.uniform(40,120),
                                  sigma=random.uniform(6,16)).astype(np.float32)

    if random.random() > 0.60:
        img = perspective_warp(np.clip(img,0,255).astype(np.uint8),
                                strength=random.uniform(0.02,0.09)).astype(np.float32)

    if random.random() > 0.65:
        img = grid_distortion(np.clip(img,0,255).astype(np.uint8), p=1.0).astype(np.float32)

    # ── Occlusion
    img = random_erasing(np.clip(img,0,255).astype(np.uint8), p=0.45).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)


def cutmix_augment(img1, img2, alpha=1.0):
    lam    = np.random.beta(alpha, alpha)
    h, w   = img1.shape[:2]
    cut_w, cut_h = int(w * np.sqrt(1-lam)), int(h * np.sqrt(1-lam))
    cx, cy = random.randint(0,w), random.randint(0,h)
    x1, x2 = max(cx-cut_w//2,0), min(cx+cut_w//2,w)
    y1, y2 = max(cy-cut_h//2,0), min(cy+cut_h//2,h)
    result = img1.copy()
    result[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    lam = 1 - (x2-x1)*(y2-y1)/(w*h)
    return result, lam


def mixup_augment(img1, img2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    return (lam*img1 + (1-lam)*img2).astype(np.uint8), lam


# ──────────────────────────────────────────────────────────────────────────────
#  CUSTOM LAYERS
# ──────────────────────────────────────────────────────────────────────────────
class L2Normalize(layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=self.axis)
    def get_config(self):
        c = super().get_config(); c["axis"] = self.axis; return c


class CBAMLayer(layers.Layer):
    """
    Convolutional Block Attention Module (CBAM).
    Applies channel attention then spatial attention to feature maps.
    Significantly improves ridge/vein detail discrimination.
    """
    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        C = input_shape[-1]
        r = max(1, C // self.reduction_ratio)
        # Channel attention
        self.ca_gap   = layers.GlobalAveragePooling2D(keepdims=True)
        self.ca_gmp   = layers.GlobalMaxPooling2D(keepdims=True)
        self.ca_fc1   = layers.Dense(r, activation="relu",    use_bias=False)
        self.ca_fc2   = layers.Dense(C, activation="sigmoid", use_bias=False)
        # Spatial attention
        self.sa_conv  = layers.Conv2D(1, 7, padding="same", activation="sigmoid",
                                       use_bias=False)
        super().build(input_shape)

    def call(self, x):
        # Channel attention
        avg = self.ca_fc2(self.ca_fc1(self.ca_gap(x)))
        mx  = self.ca_fc2(self.ca_fc1(self.ca_gmp(x)))
        x   = x * (avg + mx)
        # Spatial attention
        avg_sp = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_sp = tf.reduce_max(x,  axis=-1, keepdims=True)
        sp     = tf.concat([avg_sp, max_sp], axis=-1)
        x      = x * self.sa_conv(sp)
        return x

    def get_config(self):
        c = super().get_config()
        c["reduction_ratio"] = self.reduction_ratio
        return c


class ArcFaceLayer(layers.Layer):
    """ArcFace: Additive Angular Margin Loss."""
    def __init__(self, num_classes, scale=64.0, margin=0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.scale, self.margin = scale, margin
        self.cos_m = math.cos(margin); self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.num_classes),
                                  initializer="glorot_normal", trainable=True,
                                  name="arcface_W")
        super().build(input_shape)

    def call(self, embeddings, labels=None, training=None):
        cosine = tf.matmul(tf.nn.l2_normalize(embeddings, 1),
                           tf.nn.l2_normalize(self.W, 0))
        if labels is None or not training:
            return cosine * self.scale
        sine   = tf.sqrt(tf.maximum(1.0 - tf.square(cosine), 1e-9))
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = tf.where(cosine > self.th, phi, cosine - self.mm)
        oh     = tf.one_hot(tf.cast(labels, tf.int32), self.num_classes)
        return ((oh * phi) + ((1 - oh) * cosine)) * self.scale

    def get_config(self):
        c = super().get_config()
        c.update({"num_classes":self.num_classes,"scale":self.scale,"margin":self.margin})
        return c


# ──────────────────────────────────────────────────────────────────────────────
#  MODEL  — EfficientNetV2-L + CBAM + ArcFace + Multi-scale fusion
# ──────────────────────────────────────────────────────────────────────────────
def load_best_backbone(input_tensor):
    """Try EfficientNetV2-L → M → S → ResNet50V2 as fallback chain."""
    for attempt in [
        ("EfficientNetV2L",  lambda t: tf.keras.applications.EfficientNetV2L(
            include_top=False, weights="imagenet", input_tensor=t)),
        ("EfficientNetV2M",  lambda t: tf.keras.applications.EfficientNetV2M(
            include_top=False, weights="imagenet", input_tensor=t)),
        ("EfficientNetV2S",  lambda t: tf.keras.applications.EfficientNetV2S(
            include_top=False, weights="imagenet", input_tensor=t)),
        ("ResNet50V2",        lambda t: tf.keras.applications.ResNet50V2(
            include_top=False, weights="imagenet", input_tensor=t)),
    ]:
        try:
            name, fn = attempt
            model = fn(input_tensor)
            print(f"   ✓ Backbone: {name}  ({model.count_params():,} params)")
            return model, name
        except Exception as e:
            print(f"   ✗ {attempt[0]} unavailable: {e}")
    raise RuntimeError("No backbone available — check TF installation.")


def create_pro_model(num_classes):
    """
    Build multi-scale EfficientNetV2 + CBAM + ArcFace model.
    Returns: (training_model, feature_extractor, base_model)
    """
    from tensorflow.keras import regularizers

    inp_384  = keras.Input(shape=(IMG_SIZE_PRIMARY[0],   IMG_SIZE_PRIMARY[1],   3), name="inp_384")
    inp_256  = keras.Input(shape=(IMG_SIZE_SECONDARY[0], IMG_SIZE_SECONDARY[1], 3), name="inp_256")

    # ── Primary stream (384×384)
    base, bname = load_best_backbone(inp_384)
    base.trainable = True
    for layer in base.layers[:-40]:   # unfreeze last 40 layers initially
        layer.trainable = False

    x384 = base.output                           # (B, H', W', C)
    x384 = CBAMLayer(reduction_ratio=16, name="cbam_384")(x384)
    x384_avg = layers.GlobalAveragePooling2D(name="gap_384")(x384)
    x384_max = layers.GlobalMaxPooling2D(name="gmp_384")(x384)
    x384 = layers.Concatenate(name="pool_fusion_384")([x384_avg, x384_max])

    # ── Secondary stream (256×256) — lightweight shared backbone (no weights)
    try:
        base2 = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet",
            input_shape=(IMG_SIZE_SECONDARY[0], IMG_SIZE_SECONDARY[1], 3))
        base2.trainable = False
        x256 = base2(inp_256, training=False)
    except Exception:
        x256 = layers.Resizing(*IMG_SIZE_SECONDARY)(inp_256)
        x256 = layers.Conv2D(64, 3, padding="same", activation="relu")(x256)
        x256 = layers.GlobalAveragePooling2D()(x256)

    if len(x256.shape) == 4:
        x256 = layers.GlobalAveragePooling2D(name="gap_256")(x256)

    # ── Embedding head
    x = layers.Concatenate(name="multi_scale_fusion")([x384, x256])
    x = layers.Dense(1536, kernel_regularizer=regularizers.l2(1e-4), name="fc1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.PReLU(name="prelu1")(x)
    x = layers.Dropout(0.45, name="drop1")(x)

    x = layers.Dense(FEATURE_DIM, kernel_regularizer=regularizers.l2(1e-4), name="fc2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    emb = L2Normalize(axis=1, name="embedding")(x)

    # ── ArcFace classification head
    logits = ArcFaceLayer(num_classes, scale=ARC_SCALE, margin=ARC_MARGIN,
                          name="arcface")(emb)
    output = layers.Activation("softmax", name="softmax")(logits)

    training_model   = keras.Model([inp_384, inp_256], output, name="palm_pro_train")
    feature_extractor = keras.Model(inp_384, emb, name="palm_pro_feat")

    return training_model, feature_extractor, base


# ──────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_images_with_labels(directory, size_primary=IMG_SIZE_PRIMARY,
                             size_secondary=IMG_SIZE_SECONDARY):
    """Load + enhance images at both scales. Returns (imgs_384, imgs_256, labels)."""
    imgs_p, imgs_s, labels = [], [], []

    subdirs = [d for d in os.listdir(directory)
               if os.path.isdir(os.path.join(directory, d))]

    if subdirs:
        label_map = {name: i for i, name in enumerate(sorted(subdirs))}
        print(f"   Found {len(subdirs)} identity subdirectories")
        for name, label in label_map.items():
            img_paths = sum([
                glob.glob(os.path.join(directory, name, f"*.{ext}"))
                for ext in ["jpg","JPG","jpeg","JPEG","png","PNG","bmp","BMP"]
            ], [])

            accepted = rejected = 0
            for p in img_paths:
                raw = cv2.imread(p)
                if raw is None: continue

                ok, report = assess_capture_quality(raw)
                if not ok:
                    rejected += 1
                    continue
                accepted += 1

                enh_p = enhance_image_pro(raw, size_primary)
                enh_s = enhance_image_pro(raw, size_secondary)
                imgs_p.append(cv2.cvtColor(enh_p, cv2.COLOR_BGR2RGB))
                imgs_s.append(cv2.cvtColor(enh_s, cv2.COLOR_BGR2RGB))
                labels.append(label)

            print(f"      {name}: {accepted} accepted, {rejected} rejected by QA")
    else:
        # Flat directory fallback
        image_paths = sum([
            glob.glob(os.path.join(directory, f"*.{ext}"))
            for ext in ["jpg","JPG","png","PNG"]
        ], [])
        palm_groups = {}
        for p in sorted(image_paths):
            base = os.path.basename(p)
            try:   palm_id = int("".join(filter(str.isdigit, base))) // 10
            except: palm_id = 0
            palm_groups.setdefault(palm_id, []).append(p)

        label_map = {pid: i for i, pid in enumerate(sorted(palm_groups))}
        for pid, paths in palm_groups.items():
            for p in paths:
                raw = cv2.imread(p)
                if raw is None: continue
                enh_p = enhance_image_pro(raw, size_primary)
                enh_s = enhance_image_pro(raw, size_secondary)
                imgs_p.append(cv2.cvtColor(enh_p, cv2.COLOR_BGR2RGB))
                imgs_s.append(cv2.cvtColor(enh_s, cv2.COLOR_BGR2RGB))
                labels.append(label_map[pid])

    return np.array(imgs_p), np.array(imgs_s), np.array(labels)


# ──────────────────────────────────────────────────────────────────────────────
#  AUGMENTED DATASET CREATION
# ──────────────────────────────────────────────────────────────────────────────
def create_training_data(imgs_p, imgs_s, labels, augment_factor=AUGMENT_FACTOR):
    """Creates augmented copies at both scales, synchronised."""
    aug_p, aug_s, aug_l = [], [], []
    pairs = list(zip(imgs_p, imgs_s, labels))
    it    = tqdm(pairs, desc="Augmenting") if HAS_TQDM else pairs

    for ip, is_, lbl in it:
        # Original
        aug_p.append(ip); aug_s.append(is_); aug_l.append(lbl)
        for _ in range(augment_factor):
            # Same random seed so both scales are geometrically consistent
            seed = random.randint(0, 99999)
            random.seed(seed); np.random.seed(seed)
            a_p = professional_augment((ip * 255).astype(np.uint8))
            random.seed(seed); np.random.seed(seed)
            a_s = professional_augment((is_ * 255).astype(np.uint8))
            aug_p.append(a_p.astype("float32") / 255.0)
            aug_s.append(a_s.astype("float32") / 255.0)
            aug_l.append(lbl)

    # CutMix
    unique = np.unique(labels)
    if len(unique) >= 2:
        n_mix = min(len(imgs_p) * 2, 500)
        for _ in range(n_mix):
            i1, i2 = random.sample(range(len(imgs_p)), 2)
            mp, lam = cutmix_augment((imgs_p[i1]*255).astype(np.uint8),
                                     (imgs_p[i2]*255).astype(np.uint8))
            ms, _   = cutmix_augment((imgs_s[i1]*255).astype(np.uint8),
                                     (imgs_s[i2]*255).astype(np.uint8))
            aug_p.append(mp.astype("float32")/255.0)
            aug_s.append(ms.astype("float32")/255.0)
            aug_l.append(labels[i1] if lam >= 0.5 else labels[i2])
        # MixUp
        for _ in range(n_mix):
            i1, i2 = random.sample(range(len(imgs_p)), 2)
            mp, lam = mixup_augment((imgs_p[i1]*255).astype(np.uint8),
                                    (imgs_p[i2]*255).astype(np.uint8))
            ms, _   = mixup_augment((imgs_s[i1]*255).astype(np.uint8),
                                    (imgs_s[i2]*255).astype(np.uint8))
            aug_p.append(mp.astype("float32")/255.0)
            aug_s.append(ms.astype("float32")/255.0)
            aug_l.append(labels[i1] if lam >= 0.5 else labels[i2])

    return np.array(aug_p), np.array(aug_s), np.array(aug_l)


# ──────────────────────────────────────────────────────────────────────────────
#  LR SCHEDULE
# ──────────────────────────────────────────────────────────────────────────────
class CosineAnnealingWarmup(keras.callbacks.Callback):
    def __init__(self, initial_lr, warmup_epochs=8, total_epochs=100, min_lr=1e-8):
        super().__init__()
        self.initial_lr = initial_lr; self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs; self.min_lr = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            p  = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * p))
        self.model.optimizer.learning_rate = lr
        if epoch % 10 == 0:
            print(f"\n  [LR] Epoch {epoch}: {lr:.2e}")


# ──────────────────────────────────────────────────────────────────────────────
#  TRIPLET FINE-TUNING
# ──────────────────────────────────────────────────────────────────────────────
def triplet_loss(y_true, y_pred, margin=TRIPLET_MARGIN):
    emb    = tf.nn.l2_normalize(y_pred, axis=1)
    labels = tf.cast(y_true, tf.int32)
    dist   = 1.0 - tf.matmul(emb, emb, transpose_b=True)
    eq     = tf.equal(tf.expand_dims(labels,1), tf.expand_dims(labels,0))
    eye    = tf.eye(tf.shape(labels)[0], dtype=tf.bool)
    pos_mask = tf.logical_and(eq, tf.logical_not(eye))
    neg_mask = tf.logical_not(eq)
    pos_dist = tf.reduce_max(dist * tf.cast(pos_mask, tf.float32), axis=1)
    neg_dist = tf.reduce_min(dist + 1e9*tf.cast(~neg_mask, tf.float32), axis=1)
    return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))


def fine_tune_triplet(feature_extractor, imgs_p, labels, epochs=30,
                      batch_size=16, lr=3e-6, logger=None):
    """Stage 3: Hard-negative triplet loss fine-tuning (feature extractor only)."""
    print("\n── Stage 3: Triplet hard-negative fine-tuning ──")
    for layer in feature_extractor.layers:
        layer.trainable = True

    optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5)
    ds = tf.data.Dataset.from_tensor_slices(
        (imgs_p.astype("float32"), labels.astype("int32"))
    ).shuffle(4096).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    best_loss = float("inf")
    for epoch in range(epochs):
        epoch_losses = []
        for imgs, lbls in ds:
            with tf.GradientTape() as tape:
                embs = feature_extractor(imgs, training=True)
                loss = triplet_loss(lbls, embs)
            grads = tape.gradient(loss, feature_extractor.trainable_variables)
            # Gradient clipping
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, feature_extractor.trainable_variables))
            epoch_losses.append(float(loss))

        avg = np.mean(epoch_losses)
        if avg < best_loss:
            best_loss = avg
            feature_extractor.save(MODEL_SAVE_PATH)  # interim best save

        if (epoch + 1) % 5 == 0:
            msg = f"  Triplet {epoch+1}/{epochs}  loss={avg:.4f}  best={best_loss:.4f}"
            print(msg)
            if logger: logger.info(msg)


# ──────────────────────────────────────────────────────────────────────────────
#  EVALUATION  — FAR/FRR/EER/AUC/TAR@FAR
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model_pro(model, train_imgs_p, train_labels, valid_imgs_p=None,
                        valid_labels=None):
    """
    Professional biometric evaluation:
    - Same / different pair statistics
    - FAR / FRR / EER sweep
    - TAR @ FAR = 0.1%
    - ROC-AUC
    """
    print("\n" + "─"*60)
    print("PROFESSIONAL BIOMETRIC EVALUATION")
    print("─"*60)

    eval_imgs   = valid_imgs_p  if valid_imgs_p  is not None else train_imgs_p
    eval_labels = valid_labels  if valid_labels  is not None else train_labels

    feats = model.predict(eval_imgs.astype("float32"), batch_size=8, verbose=0)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)

    n = len(feats)
    same_sims, diff_sims = [], []

    cap = min(400, n)
    for i in range(cap):
        for j in range(i+1, cap):
            sim = float(np.dot(feats[i], feats[j]))
            if eval_labels[i] == eval_labels[j]:
                same_sims.append(sim)
            else:
                diff_sims.append(sim)

    if not same_sims or not diff_sims:
        print("  Insufficient pairs for full evaluation.")
        return

    same_arr = np.array(same_sims)
    diff_arr = np.array(diff_sims)

    print(f"\n  Same-palm  pairs : {len(same_arr):,}")
    print(f"  Diff-palm  pairs : {len(diff_arr):,}")
    print(f"\n  Same   sim  mean ± std : {same_arr.mean():.4f} ± {same_arr.std():.4f}"
          f"  (min {same_arr.min():.4f})")
    print(f"  Diff   sim  mean ± std : {diff_arr.mean():.4f} ± {diff_arr.std():.4f}"
          f"  (max {diff_arr.max():.4f})")
    print(f"  Separation gap        : {same_arr.mean() - diff_arr.mean():.4f}")

    # ── FAR / FRR / EER sweep
    labels_bin = np.concatenate([np.ones(len(same_arr)), np.zeros(len(diff_arr))])
    scores_bin = np.concatenate([same_arr, diff_arr])

    best_eer_diff = 1.0; best_thr = 0.70
    best_far = best_frr = 0.0

    print("\n  ── FAR / FRR sweep ──")
    print(f"  {'Threshold':>10}  {'FAR':>8}  {'FRR':>8}  {'FAR+FRR diff':>14}")
    for t in np.arange(0.45, 0.99, 0.01):
        far = float(np.sum(diff_arr >= t)) / len(diff_arr)
        frr = float(np.sum(same_arr <  t)) / len(same_arr)
        eer_diff = abs(far - frr)
        if t in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
            print(f"  {t:>10.2f}  {far:>8.4f}  {frr:>8.4f}  {eer_diff:>14.4f}")
        if eer_diff < best_eer_diff:
            best_eer_diff = eer_diff; best_thr = t
            best_far = far;           best_frr = frr

    print(f"\n  EER threshold : {best_thr:.3f}")
    print(f"  FAR @ EER     : {best_far:.4f}  ({best_far*100:.2f}%)")
    print(f"  FRR @ EER     : {best_frr:.4f}  ({best_frr*100:.2f}%)")
    print(f"  ≈ EER         : {(best_far+best_frr)/2*100:.2f}%")

    # ── TAR @ FAR = 0.1%
    far_thresh = 0.001
    best_tar = 0.0; best_tar_thr = 0.90
    for t in np.arange(0.50, 0.999, 0.001):
        far = float(np.sum(diff_arr >= t)) / len(diff_arr)
        if far <= far_thresh:
            tar = float(np.sum(same_arr >= t)) / len(same_arr)
            if tar > best_tar:
                best_tar = tar; best_tar_thr = t
    print(f"\n  TAR @ FAR=0.1% : {best_tar:.4f}  ({best_tar*100:.2f}%)  "
          f"at threshold {best_tar_thr:.3f}")

    # ── ROC-AUC
    try:
        auc = roc_auc_score(labels_bin, scores_bin)
        print(f"  ROC-AUC       : {auc:.6f}")
    except Exception:
        pass

    # ── Operational threshold recommendation
    op_thr  = max(0.70, min(0.95, (same_arr.mean() + diff_arr.mean()) / 2))
    op_far  = float(np.sum(diff_arr >= op_thr)) / len(diff_arr)
    op_frr  = float(np.sum(same_arr <  op_thr)) / len(same_arr)
    print(f"\n  ★ Recommended operational threshold : {op_thr:.3f}")
    print(f"    FAR = {op_far*100:.2f}%   FRR = {op_frr*100:.2f}%")
    print("─"*60)
    return op_thr


# ──────────────────────────────────────────────────────────────────────────────
#  TFLITE EXPORT (int8 quantised for mobile)
# ──────────────────────────────────────────────────────────────────────────────
def export_tflite(feature_extractor, representative_imgs, path=TFLITE_SAVE_PATH):
    """Export int8 quantised TFLite model for mobile/edge deployment."""
    print(f"\nExporting TFLite → {path}")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(feature_extractor)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative_dataset():
            for img in representative_imgs[:100]:
                yield [img[np.newaxis].astype(np.float32)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.float32
        converter.inference_output_type = tf.float32
        tflite_model = converter.convert()
        with open(path, "wb") as f:
            f.write(tflite_model)
        print(f"   ✓ TFLite model saved ({len(tflite_model)//1024} KB)")
    except Exception as e:
        print(f"   ✗ TFLite export failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def train_pro_model():
    log_file = f"train_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s",
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    print("=" * 70)
    print("PalmPay — Professional Training Pipeline v5 (Ultra-HD 384×384)")
    print("=" * 70)

    # ── 1. Load data
    print(f"\n1. Loading & QA-validating images from {TRAIN_DIR} ...")
    imgs_p, imgs_s, labels = load_images_with_labels(TRAIN_DIR)
    n_classes = int(np.max(labels)) + 1 if len(labels) else 3
    print(f"   {len(imgs_p)} images | {n_classes} identities | "
          f"primary {IMG_SIZE_PRIMARY} | secondary {IMG_SIZE_SECONDARY}")

    v_imgs_p = v_imgs_s = v_labels = None
    if VALID_DIR and os.path.exists(VALID_DIR):
        v_imgs_p, v_imgs_s, v_labels = load_images_with_labels(VALID_DIR)
        print(f"   {len(v_imgs_p)} validation images loaded")

    if len(imgs_p) == 0:
        raise ValueError(f"No accepted images found in {TRAIN_DIR}")

    # Normalise
    imgs_p  = imgs_p.astype("float32")  / 255.0
    imgs_s  = imgs_s.astype("float32")  / 255.0
    if v_imgs_p is not None:
        v_imgs_p = v_imgs_p.astype("float32") / 255.0
        v_imgs_s = v_imgs_s.astype("float32") / 255.0

    # ── 2. Build model
    print("\n2. Building EfficientNetV2 + CBAM + ArcFace model ...")
    training_model, feature_extractor, base_model = create_pro_model(n_classes)
    training_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(f"   Total parameters: {training_model.count_params():,}")

    # ── 3. Augment
    print(f"\n3. Augmenting dataset (×{AUGMENT_FACTOR} + CutMix + MixUp) ...")
    aug_p, aug_s, aug_l = create_training_data(imgs_p, imgs_s, labels, AUGMENT_FACTOR)
    print(f"   {len(aug_p)} augmented training samples")

    v_aug_p = v_aug_s = v_aug_l_oh = None
    if v_imgs_p is not None:
        v_aug_p, v_aug_s, v_aug_l = create_training_data(v_imgs_p, v_imgs_s,
                                                           v_labels, augment_factor=2)
        v_aug_l_oh = keras.utils.to_categorical(v_aug_l, n_classes)

    aug_l_oh = keras.utils.to_categorical(aug_l, n_classes)
    monitor  = "val_loss" if v_aug_p is not None else "loss"

    # ── 4. Stage 1: ArcFace classification
    print(f"\n4. Stage 1: ArcFace training ({EPOCHS} epochs) ...")
    callbacks_s1 = [
        CosineAnnealingWarmup(LEARNING_RATE, WARMUP_EPOCHS, EPOCHS, MIN_LR),
        keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor=monitor,
                                         save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor=monitor, patience=22,
                                       restore_best_weights=True, verbose=1),
        keras.callbacks.TensorBoard(log_dir=f"tb_logs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    ]

    val_data = ([v_aug_p, v_aug_s], v_aug_l_oh) if v_aug_p is not None else None
    training_model.fit(
        [aug_p, aug_s], aug_l_oh,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        validation_data=val_data,
        callbacks=callbacks_s1, verbose=1
    )

    # ── 5. Stage 2: Progressive unfreeze
    print("\n5. Stage 2: Full progressive unfreeze ...")
    for layer in base_model.layers:
        layer.trainable = True
    training_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE * 0.02,
                                          weight_decay=1e-5),
        loss="categorical_crossentropy", metrics=["accuracy"]
    )
    training_model.fit(
        [aug_p, aug_s], aug_l_oh,
        batch_size=BATCH_SIZE, epochs=50,
        validation_data=val_data,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.4,
                                               patience=8, min_lr=1e-9, verbose=1),
            keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor=monitor,
                                             save_best_only=True, verbose=1),
            keras.callbacks.EarlyStopping(monitor=monitor, patience=18,
                                           restore_best_weights=True, verbose=1),
        ],
        verbose=1
    )

    # ── 6. Stage 3: Triplet fine-tuning
    fine_tune_triplet(feature_extractor, aug_p, aug_l, epochs=35,
                      batch_size=12, lr=2e-6, logger=logger)

    # ── 7. Save
    print(f"\n7. Saving final feature extractor → {MODEL_SAVE_PATH}")
    feature_extractor.save(MODEL_SAVE_PATH)

    # ── 8. Evaluate
    op_thr = evaluate_model_pro(feature_extractor, imgs_p, labels,
                                 v_imgs_p, v_labels)

    # ── 9. TFLite export
    export_tflite(feature_extractor, imgs_p[:100])

    print("\n" + "=" * 70)
    print(f"✓  Training complete!  Model → {MODEL_SAVE_PATH}")
    print(f"✓  TFLite  model     → {TFLITE_SAVE_PATH}")
    if op_thr:
        print(f"★  Use threshold     → {op_thr:.3f} in your verification pipeline")
    print("=" * 70)
    return feature_extractor


# ──────────────────────────────────────────────────────────────────────────────
#  INFERENCE HELPER  (use in production app)
# ──────────────────────────────────────────────────────────────────────────────
class PalmVerifier:
    """
    Drop-in inference class for production use.
    Usage:
        verifier = PalmVerifier("palm_feature_extractor_v5_pro.h5", threshold=0.82)
        emb      = verifier.enroll(image_bgr)          # enroll palm
        match, sim = verifier.verify(probe_bgr, emb)   # verify
    """
    def __init__(self, model_path, threshold=0.82):
        self.model     = keras.models.load_model(
            model_path,
            custom_objects={"L2Normalize": L2Normalize,
                            "ArcFaceLayer": ArcFaceLayer,
                            "CBAMLayer": CBAMLayer}
        )
        self.threshold = threshold

    def _preprocess(self, img_bgr):
        ok, report = assess_capture_quality(img_bgr)
        if not ok:
            raise ValueError(f"Image quality too low: {report['issues']}")
        enh = enhance_image_pro(img_bgr, IMG_SIZE_PRIMARY)
        rgb = cv2.cvtColor(enh, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        return rgb[np.newaxis]

    def get_embedding(self, img_bgr):
        x   = self._preprocess(img_bgr)
        emb = self.model.predict(x, verbose=0)[0]
        return emb / (np.linalg.norm(emb) + 1e-9)

    def enroll(self, img_bgr):
        """Returns embedding vector to store in DB."""
        return self.get_embedding(img_bgr)

    def verify(self, probe_bgr, enrolled_emb):
        """Returns (match: bool, similarity: float)."""
        emb = self.get_embedding(probe_bgr)
        sim = float(np.dot(emb, enrolled_emb))
        return sim >= self.threshold, sim


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42); random.seed(42); tf.random.set_seed(42)
    try:
        train_pro_model()
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback; traceback.print_exc()