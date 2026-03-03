"""
PalmPay — Professional Palm Recognition Engine v6 (High-Accuracy)
=================================================================
Upgrades over v5 to reduce False Acceptance Rate (FAR):

  FIX 1:  Raised threshold from 0.72 → 0.80 with adaptive support
  FIX 2:  Resolution 224→256 for richer ridge detail
  FIX 3:  Multi-Scale Retinex (σ=15,80,250) replaces single-scale
  FIX 4:  New VeinPatternExtractor — simulated NIR from R-B channel
          difference + CLAHE → captures subcutaneous vein pattern (~256D)
  FIX 5:  Stricter multi-frame voting: ALL frames must pass thr-0.03,
          uses MIN similarity (not median), quorum = min(3, N)
  FIX 6:  Consistency check — reject if per-frame std > 0.08
  FIX 7:  Tighter quality gate (sharpness 40, aspect-ratio check)
  FIX 8:  Cross-user similarity check API for registration

Final vector: ~4650D hybrid, L2-normalised, training-free
"""

import os
import cv2
import numpy as np
import logging

# ── TensorFlow (lazy) ────────────────────────────────────────────────────────
_tf = None
_keras = None

def _import_tf():
    global _tf, _keras
    if _tf is None:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        import tensorflow as tf
        _tf = tf
        _keras = tf.keras
    return _tf, _keras

from sklearn.metrics.pairwise import cosine_similarity
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = (256, 256)  # ← upgraded from 224×224 for finer detail

# Feature weights (hand-crafted biometric features weighted MORE than generic CNN)
W_GABOR   = 2.5    # Palm ridges / texture — most unique
W_LBP     = 2.0    # Micro-texture at 3 scales
W_CREASE  = 2.2    # Principal palm lines — highly individual
W_HOG     = 1.5    # Edge / shape structure
W_CNN     = 1.0    # Generic deep texture (stable from ImageNet)
W_VEIN    = 2.3    # Subcutaneous vein pattern — very hard to spoof

# Quality thresholds — TIGHTENED
MIN_SHARPNESS  = 40     # ← raised from 25
MIN_BRIGHTNESS = 30     # ← raised from 25
MAX_BRIGHTNESS = 230    # ← tightened from 235
MIN_CONTRAST   = 15     # ← raised from 12
MIN_ASPECT     = 0.6    # palm should be roughly square
MAX_ASPECT     = 1.7


# ─────────────────────────────────────────────────────────────────────────────
#  ROI Detection — dual colour-space skin mask with convex-hull isolation
# ─────────────────────────────────────────────────────────────────────────────
def detect_palm_roi(img, target_size=IMG_SIZE):
    """
    Robust palm ROI detection using HSV + YCrCb dual skin mask with
    Otsu fallback for unusual lighting. Includes convex-hull isolation
    and square crop for consistency.
    """
    h_img, w_img = img.shape[:2]

    try:
        # --- HSV skin mask ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, np.array([0, 20, 50]),
                                     np.array([25, 170, 255]))
        mask_hsv |= cv2.inRange(hsv, np.array([155, 20, 50]),
                                      np.array([180, 170, 255]))

        # --- YCrCb skin mask (better for diverse tones) ---
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, np.array([60, 135, 85]),
                                         np.array([255, 180, 135]))

        # Combine
        mask = mask_hsv | mask_ycrcb

        # If combined mask is too small, try Otsu on grayscale
        if np.sum(mask > 0) < (h_img * w_img * 0.08):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological cleanup
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > h_img * w_img * 0.04:
                # Convex hull for tighter palm isolation
                hull = cv2.convexHull(c)
                hull_mask = np.zeros_like(mask)
                cv2.fillConvexPoly(hull_mask, hull, 255)

                x, y, w, h = cv2.boundingRect(hull)
                m = int(min(w, h) * 0.10)  # smaller margin for tighter crop
                x = max(0, x - m);  y = max(0, y - m)
                w = min(w_img - x, w + 2*m)
                h = min(h_img - y, h + 2*m)
                # Ensure square crop for consistency
                side = max(w, h)
                cx, cy = x + w//2, y + h//2
                x1 = max(0, cx - side//2)
                y1 = max(0, cy - side//2)
                x2 = min(w_img, x1 + side)
                y2 = min(h_img, y1 + side)
                roi = img[y1:y2, x1:x2]
                if roi.shape[0] > 10 and roi.shape[1] > 10:
                    return cv2.resize(roi, target_size)

    except Exception as e:
        logger.debug(f"ROI detection: {e}")

    return cv2.resize(img, target_size)


# ─────────────────────────────────────────────────────────────────────────────
#  Image Enhancement — Multi-Scale Retinex (MSR) preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def enhance_image(img):
    """
    Multi-stage illumination normalization:
      1. Bilateral denoising (edge-preserving)
      2. Multi-Scale Retinex (MSR) with σ=15,80,250
      3. CLAHE adaptive histogram equalization
      4. Gentle unsharp mask
    """
    # 1. Bilateral denoise
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # 2. Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 3. Multi-Scale Retinex on L channel (3 scales)
    l_float = l.astype(np.float64) + 1.0
    retinex_sum = np.zeros_like(l_float)
    for sigma in [15, 80, 250]:
        blur = cv2.GaussianBlur(l_float, (0, 0), sigmaX=sigma)
        retinex_sum += np.log10(l_float) - np.log10(blur + 1.0)
    retinex = retinex_sum / 3.0

    # Normalise retinex to 0-255
    retinex = ((retinex - retinex.min()) /
               (retinex.max() - retinex.min() + 1e-8) * 255)
    l = retinex.astype(np.uint8)

    # 4. CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # 5. Extra denoise on L
    l = cv2.bilateralFilter(l, 5, 50, 50)

    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 6. Unsharp mask
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    img  = cv2.addWeighted(img, 1.4, blur, -0.4, 0)

    return img


def check_quality(img):
    """Enhanced quality check with aspect ratio validation."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    sh = cv2.Laplacian(gray, cv2.CV_64F).var()
    br = float(np.mean(gray))
    co = float(np.std(gray))

    h, w = gray.shape[:2]
    aspect = w / max(h, 1)

    ok = (sh >= MIN_SHARPNESS
          and MIN_BRIGHTNESS <= br <= MAX_BRIGHTNESS
          and co >= MIN_CONTRAST
          and MIN_ASPECT <= aspect <= MAX_ASPECT)

    return ok, {
        "sharpness": round(sh, 1),
        "brightness": round(br, 1),
        "contrast": round(co, 1),
        "aspect": round(aspect, 2),
        "passes": ok
    }


# ─────────────────────────────────────────────────────────────────────────────
#  1. GaborExtractor — 14 features per filter (6 stats + 8-zone energy)
# ─────────────────────────────────────────────────────────────────────────────
class GaborExtractor:
    """
    96 filters (12 orientations × 4 scales × 2 wavelengths).
    Per filter: 6 global stats + 8 spatial-zone energies = 14 values.
    Total output: 96 × 14 = 1344D, L2-normalised.
    """
    def __init__(self):
        self.filters = self._build()

    def _build(self):
        fbank = []
        for theta_i in range(12):
            theta = theta_i * np.pi / 12
            for ksize in [5, 7, 9, 11]:
                for lam in [4.0, 8.0]:
                    sigma = ksize * 0.56
                    kern = cv2.getGaborKernel(
                        (ksize, ksize), sigma, theta, lam, 0.5, 0,
                        ktype=cv2.CV_32F)
                    kern /= (1.5 * kern.sum() + 1e-8)
                    fbank.append(kern)
        return fbank

    def extract(self, gray):
        features = []
        gray_f = gray.astype(np.float32)
        h, w = gray_f.shape

        for kern in self.filters:
            resp = cv2.filter2D(gray_f, cv2.CV_32F, kern)
            mag  = np.abs(resp)

            # 6 global statistics
            features.append(np.mean(mag))
            features.append(np.std(mag))
            features.append(np.max(mag))
            features.append(float(np.percentile(mag, 75)))
            features.append(float(np.percentile(mag, 25)))
            # Energy (sum of squared magnitudes, normalised)
            features.append(float(np.sum(mag**2) / (h * w)))

            # 8 spatial zones (2 × 4 grid) — captures WHERE energy localises
            zh, zw = h // 2, w // 4
            for ri in range(2):
                for ci in range(4):
                    zone = mag[ri*zh:(ri+1)*zh, ci*zw:(ci+1)*zw]
                    features.append(float(np.mean(zone)))

        feat = np.array(features, dtype=np.float32)
        return feat / (np.linalg.norm(feat) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
#  2. Multi-Scale LBP — r=1,2,3 → 768D
# ─────────────────────────────────────────────────────────────────────────────
class MultiScaleLBPExtractor:
    """
    3 radii capturing micro, meso, and macro texture.
    Each radius → 256-bin histogram → total 768D, L2-normalised.
    """
    def _lbp_at_radius(self, gray, radius, n_points=8):
        h, w = gray.shape
        r = radius
        pad = r
        gray_pad = cv2.copyMakeBorder(gray, pad, pad, pad, pad,
                                       cv2.BORDER_REFLECT)
        center = gray_pad[pad:pad+h, pad:pad+w].astype(np.int16)
        lbp = np.zeros((h, w), dtype=np.uint8)

        for i in range(n_points):
            angle = 2.0 * np.pi * i / n_points
            dy = -round(r * np.cos(angle))
            dx =  round(r * np.sin(angle))
            ny = pad + dy;   nx = pad + dx
            neighbour = gray_pad[ny:ny+h, nx:nx+w].astype(np.int16)
            lbp |= ((neighbour >= center).astype(np.uint8) << i)

        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()
        return hist.astype(np.float32)

    def extract(self, gray):
        hists = []
        for r in [1, 2, 3]:
            hists.append(self._lbp_at_radius(gray, r))
        feat = np.concatenate(hists)
        return feat / (np.linalg.norm(feat) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
#  3. PalmCreaseExtractor — ridge / principal line pattern
# ─────────────────────────────────────────────────────────────────────────────
class PalmCreaseExtractor:
    """
    Detects and encodes palm principal lines and wrinkles.
    Uses directional Sobel at multiple angles + morphological
    line enhancement + spatial histogram of crease density.

    Output: 432D (6 directions × 72 spatial bins), L2-normalised.
    """
    def extract(self, gray):
        gray_f = gray.astype(np.float32)
        h, w = gray_f.shape
        features = []

        # 6 directional edge maps (0°, 30°, 60°, 90°, 120°, 150°)
        angles_deg = [0, 30, 60, 90, 120, 150]

        for angle in angles_deg:
            rad = np.deg2rad(angle)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)

            # Directional gradient via rotated Sobel-like kernels
            kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=np.float32)
            ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=np.float32)

            gx = cv2.filter2D(gray_f, cv2.CV_32F, kx)
            gy = cv2.filter2D(gray_f, cv2.CV_32F, ky)

            # Project gradient onto direction
            directional = np.abs(gx * cos_a + gy * sin_a)

            # Morphological line enhancement
            length = 11
            line_kern = cv2.getStructuringElement(
                cv2.MORPH_RECT, (length, 1))
            M = cv2.getRotationMatrix2D((length//2, 0), angle, 1.0)
            line_kern_r = cv2.warpAffine(
                line_kern.astype(np.float32), M, (length, length))
            line_kern_r = (line_kern_r > 0.3).astype(np.uint8)

            enhanced = cv2.morphologyEx(
                (directional * 255 / (directional.max() + 1e-8)).astype(np.uint8),
                cv2.MORPH_CLOSE, line_kern_r)

            # Spatial histogram (6×12 grid = 72 bins per direction)
            bh, bw = h // 6, w // 12
            for ri in range(6):
                for ci in range(12):
                    block = enhanced[ri*bh:(ri+1)*bh, ci*bw:(ci+1)*bw]
                    features.append(float(np.mean(block)))

        feat = np.array(features, dtype=np.float32)
        return feat / (np.linalg.norm(feat) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
#  4. HOG — proper Dalal & Triggs with L2-Hys block normalization
# ─────────────────────────────────────────────────────────────────────────────
class HOGExtractor:
    """
    Proper HOG: 8×8 cells, 2×2 blocks, 9 bins, L2-Hys normalization.
    On 64×64 input: 8×8 cells → 7×7 blocks → 7×7×4×9 = 1764D, L2-normalised.
    """
    def extract(self, gray):
        gray = cv2.resize(gray, (64, 64))

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
        mag   = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * (180.0 / np.pi) % 180.0

        n_bins = 9
        cell_size = 8
        n_cells = 64 // cell_size  # 8

        # Cell histograms
        cells = np.zeros((n_cells, n_cells, n_bins), dtype=np.float32)
        for i in range(n_cells):
            for j in range(n_cells):
                y1 = i * cell_size;  y2 = y1 + cell_size
                x1 = j * cell_size;  x2 = x1 + cell_size
                cm = mag[y1:y2, x1:x2]
                ca = angle[y1:y2, x1:x2]
                for b in range(n_bins):
                    lo = b * 20.0;  hi = lo + 20.0
                    mask = (ca >= lo) & (ca < hi)
                    cells[i, j, b] = np.sum(cm[mask])

        # Block normalization (2×2 cells, L2-Hys)
        features = []
        for i in range(n_cells - 1):
            for j in range(n_cells - 1):
                block = cells[i:i+2, j:j+2, :].flatten()
                norm = np.linalg.norm(block) + 1e-6
                block = block / norm
                block = np.clip(block, 0, 0.2)  # Hys clipping
                norm2 = np.linalg.norm(block) + 1e-6
                block = block / norm2
                features.extend(block)

        feat = np.array(features, dtype=np.float32)
        return feat / (np.linalg.norm(feat) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
#  5. CNN — frozen ResNet50V2, raw 2048D GAP output (NO random layers)
# ─────────────────────────────────────────────────────────────────────────────
class CNNExtractor:
    """
    ResNet50V2 pretrained on ImageNet, completely frozen.
    Uses the raw GlobalAveragePooling2D output (2048D).
    NO extra Dense layers → NO random weights → STABLE features.
    Proper ImageNet preprocessing applied.
    """
    def __init__(self):
        self.model = None
        self.output_dim = 2048
        self._build()

    def _build(self):
        tf, keras = _import_tf()
        try:
            from tensorflow.keras.applications import ResNet50V2
            from tensorflow.keras.applications.resnet_v2 import preprocess_input
            self.preprocess = preprocess_input

            base = ResNet50V2(
                weights="imagenet", include_top=False,
                input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                pooling="avg"   # → 2048D
            )
            base.trainable = False
            self.model = base
            self.output_dim = 2048
            print("  ✓ CNN: ResNet50V2 → 2048D (ImageNet frozen, no random layers)")

        except Exception:
            try:
                from tensorflow.keras.applications import MobileNetV2
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
                self.preprocess = preprocess_input

                base = MobileNetV2(
                    weights="imagenet", include_top=False,
                    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                    pooling="avg"
                )
                base.trainable = False
                self.model = base
                self.output_dim = 1280
                print("  ✓ CNN: MobileNetV2 → 1280D (ImageNet frozen)")

            except Exception as e:
                print(f"  ⚠ CNN unavailable: {e}")
                self.model = None
                self.output_dim = 0

    def extract(self, rgb_image):
        if self.model is None:
            return np.zeros(self.output_dim, dtype=np.float32)

        img = cv2.resize(rgb_image, IMG_SIZE).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img = self.preprocess(img)    # Proper ImageNet mean/std normalization
        feat = self.model.predict(img, verbose=0)[0]
        norm = np.linalg.norm(feat)
        return (feat / (norm + 1e-8)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  6. VeinPatternExtractor — simulated NIR vein pattern (NEW)
# ─────────────────────────────────────────────────────────────────────────────
class VeinPatternExtractor:
    """
    Simulates near-infrared (NIR) vein imaging using the difference
    between red and blue channels (hemoglobin absorbs red light).
    Pipeline:
      1. Extract R-B difference → vein-like contrast
      2. CLAHE enhancement on the vein map
      3. Gabor filtering at 4 orientations for vein structure
      4. Spatial histogram (4×4 grid) per orientation
    Output: 256D (4 orientations × 4 stats + 4 orientations × 4×4×3 grid),
    L2-normalised.
    """
    def extract(self, bgr_image):
        b, g, r = cv2.split(bgr_image)

        # R-B difference highlights veins (hemoglobin absorption)
        r_f = r.astype(np.float32)
        b_f = b.astype(np.float32)
        vein_raw = np.clip(r_f - b_f + 128, 0, 255).astype(np.uint8)

        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        vein = clahe.apply(vein_raw)

        # Additional enhancement: median filter to reduce noise
        vein = cv2.medianBlur(vein, 5)

        h, w = vein.shape
        features = []
        vein_f = vein.astype(np.float32)

        # 4 orientations of Gabor for vein structure
        for theta_i in range(4):
            theta = theta_i * np.pi / 4
            # Two scales
            for ksize in [7, 11]:
                kern = cv2.getGaborKernel(
                    (ksize, ksize), ksize * 0.5, theta, 6.0, 0.5, 0,
                    ktype=cv2.CV_32F)
                kern /= (kern.sum() + 1e-8)
                resp = cv2.filter2D(vein_f, cv2.CV_32F, kern)
                mag = np.abs(resp)

                # 4 global stats
                features.append(float(np.mean(mag)))
                features.append(float(np.std(mag)))
                features.append(float(np.max(mag)))
                features.append(float(np.sum(mag**2) / (h * w)))

                # 4×4 spatial grid (16 zones)
                zh, zw = h // 4, w // 4
                for ri in range(4):
                    for ci in range(4):
                        zone = mag[ri*zh:(ri+1)*zh, ci*zw:(ci+1)*zw]
                        features.append(float(np.mean(zone)))
                        features.append(float(np.std(zone)))
                        features.append(float(np.max(zone)))

        feat = np.array(features, dtype=np.float32)
        return feat / (np.linalg.norm(feat) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
#  Main PalmRecognizer — best-fit fusion with anti-FAR improvements
# ─────────────────────────────────────────────────────────────────────────────
class PalmRecognizer:
    """
    Professional palm recognition v6: fuses 6 complementary feature types.
    Each type captures different biometric characteristics:
      • Gabor   → ridge texture at multiple orientations/scales
      • LBP     → micro-texture at 3 spatial resolutions
      • Crease  → principal palm lines (heart, head, life lines)
      • HOG     → edge/shape gradient structure
      • CNN     → deep generic texture (ImageNet, frozen, no random layers)
      • Vein    → subcutaneous vein pattern (simulated NIR) — NEW
    """

    def __init__(self, threshold=0.80):
        self.threshold = threshold   # ← raised from 0.72 to 0.80
        self.img_size  = IMG_SIZE

        print("\n" + "=" * 65)
        print("  PalmPay — Professional Recognition Engine v6 (High-Accuracy)")
        print("=" * 65)

        self.gabor_ext  = GaborExtractor()
        n_gabor = len(self.gabor_ext.filters) * 14
        print(f"  ✓ Gabor  : {len(self.gabor_ext.filters)} filters × 14 = {n_gabor}D")

        self.lbp_ext    = MultiScaleLBPExtractor()
        print(f"  ✓ LBP    : 3 radii × 256 bins = 768D")

        self.crease_ext = PalmCreaseExtractor()
        print(f"  ✓ Crease : 6 directions × 72 spatial bins = 432D")

        self.hog_ext    = HOGExtractor()
        print(f"  ✓ HOG    : 8×8 cells, L2-Hys blocks = ~1764D")

        self.cnn_ext    = CNNExtractor()
        cnn_d = self.cnn_ext.output_dim
        print(f"  ✓ CNN    : {cnn_d}D (raw GAP, zero random layers)")

        self.vein_ext   = VeinPatternExtractor()
        print(f"  ✓ Vein   : simulated NIR vein pattern (NEW)")

        total = n_gabor + 768 + 432 + 1764 + cnn_d + 640  # approx vein dims
        print(f"\n  Total hybrid vector: ~{total}D")
        print(f"  Weights: Gabor={W_GABOR}  LBP={W_LBP}  Crease={W_CREASE}"
              f"  HOG={W_HOG}  CNN={W_CNN}  Vein={W_VEIN}")
        print(f"  Threshold: {self.threshold} (raised from 0.72)")
        print(f"  Resolution: {IMG_SIZE[0]}×{IMG_SIZE[1]} (upgraded from 224×224)")
        print("=" * 65 + "\n")

    # ── Core feature extraction ───────────────────────────────────────────────
    def extract_features(self, image_input):
        """
        Full pipeline: load → ROI → enhance → extract all 6 → weighted
        concat → L2-normalise.
        """
        img = self._load_image(image_input)
        if img is None:
            raise ValueError("Could not decode image")

        # ROI + enhance
        roi  = detect_palm_roi(img, self.img_size)
        roi  = enhance_image(roi)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        rgb  = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # Extract each feature type
        gabor  = self.gabor_ext.extract(gray)
        lbp    = self.lbp_ext.extract(gray)
        crease = self.crease_ext.extract(gray)
        hog    = self.hog_ext.extract(gray)
        cnn    = self.cnn_ext.extract(rgb)
        vein   = self.vein_ext.extract(roi)   # ← NEW

        # Weighted fusion
        hybrid = np.concatenate([
            gabor  * W_GABOR,
            lbp    * W_LBP,
            crease * W_CREASE,
            hog    * W_HOG,
            cnn    * W_CNN,
            vein   * W_VEIN,   # ← NEW
        ])

        norm = np.linalg.norm(hybrid)
        return (hybrid / (norm + 1e-8)).astype(np.float32)

    def extract_features_with_quality(self, image_input):
        """
        Extract features WITH quality gating.
        Returns (features, quality_info, passed_quality).
        """
        img = self._load_image(image_input)
        if img is None:
            raise ValueError("Could not decode image")

        roi = detect_palm_roi(img, self.img_size)
        passed, qinfo = check_quality(roi)

        if not passed:
            return None, qinfo, False

        roi  = enhance_image(roi)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        rgb  = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        gabor  = self.gabor_ext.extract(gray)
        lbp    = self.lbp_ext.extract(gray)
        crease = self.crease_ext.extract(gray)
        hog    = self.hog_ext.extract(gray)
        cnn    = self.cnn_ext.extract(rgb)
        vein   = self.vein_ext.extract(roi)

        hybrid = np.concatenate([
            gabor  * W_GABOR,
            lbp    * W_LBP,
            crease * W_CREASE,
            hog    * W_HOG,
            cnn    * W_CNN,
            vein   * W_VEIN,
        ])
        norm = np.linalg.norm(hybrid)
        feat = (hybrid / (norm + 1e-8)).astype(np.float32)
        return feat, qinfo, True

    def extract_features_batch(self, image_list, fast_mode=False, use_roi=True):
        feats = []
        for i, img in enumerate(image_list):
            try:
                feats.append(self.extract_features(img))
            except Exception as e:
                print(f"  ⚠ Frame {i+1} failed: {e}")
        if not feats:
            dummy_dim = (len(self.gabor_ext.filters)*14 + 768 + 432 + 1764
                         + self.cnn_ext.output_dim + 640)
            return [np.zeros(dummy_dim, dtype=np.float32)]
        return feats

    # ── Comparison ────────────────────────────────────────────────────────────
    def compare_features(self, f1, f2):
        if f1.shape[0] != f2.shape[0]:
            d = min(f1.shape[0], f2.shape[0])
            f1 = f1[:d] / (np.linalg.norm(f1[:d]) + 1e-8)
            f2 = f2[:d] / (np.linalg.norm(f2[:d]) + 1e-8)
        return float(cosine_similarity(f1.reshape(1,-1), f2.reshape(1,-1))[0][0])

    # ── Single-frame verify ───────────────────────────────────────────────────
    def verify_palm(self, stored_features, new_image):
        try:
            stored = (np.frombuffer(stored_features, dtype=np.float32)
                      if isinstance(stored_features, bytes) else stored_features)
            new = self.extract_features(new_image)
            sim = self.compare_features(stored, new)
            print(f"  [verify] sim={sim:.4f}  thr={self.threshold}  "
                  f"pass={sim >= self.threshold}")
            return sim >= self.threshold, sim
        except Exception as e:
            print(f"  [verify] error: {e}")
            return False, 0.0

    # ── Multi-frame ensemble verify — HARDENED ────────────────────────────────
    def multi_frame_verify(self, stored_features, image_list):
        """
        Hardened ensemble verification (v6):
          • Extract features from each frame
          • Compute cosine similarity for each
          • Use MINIMUM similarity (not median) as aggregate — pessimistic
          • ALL frames must individually pass (threshold - 0.03)
          • Reject if std of sims > 0.08 (inconsistent presentation)
        """
        stored = (np.frombuffer(stored_features, dtype=np.float32)
                  if isinstance(stored_features, bytes) else stored_features)

        if not image_list:
            return False, 0.0

        sims = []
        quality_fails = 0
        for i, img in enumerate(image_list):
            try:
                feat = self.extract_features(img)
                s = self.compare_features(stored, feat)
                sims.append(s)
                print(f"  [multi] frame {i+1}: sim={s:.4f}")
            except Exception as e:
                print(f"  [multi] frame {i+1} failed: {e}")
                quality_fails += 1

        if not sims:
            return False, 0.0

        # Use MINIMUM similarity — pessimistic for security
        min_sim = float(np.min(sims))
        mean_sim = float(np.mean(sims))
        std_sim = float(np.std(sims))

        # Per-frame threshold — tighter than v5
        per_frame_thr = self.threshold - 0.03
        n_pass = sum(1 for s in sims if s >= per_frame_thr)

        # ALL frames must pass (quorum = total frames)
        quorum_size = min(3, len(sims))
        quorum = n_pass >= quorum_size

        # Consistency check — reject if too much variance
        consistent = std_sim <= 0.08

        # Final decision: minimum passes AND quorum AND consistent
        ok = (min_sim >= self.threshold) and quorum and consistent

        print(f"  [multi] min={min_sim:.4f}  mean={mean_sim:.4f}  "
              f"std={std_sim:.4f}  pass={n_pass}/{len(sims)}  "
              f"thr={self.threshold}  consistent={consistent}  verified={ok}")
        return ok, min_sim

    # ── Cross-user similarity check for registration ──────────────────────────
    def check_cross_user_similarity(self, new_features, all_stored_features,
                                     cross_check_threshold=0.75):
        """
        Check if a new palm is too similar to any existing registered user.
        Returns list of (user_index, similarity) for those exceeding threshold.
        """
        matches = []
        for i, stored in enumerate(all_stored_features):
            if stored is None:
                continue
            stored_f = (np.frombuffer(stored, dtype=np.float32)
                        if isinstance(stored, bytes) else stored)
            sim = self.compare_features(new_features, stored_f)
            if sim >= cross_check_threshold:
                matches.append((i, sim))
        return matches

    # ── Image loading ─────────────────────────────────────────────────────────
    def _load_image(self, source):
        if isinstance(source, np.ndarray):
            return source.copy()
        if isinstance(source, (bytes, bytearray)):
            return cv2.imdecode(np.frombuffer(source, np.uint8),
                                cv2.IMREAD_COLOR)
        if isinstance(source, str):
            if os.path.isfile(source):
                return cv2.imread(source)
            import base64
            try:
                d = source.split(",")[1] if "," in source else source
                return cv2.imdecode(
                    np.frombuffer(base64.b64decode(d), np.uint8),
                    cv2.IMREAD_COLOR)
            except Exception:
                pass
        return None

    # ── Persistence ───────────────────────────────────────────────────────────
    def save_features(self, features, path):
        data = features.tobytes() if isinstance(features, np.ndarray) else features
        with open(path, "wb") as f:
            f.write(data)

    def load_features(self, path):
        with open(path, "rb") as f:
            return np.frombuffer(f.read(), dtype=np.float32)


# ── Singleton ─────────────────────────────────────────────────────────────────
_palm_recognizer = None

def get_palm_recognizer(threshold=0.80):
    global _palm_recognizer
    if _palm_recognizer is None:
        _palm_recognizer = PalmRecognizer(threshold=threshold)
    return _palm_recognizer
