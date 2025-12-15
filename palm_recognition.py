"""
Palm Recognition Module
SIFT/ORB fallback, safe image handling and diagnostics.
"""
try:
    import cv2
    _cv2_import_error = None
except Exception as e:
    cv2 = None
    _cv2_import_error = e

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class PalmRecognizer:
    def __init__(self):
        if cv2 is None:
            raise ImportError(f"OpenCV import failed: {_cv2_import_error}")
        try:
            self.sift = cv2.SIFT_create()
            self.use_sift = True
        except Exception:
            self.sift = None
            self.orb = cv2.ORB_create(1000)
            self.use_sift = False

    def _enhance_image(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        l = cv2.bilateralFilter(l, 9, 75, 75)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        l = cv2.filter2D(l, -1, kernel)
        img = cv2.merge([l, a, b])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
        img = cv2.addWeighted(img, 1.3, blurred, -0.3, 0)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _extract_hand_roi(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = int(0.12 * max(w, h))
        x, y = max(0, x - pad), max(0, y - pad)
        w, h = min(img.shape[1] - x, w + 2 * pad), min(img.shape[0] - y, h + 2 * pad)
        roi = img[y:y+h, x:x+w]
        return roi if roi.size > 0 else img

    def extract_features(self, image_or_bytes):
        """Accept bytes or ndarray -> fixed-size float32 vector"""
        if isinstance(image_or_bytes, (bytes, bytearray)):
            arr = np.frombuffer(image_or_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        elif isinstance(image_or_bytes, np.ndarray):
            img = image_or_bytes.copy()
        else:
            raise ValueError("Unsupported image input type")
        if img is None:
            raise ValueError("Failed to decode image")

        h, w = img.shape[:2]
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

        img = self._enhance_image(img)
        roi = self._extract_hand_roi(img)
        roi = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if self.use_sift:
            kp, des = self.sift.detectAndCompute(gray, None)
        else:
            kp, des = self.orb.detectAndCompute(gray, None)

        if des is None or len(des) == 0:
            raise ValueError("No descriptors found (try better lighting / steadier capture)")

        sift_feat = des.flatten().astype(np.float32)
        sift_feat = sift_feat[:min(2048, len(sift_feat))]
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().astype(np.float32)
        hist /= (hist.sum() + 1e-6)
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256]).flatten().astype(np.float32)
        edge_hist /= (edge_hist.sum() + 1e-6)

        combined = np.concatenate([sift_feat, hist, edge_hist])
        target_size = 2500
        if len(combined) < target_size:
            combined = np.pad(combined, (0, target_size - len(combined)), mode='constant')
        else:
            combined = combined[:target_size]
        return combined.astype(np.float32)

    def verify_palm(self, stored_features, new_image_bytes, threshold=0.72):
        try:
            new_feat = self.extract_features(new_image_bytes)
        except Exception as e:
            return False, 0.0, f"Feature extraction failed: {e}"

        if isinstance(stored_features, (bytes, bytearray)):
            try:
                stored = np.frombuffer(stored_features, dtype=np.float32)
            except Exception:
                stored = np.array([], dtype=np.float32)
        else:
            stored = np.array(stored_features, dtype=np.float32).flatten()

        if stored.size == 0:
            return False, 0.0, "Stored features invalid"

        L = max(len(stored), len(new_feat))
        s = np.pad(stored, (0, L - len(stored)), mode='constant')
        n = np.pad(new_feat, (0, L - len(new_feat)), mode='constant')

        s = (s - s.mean()) / (s.std() + 1e-8)
        n = (n - n.mean()) / (n.std() + 1e-8)

        euclidean_dist = float(np.linalg.norm(s - n))
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        cosine_sim = float(cosine_similarity(s.reshape(1, -1), n.reshape(1, -1))[0, 0])
        cosine_sim = max(0.0, min(1.0, (cosine_sim + 1) / 2))

        final_sim = 0.55 * euclidean_sim + 0.45 * cosine_sim
        is_match = final_sim >= threshold
        return bool(is_match), float(final_sim), ""

    def debug_summary(self, image_or_bytes):
        try:
            if isinstance(image_or_bytes, (bytes, bytearray)):
                arr = np.frombuffer(image_or_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            elif isinstance(image_or_bytes, np.ndarray):
                img = image_or_bytes.copy()
            else:
                return {"ok": False, "error": "Unsupported input type"}
            if img is None:
                return {"ok": False, "error": "Failed to decode image"}
            h, w = img.shape[:2]
            if max(h, w) > 512:
                scale = 512 / max(h, w)
                img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
            img_e = self._enhance_image(img)
            roi = self._extract_hand_roi(img_e)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            kp_count = 0
            des_count = 0
            try:
                if self.use_sift:
                    kp, des = self.sift.detectAndCompute(gray, None)
                else:
                    kp, des = self.orb.detectAndCompute(gray, None)
                kp_count = 0 if kp is None else len(kp)
                des_count = 0 if des is None else des.shape[0]
            except Exception:
                pass
            return {"ok": True, "image_shape": (h, w), "roi_shape": roi.shape, "kp_count": kp_count, "des_count": des_count, "using": "SIFT" if self.use_sift else "ORB"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

def get_palm_recognizer():
    return PalmRecognizer()

# NOTE: Removed Flask routes and any code that depends on `app`, `request`, templates or DB.
# Keep web routes in app.py to avoid circular imports and startup failures.