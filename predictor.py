"""
predictor.py — BIM Sign Recognition Inference Core
====================================================
Encapsulates the full MediaPipe → EfficientNet pipeline so that
the same logic can be called from:
  • a local OpenCV demo  (demo_local.py)
  • a Flask API endpoint  (/api/predict)
  • any future runner (FastAPI, Celery worker, etc.)

Usage
-----
from predictor import BIMPredictor

predictor = BIMPredictor("efficientnet_b0_bim_static_model_finetuned.keras")

# Single frame (numpy BGR array from OpenCV or decoded JPEG bytes):
result = predictor.predict_frame(bgr_frame)
# → {"label": "makan", "confidence": 0.82, "status": "ok", "bbox": [x1,y1,x2,y2]}
# → {"label": None, "confidence": 0.0,  "status": "no_hand"}
# → {"label": None, "confidence": 0.42,  "status": "low_confidence"}
"""

import os
import collections
import urllib.request
import cv2
import numpy as np



# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_LABELS = ["air", "demam", "dengar", "makan", "minum",
                "salah", "saya", "senyap", "tidur", "waktu"]

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "efficientnet_b0_bim_static_model_finetuned.keras")

MODEL_INPUT_SIZE      = (224, 224)
CONFIDENCE_THRESHOLD  = 0.60
PADDING_RATIO         = 0.10
SMOOTHING_WINDOW      = 7      # majority-vote history length


# ── Helper ─────────────────────────────────────────────────────────────────────
def _combined_bbox(hands, img_w, img_h, padding=PADDING_RATIO):
    """Return a single padded bounding box covering all detected hand landmarks."""
    if not hands:
        return None
    xs, ys = [], []
    for hand in hands:
        for lm in hand:
            xs.append(lm.x)
            ys.append(lm.y)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    px = (xmax - xmin) * padding
    py = (ymax - ymin) * padding
    x1 = int(max(0,     (xmin - px) * img_w))
    y1 = int(max(0,     (ymin - py) * img_h))
    x2 = int(min(img_w, (xmax + px) * img_w))
    y2 = int(min(img_h, (ymax + py) * img_h))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


# ── Main class ─────────────────────────────────────────────────────────────────
class BIMPredictor:
    """
    Stateful predictor.  Maintains a smoothing history buffer across calls so
    that majority-vote works correctly in a streaming (frame-by-frame) context.

    For a stateless single-frame prediction (e.g. one REST request per image),
    call predict_frame() with smooth=False, or instantiate a fresh object per
    request (cheap — model loading is done once at startup via a module-level
    singleton, see get_predictor() below).
    """
    

    def __init__(self, model_path: str, hand_model_path: str = HAND_MODEL_PATH):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import tensorflow as tf
        # ── Download MediaPipe hand model if needed ────────────────────────────
        if not os.path.exists(hand_model_path):
            print(f"[BIMPredictor] Downloading hand landmarker model …")
            urllib.request.urlretrieve(HAND_MODEL_URL, hand_model_path)

        # ── Load Keras model ───────────────────────────────────────────────────
        print(f"[BIMPredictor] Loading {model_path} …")
        self.model = tf.keras.models.load_model(model_path)
        self.class_labels = CLASS_LABELS

        # ── Build MediaPipe landmarker (IMAGE mode for single frames) ──────────
        base_opts = mp_python.BaseOptions(model_asset_path=hand_model_path)
        lm_opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(lm_opts)

        # ── Smoothing state (only used when smooth=True) ───────────────────────
        self._history: collections.deque = collections.deque(maxlen=SMOOTHING_WINDOW)

    def reset_history(self):
        """Call this when switching target word or starting a new attempt."""
        self._history.clear()

    def predict_frame(self, bgr_frame: np.ndarray, smooth: bool = True) -> dict:
        """
        Run the full pipeline on one BGR frame.

        Parameters
        ----------
        bgr_frame : np.ndarray  — OpenCV-style BGR image
        smooth    : bool        — whether to apply majority-vote smoothing

        Returns
        -------
        dict with keys:
            status      : "ok" | "no_hand" | "low_confidence"
            label       : str | None
            confidence  : float
            all_scores  : dict[label → float]  (always present)
            bbox        : [x1, y1, x2, y2] | None
        """
        import mediapipe as mp
        from keras.applications.efficientnet import preprocess_input
        
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect(mp_img)
        hands  = result.hand_landmarks or []

        bbox = _combined_bbox(hands, w, h, padding=PADDING_RATIO)

        if bbox is None:
            self._history.clear()
            return {"status": "no_hand", "label": None, "confidence": 0.0,
                    "all_scores": {}, "bbox": None}

        x1, y1, x2, y2 = bbox
        crop = bgr_frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized  = cv2.resize(crop_rgb, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        batch    = preprocess_input(np.expand_dims(resized.astype(np.float32), 0))

        preds    = self.model.predict(batch, verbose=0)[0]
        top_idx  = int(np.argmax(preds))
        top_conf = float(preds[top_idx])

        all_scores = {lbl: round(float(p), 4)
                      for lbl, p in zip(self.class_labels, preds)}

        if top_conf < CONFIDENCE_THRESHOLD:
            self._history.clear()
            return {"status": "low_confidence", "label": None,
                    "confidence": top_conf, "all_scores": all_scores,
                    "bbox": list(bbox)}

        # Confident prediction — optionally smooth
        if smooth:
            self._history.append(top_idx)
            smoothed_idx = max(set(self._history), key=self._history.count)
        else:
            smoothed_idx = top_idx

        return {
            "status":     "ok",
            "label":      self.class_labels[smoothed_idx],
            "confidence": round(float(preds[smoothed_idx]), 4),
            "all_scores": all_scores,
            "bbox":       list(bbox),
        }

    def predict_jpeg_bytes(self, jpeg_bytes: bytes, smooth: bool = False) -> dict:
        """
        Convenience wrapper for the web API path: decode JPEG bytes → predict.
        smooth=False by default because each HTTP request is independent; the
        client-side JS manages its own smoothing window if needed.
        """
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"status": "decode_error", "label": None,
                    "confidence": 0.0, "all_scores": {}, "bbox": None}
        return self.predict_frame(frame, smooth=smooth)


# ── Module-level singleton (imported once per process) ─────────────────────────
_predictor_instance: BIMPredictor | None = None

def get_predictor(model_path: str = MODEL_PATH) -> BIMPredictor:
    """
    Return the shared predictor instance, creating it on first call.
    Use this in Flask/Gunicorn so the model is loaded once per worker process.
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = BIMPredictor(model_path)
    return _predictor_instance
