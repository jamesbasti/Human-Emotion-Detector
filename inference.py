"""
Inference pipeline: ONNX person detector (YOLO) -> ONNX emotion classifier.

app.py and utils.py only use `run_inference()` and the `Person` / `BoundingBox`
shapes below.

Models (in the `models/` folder):
  - detector.onnx               YOLO26n, 1 class ("person"), 640x640 input,
                                raw output [1, 5, 8400] = (cx, cy, w, h, score)
  - emotic_attribute_model_1.onnx ResNet18, 224x224 person crop ->
                                emotions (26 logits), continuous (VAD),
                                age (3 logits), gender (2 logits)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from emotion_categories import EMOTIC_CATEGORIES, CATEGORY_TO_GROUP, VAD_DIMENSIONS

# CONFIG{

MODELS_DIR = Path(__file__).parent / "models"
DETECTOR_PATH = MODELS_DIR / "detector.onnx"
CLASSIFIER_CANDIDATES = [
    MODELS_DIR / "emotic_attribute_model.onnx",  
]
# Per-category training prevalence, produced by tools/compute_priors.py from
# the same training CSV the classifier was trained on. When present, scores
# are calibrated against it (see _calibrate) before ranking, the same idea
# as calibrated_generalized_scores() in the training notebook -- it stops
# whatever category is most common in training (Engagement, here) from
# winning by default regardless of what's actually in the image. When
# absent, the app just uses raw sigmoid scores like it always has.
CALIBRATION_PATH = Path(__file__).parent / "calibration.json"

# Input sizes are read from the models themselves (see _input_size), so
# re-exporting at a different resolution needs no code change.
DETECTION_CONF = 0.25        # min person confidence to keep a box
NMS_IOU = 0.45               # overlap threshold for non-max suppression
TOP_K = 3                    # how many top emotion categories to report per person

# ASSUMPTION: classifier was trained with ImageNet normalization.
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ASSUMPTION: EMOTIC attribute order. Only used for the optional fields.
AGE_CLASSES = ["Kid", "Teenager", "Adult"]
GENDER_CLASSES = ["Female", "Male"]

# }CONFIG


@lru_cache(maxsize=1)
def _load_calibration() -> dict | None:
    """{category: training prevalence in (0,1)}, or None if no calibration.json."""
    if not CALIBRATION_PATH.exists():
        return None
    priors = json.loads(CALIBRATION_PATH.read_text())
    missing = [c for c in EMOTIC_CATEGORIES if c not in priors]
    if missing:
        raise InferenceError(f"calibration.json is missing categories: {missing}")
    return priors


def _calibrate(scores: dict) -> dict | None:
    """
    Log-odds calibration: subtract each category's training prevalence
    (in log-odds space) from its raw sigmoid score, so a category that's
    just generically common doesn't automatically rank first. Returns
    None (meaning "not calibrated") if no calibration.json is present.
    """
    priors = _load_calibration()
    if priors is None:
        return None
    eps = 1e-4
    calibrated = {}
    for c, p in scores.items():
        p = min(max(p, eps), 1 - eps)
        prior = priors[c]
        evidence = np.log(p / (1 - p)) - np.log(prior / (1 - prior))
        calibrated[c] = round(float(1 / (1 + np.exp(-evidence))), 4)
    return calibrated


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class Person:
    box: BoundingBox
    detection_confidence: float
    generalized_emotion: str
    category_scores: dict = field(default_factory=dict)   # 26 categories -> 0-1
    vad: dict = field(default_factory=dict)                # V/A/D -> 1-10
    age: str = ""
    gender: str = ""
    top_categories: list = field(default_factory=list)    # [(name, score)] top-K, best first
    raw_output: dict = field(default_factory=dict)         # untouched model outputs (for debugging)
    calibrated: bool = False                                # True if calibration.json was applied
    calibrated_scores: dict = field(default_factory=dict)   # category_scores after calibration (empty if not calibrated)


class InferenceError(Exception):
    """Raised when the pipeline itself fails (not a validation error)."""


@lru_cache(maxsize=1)
def _load_sessions():
    classifier_path = next((p for p in CLASSIFIER_CANDIDATES if p.exists()), None)
    if not DETECTOR_PATH.exists() or classifier_path is None:
        raise InferenceError(
            "Model file(s) not found. Put detector.onnx and "
            "emotic_attribute_model.onnx in the 'models' folder."
        )
    providers = ["CPUExecutionProvider"]
    return (
        ort.InferenceSession(str(DETECTOR_PATH), providers=providers),
        ort.InferenceSession(str(classifier_path), providers=providers),
    )


def _input_size(session) -> int:
    """Square input size (H = W) the model expects, e.g. 224 or 640."""
    return int(session.get_inputs()[0].shape[2])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _detect_people(session, image: np.ndarray) -> list[tuple[BoundingBox, float, list]]:
    """Letterbox -> YOLO forward -> decode -> NMS -> boxes in original pixels."""
    size = _input_size(session)
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    blob = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    raw = session.run(None, {session.get_inputs()[0].name: blob})[0]  # [1, 5, N]
    preds = raw[0].T                                                    # [N, 5]

    keep = preds[:, 4] >= DETECTION_CONF
    preds = preds[keep]
    if len(preds) == 0:
        return []

    cx, cy, bw, bh, scores = preds.T
    x1 = (cx - bw / 2 - pad_x) / scale
    y1 = (cy - bh / 2 - pad_y) / scale
    bw_o, bh_o = bw / scale, bh / scale

    boxes_xywh = np.stack([x1, y1, bw_o, bh_o], axis=1)
    idxs = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), DETECTION_CONF, NMS_IOU)

    results = []
    for i in np.array(idxs).flatten():
        bx1 = int(np.clip(x1[i], 0, w - 1))
        by1 = int(np.clip(y1[i], 0, h - 1))
        bx2 = int(np.clip(x1[i] + bw_o[i], 0, w))
        by2 = int(np.clip(y1[i] + bh_o[i], 0, h))
        if bx2 - bx1 < 4 or by2 - by1 < 4:
            continue
        # raw = the detector's untouched (cx, cy, w, h, score) row, in 640x640 letterbox space
        raw_row = [round(float(v), 4) for v in preds[i]]
        results.append((BoundingBox(bx1, by1, bx2, by2), float(scores[i]), raw_row))
    return results


def _classify_person(session, image: np.ndarray, box: BoundingBox) -> dict:
    crop = image[box.y1:box.y2, box.x1:box.x2]
    size = _input_size(session)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    x = (crop.astype(np.float32) / 255.0 - NORM_MEAN) / NORM_STD
    x = x.transpose(2, 0, 1)[None].astype(np.float32)

    age, gender, continuous, emotions = session.run(
        None, {session.get_inputs()[0].name: x}
    )
    return {
        "emotions": _sigmoid(emotions[0]),      # multi-label -> independent probs
        "vad": np.clip(continuous[0], 1, 10),
        "age": AGE_CLASSES[int(np.argmax(_softmax(age[0])))],
        "gender": GENDER_CLASSES[int(np.argmax(_softmax(gender[0])))],
        # Raw, unprocessed network outputs (before sigmoid / softmax / clipping)
        "raw": {
            "emotion_logits": {c: round(float(v), 4) for c, v in zip(EMOTIC_CATEGORIES, emotions[0])},
            "vad_raw": {d: round(float(v), 4) for d, v in zip(VAD_DIMENSIONS, continuous[0])},
            "age_logits": {c: round(float(v), 4) for c, v in zip(AGE_CLASSES, age[0])},
            "gender_logits": {c: round(float(v), 4) for c, v in zip(GENDER_CLASSES, gender[0])},
        },
    }


def run_inference(image: np.ndarray) -> list[Person]:
    """
    Detect people and classify each one's emotion.

    Args:
        image: HxWx3 RGB uint8 numpy array.
    Returns:
        List of Person (empty if nobody was detected).
    """
    if image is None or image.size == 0:
        raise InferenceError("Received an empty image.")

    try:
        detector, classifier = _load_sessions()
        detections = _detect_people(detector, image)

        people = []
        for box, conf, det_raw in detections:
            out = _classify_person(classifier, image, box)
            scores = {c: round(float(s), 4) for c, s in zip(EMOTIC_CATEGORIES, out["emotions"])}
            calibrated_scores = _calibrate(scores)
            # Rank by calibrated scores when available; category_scores stays
            # the raw sigmoid output either way (shown in the raw-output view).
            ranked = sorted((calibrated_scores or scores).items(), key=lambda kv: kv[1], reverse=True)
            top = ranked[0][0]
            people.append(Person(
                box=box,
                detection_confidence=round(conf, 3),
                generalized_emotion=CATEGORY_TO_GROUP[top],
                category_scores=scores,
                vad={d: round(float(v), 1) for d, v in zip(VAD_DIMENSIONS, out["vad"])},
                age=out["age"],
                gender=out["gender"],
                top_categories=ranked[:TOP_K],
                calibrated=calibrated_scores is not None,
                calibrated_scores=calibrated_scores or {},
                raw_output={
                    "detector_row_cx_cy_w_h_score": det_raw,
                    **out["raw"],
                },
            ))
        return people
    except InferenceError:
        raise
    except Exception as e:  # surface pipeline failures as a friendly UI error
        raise InferenceError(str(e)) from e