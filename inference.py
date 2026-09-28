"""
Inference pipeline: ONNX person detector (YOLO11n) -> ONNX emotion classifier.

app.py and utils.py only use `run_inference()` and the `Person` / `BoundingBox`
shapes below.

Models (place in the `models/` folder next to this file):
  - detector.onnx               YOLO11n, 1 class ("person"), 224x224 input,
                                raw output [1, 5, 1029] = (cx, cy, w, h, score)
  - emotic_attribute_model.onnx ResNet18, 224x224 person crop ->
                                emotions (26 logits), continuous (VAD),
                                age (3 logits), gender (2 logits)

Things to CONFIRM with whoever trained the classifier (see CONFIG below):
  1. Input normalization (assumed ImageNet mean/std on RGB in [0, 1]).
  2. Order of the 26 emotion outputs (assumed EMOTIC's canonical order).
  3. Age / gender class order (only used for the optional attributes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from emotion_categories import EMOTIC_CATEGORIES, CATEGORY_TO_GROUP, VAD_DIMENSIONS

# ------------------------------- CONFIG ------------------------------------
MODELS_DIR = Path(__file__).parent / "models"
DETECTOR_PATH = MODELS_DIR / "detector.onnx"
CLASSIFIER_PATH = MODELS_DIR / "emotic_attribute_model.onnx"

DETECTOR_SIZE = 224          # detector input is fixed at 224x224
CLASSIFIER_SIZE = 224        # classifier input is fixed at 224x224
DETECTION_CONF = 0.25        # min person confidence to keep a box
NMS_IOU = 0.45               # overlap threshold for non-max suppression

# ASSUMPTION: classifier was trained with ImageNet normalization.
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ASSUMPTION: EMOTIC attribute order. Only used for the optional fields.
AGE_CLASSES = ["Kid", "Teenager", "Adult"]
GENDER_CLASSES = ["Female", "Male"]
# ---------------------------------------------------------------------------


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


class InferenceError(Exception):
    """Raised when the pipeline itself fails (not a validation error)."""


@lru_cache(maxsize=1)
def _load_sessions():
    for p in (DETECTOR_PATH, CLASSIFIER_PATH):
        if not p.exists():
            raise InferenceError(
                f"Model file not found: {p}. Put detector.onnx and "
                f"emotic_attribute_model.onnx in the 'models' folder."
            )
    providers = ["CPUExecutionProvider"]
    return (
        ort.InferenceSession(str(DETECTOR_PATH), providers=providers),
        ort.InferenceSession(str(CLASSIFIER_PATH), providers=providers),
    )


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _detect_people(session, image: np.ndarray) -> list[tuple[BoundingBox, float]]:
    """Letterbox -> YOLO forward -> decode -> NMS -> boxes in original pixels."""
    h, w = image.shape[:2]
    scale = min(DETECTOR_SIZE / h, DETECTOR_SIZE / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pad_x, pad_y = (DETECTOR_SIZE - new_w) // 2, (DETECTOR_SIZE - new_h) // 2

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((DETECTOR_SIZE, DETECTOR_SIZE, 3), 114, dtype=np.uint8)
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
        results.append((BoundingBox(bx1, by1, bx2, by2), float(scores[i])))
    return results


def _classify_person(session, image: np.ndarray, box: BoundingBox) -> dict:
    crop = image[box.y1:box.y2, box.x1:box.x2]
    crop = cv2.resize(crop, (CLASSIFIER_SIZE, CLASSIFIER_SIZE), interpolation=cv2.INTER_LINEAR)
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
        for box, conf in detections:
            out = _classify_person(classifier, image, box)
            scores = {c: round(float(s), 4) for c, s in zip(EMOTIC_CATEGORIES, out["emotions"])}
            top = max(scores, key=scores.get)
            people.append(Person(
                box=box,
                detection_confidence=round(conf, 3),
                generalized_emotion=CATEGORY_TO_GROUP[top],
                category_scores=scores,
                vad={d: round(float(v), 1) for d, v in zip(VAD_DIMENSIONS, out["vad"])},
                age=out["age"],
                gender=out["gender"],
            ))
        return people
    except InferenceError:
        raise
    except Exception as e:  # surface pipeline failures as a friendly UI error
        raise InferenceError(str(e)) from e
