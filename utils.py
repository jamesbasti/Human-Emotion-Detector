"""
Helpers for validating uploads and drawing detection overlays.
Kept separate from app.py so app.py stays readable as pure UI flow.
"""

from __future__ import annotations

import cv2
import numpy as np

# A rotating palette so multiple people in one frame get visually distinct
# boxes/labels (REQ-26), independent of what emotion they're labeled with.
BOX_COLORS = [
    (255, 99, 71),    # tomato
    (65, 105, 225),   # royal blue
    (60, 179, 113),   # medium sea green
    (255, 165, 0),    # orange
    (186, 85, 211),   # medium orchid
    (0, 191, 255),    # deep sky blue
    (255, 20, 147),   # deep pink
]

MAX_IMAGE_MB = 10
ALLOWED_IMAGE_EXTS = {"jpg", "jpeg", "png"}


def validate_image_file(uploaded_file) -> tuple[bool, str]:
    """Returns (is_valid, error_message). error_message is '' when valid."""
    if uploaded_file is None:
        return False, "No file provided."

    ext = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
    if ext not in ALLOWED_IMAGE_EXTS:
        return False, f"Unsupported file type '.{ext}'. Please upload a JPG or PNG image."

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_IMAGE_MB:
        return False, f"File is {size_mb:.1f} MB, which exceeds the {MAX_IMAGE_MB} MB limit."

    return True, ""


def draw_detections(image: np.ndarray, people: list) -> np.ndarray:
    """
    Draw one bounding box + generalized-emotion label per detected person,
    each in a distinct color, onto a copy of `image`.
    """
    annotated = image.copy()

    for i, person in enumerate(people):
        color = BOX_COLORS[i % len(BOX_COLORS)]
        box = person.box
        cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), color, 3)

        label = f"Person {i + 1}: {person.generalized_emotion}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        label_y1 = max(0, box.y1 - text_h - baseline - 6)
        cv2.rectangle(annotated, (box.x1, label_y1), (box.x1 + text_w + 8, box.y1), color, -1)

        text_color = (255, 255, 255)
        cv2.putText(
            annotated, label, (box.x1 + 4, box.y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA,
        )

    return annotated
