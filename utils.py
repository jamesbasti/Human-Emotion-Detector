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


def draw_detections(image: np.ndarray, people: list, show_attributes: bool = False) -> np.ndarray:
    """
    Draw one bounding box + a label listing the top-K emotion categories (with
    scores) per detected person, each in a distinct color, onto a copy of
    `image`. If `show_attributes` is True, age group and gender are added to
    the label header.
    Line/font sizes scale with the image so labels stay readable on big photos.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    s = max(1.0, max(h, w) / 900)
    font_scale = 0.6 * s
    box_thickness = max(2, int(3 * s))
    text_thickness = max(1, int(2 * s))

    for i, person in enumerate(people):
        color = BOX_COLORS[i % len(BOX_COLORS)]
        box = person.box
        cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), color, box_thickness)

        # Multi-line label: header + top-K categories with their scores.
        header = f"Person {i + 1}"
        if show_attributes and (person.age or person.gender):
            header += f" | {person.age}, {person.gender}"
        lines = [header] + [f"{name} {score:.0%}" for name, score in person.top_categories]

        pad = int(4 * s)
        sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness) for t in lines]
        line_h = max(th + bl for (_, th), bl in sizes) + pad
        box_w = max(tw for (tw, _), _ in sizes) + 2 * pad
        box_h = line_h * len(lines) + pad

        # Put the label above the box; if there's no room, tuck it inside the top edge.
        label_y1 = box.y1 - box_h
        if label_y1 < 0:
            label_y1 = box.y1
        label_x1 = min(box.x1, max(0, w - box_w))
        cv2.rectangle(annotated, (label_x1, label_y1), (label_x1 + box_w, label_y1 + box_h), color, -1)

        for j, text in enumerate(lines):
            (_, th), bl = sizes[j]
            ty = label_y1 + pad + j * line_h + th
            cv2.putText(
                annotated, text, (label_x1 + pad, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA,
            )

    return annotated


def fit_for_display(image: np.ndarray, max_w: int = 700, max_h: int = 520) -> np.ndarray:
    """Shrink (never enlarge) an image to fit inside max_w x max_h, keeping aspect ratio."""
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)