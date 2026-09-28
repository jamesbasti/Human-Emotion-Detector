"""
Video mode: validate an uploaded video, run detection+classification
across it on a sampled-frame basis, write an annotated MP4, and report
progress back to the caller for a progress bar.

Kept separate from inference.py, which stays about single images/frames.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio

from inference import run_inference, InferenceError
from utils import draw_detections

MAX_VIDEO_MB = 200
MAX_VIDEO_SECONDS = 60
ALLOWED_VIDEO_EXTS = {"mp4"}

# Run the (two-model) inference pipeline on 1 out of every N frames, and
# reuse the last result on the frames in between. This is the "sampling
# strategy for performance" REQ-12 asks for — full per-frame inference on
# CPU is too slow to be usable on anything but very short clips.
SAMPLE_EVERY_N_FRAMES = 5


def validate_video_file(uploaded_file) -> tuple[bool, str]:
    """Returns (is_valid, error_message). error_message is '' when valid."""
    if uploaded_file is None:
        return False, "No file provided."

    ext = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
    if ext not in ALLOWED_VIDEO_EXTS:
        return False, f"Unsupported file type '.{ext}'. Please upload an MP4 video."

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_VIDEO_MB:
        return False, f"File is {size_mb:.1f} MB, which exceeds the {MAX_VIDEO_MB} MB limit."

    return True, ""


def process_video(input_path: str, output_path: str, progress_callback=None) -> dict:
    """
    Run detection+classification across the video and write an annotated
    copy to `output_path`.

    Args:
        input_path: path to the uploaded video on disk.
        output_path: where to write the annotated MP4.
        progress_callback: optional callable(fraction_done: float) -> None.

    Returns:
        dict with basic stats (frame_count, fps, people_seen_max).

    Raises:
        InferenceError if the video can't be read or has no frames.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise InferenceError("Could not open the uploaded video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if w == 0 or h == 0:
        cap.release()
        raise InferenceError("The video has no readable frames.")

    duration = frame_count / fps if fps else 0
    if duration > MAX_VIDEO_SECONDS:
        cap.release()
        raise InferenceError(
            f"Video is {duration:.0f}s, which exceeds the {MAX_VIDEO_SECONDS}s limit."
        )

    # imageio (+ its bundled ffmpeg) encodes real H.264, so the output plays
    # in the browser. cv2.VideoWriter's mp4v codec looked fine locally but
    # is not playable in Chrome/Firefox <video>, so don't swap this back.
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=7)

    last_people = []
    max_people = 0
    i = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if i % SAMPLE_EVERY_N_FRAMES == 0:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                last_people = run_inference(rgb)
                max_people = max(max_people, len(last_people))

            annotated_rgb = draw_detections(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), last_people)
            writer.append_data(annotated_rgb)

            i += 1
            if progress_callback and frame_count:
                progress_callback(min(1.0, i / frame_count))
    finally:
        cap.release()
        writer.close()

    if i == 0:
        raise InferenceError("The video has no readable frames.")

    return {"frame_count": i, "fps": fps, "max_people_seen": max_people}
