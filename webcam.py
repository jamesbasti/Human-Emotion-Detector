"""
Webcam mode: tries live streaming via streamlit-webrtc; if that library
isn't installed (or fails to import for any reason), the app falls back
to the snapshot-based st.camera_input flow automatically.

Why an optional dependency instead of always requiring it: streamlit-webrtc
pulls in `aiortc`, which needs `av` (an FFmpeg binding). It's more likely
to fail to install or to fail to connect (firewalls, missing codecs) than
anything else in this project. Rather than risk that blocking Webcam mode
entirely, it's opt-in — see README for how to enable it.

`process_frame()` is the actual detection+drawing logic, kept as a plain
function (no av/webrtc types) so it can be unit tested without a live
video stream, and reused identically by both the live callback and the
snapshot fallback.
"""

from __future__ import annotations

import threading

import numpy as np

from inference import run_inference, InferenceError
from utils import draw_detections

# Run the model on 1 out of every N live frames and hold the last result on
# frames in between, same idea as video.py's sampling — full per-frame
# inference on CPU can't keep up with a live feed.
SAMPLE_EVERY_N_LIVE_FRAMES = 8

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


def process_frame(rgb: np.ndarray, people: list, show_attributes: bool) -> np.ndarray:
    """Draw the given (already-computed) detections onto a frame. Pure numpy in/out."""
    try:
        return draw_detections(rgb, people, show_attributes=show_attributes)
    except Exception:
        return rgb  # never let a drawing bug take down a live stream


if WEBRTC_AVAILABLE:

    class EmotionVideoProcessor(VideoProcessorBase):
        """
        Runs on a background thread managed by streamlit-webrtc, so it
        can't safely touch st.session_state — state is kept on `self`
        instead, and the app sets `show_attributes` on the live processor
        instance each rerun (see webcam.py's render_live()).
        """

        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.last_people: list = []
            self.show_attributes = False
            self._frame_idx = 0
            self._last_error = ""

        def recv(self, frame):
            rgb = frame.to_ndarray(format="rgb24")

            self._frame_idx += 1
            if self._frame_idx % SAMPLE_EVERY_N_LIVE_FRAMES == 0:
                try:
                    people = run_inference(rgb)
                    with self.lock:
                        self.last_people = people
                        self._last_error = ""
                except InferenceError as e:
                    with self.lock:
                        self._last_error = str(e)

            with self.lock:
                people, show_attributes = self.last_people, self.show_attributes

            annotated = process_frame(rgb, people, show_attributes)
            return av.VideoFrame.from_ndarray(annotated, format="rgb24")

    def render_live(show_attributes: bool):
        """Render the live webrtc component. Call from app.py inside Webcam mode."""
        ctx = webrtc_streamer(
            key="emotion-webcam",
            video_processor_factory=EmotionVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
        )
        if ctx.video_processor:
            ctx.video_processor.show_attributes = show_attributes
        return ctx
