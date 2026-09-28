"""
Emotion Detector — Streamlit front end.
"""

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import tempfile
from pathlib import Path

from inference import run_inference, InferenceError
from utils import validate_image_file, draw_detections, fit_for_display
from video import validate_video_file, process_video
from webcam import WEBRTC_AVAILABLE, process_frame
if WEBRTC_AVAILABLE:
    from webcam import render_live
from emotion_categories import EMOTIC_CATEGORIES, VAD_DIMENSIONS

st.set_page_config(page_title="Emotion Detector", layout="wide")

# Max on-page width (px) for the detection frame — images, video playback,
# and the live webcam feed all use this so nothing overflows the layout
# on a large source photo/video/camera resolution. Matches utils.fit_for_display's
# default so the sizing feels consistent across modes.
DISPLAY_MAX_W = 700

# Session state defaults
if "show_breakdown" not in st.session_state:
    st.session_state.show_breakdown = False  # persists for the session
if "show_attributes" not in st.session_state:
    st.session_state.show_attributes = False
if "show_raw" not in st.session_state:
    st.session_state.show_raw = False

# Header
st.title("Emotion Detector")

# Sidebar — mode selector, upload area, detection settings
with st.sidebar:
    mode = st.radio("Input mode", ["Image", "Video", "Webcam"], horizontal=True)  

    st.divider()

    uploaded_file = None
    if mode == "Image":
        uploaded_file = st.file_uploader(
            "Drag or upload photo here", type=["jpg", "jpeg", "png"]
        )  
    elif mode == "Video":
        uploaded_video = st.file_uploader("Upload a video (MP4)", type=["mp4"])  
    else:  # Webcam
        if WEBRTC_AVAILABLE:
            st.caption("Live webcam. Use the Start/Stop button below to release the camera.")
        else:
            st.caption(
                "Live webcam isn't available (optional package not installed), "
                "so this captures one photo at a time instead."
            )
            webcam_photo = st.camera_input("Webcam")  # browser prompts for permission

    st.divider()
    st.subheader("Detection Settings")
    st.session_state.show_breakdown = st.toggle(
        "Show Emotion category breakdown",
        value=st.session_state.show_breakdown,
    )  
    st.session_state.show_attributes = st.toggle(
        "Show age and gender",
        value=st.session_state.show_attributes,
    )
    st.session_state.show_raw = st.toggle(
        "Show raw model output",
        value=st.session_state.show_raw,
    )

# Main content — the persistent Detection frame panel
st.subheader("Detection frame")

if mode == "Video":
    if uploaded_video is None:
        st.markdown("*upload a video from the sidebar to run detection.*")  
    else:
        is_valid, error_msg = validate_video_file(uploaded_video)
        if not is_valid:
            st.error(error_msg)  
        else:
            st.caption(uploaded_video.name)  

            tmp_dir = Path(tempfile.gettempdir())
            in_path = tmp_dir / f"in_{uploaded_video.name}"
            out_path = tmp_dir / f"out_{uploaded_video.name}"
            in_path.write_bytes(uploaded_video.getvalue())

            if st.button("Process video"):
                progress = st.progress(0.0, text="Processing video...")  
                try:
                    stats = process_video(
                        str(in_path), str(out_path),
                        progress_callback=lambda f: progress.progress(f, text=f"Processing video... {f:.0%}"),
                    )
                    progress.empty()
                    st.session_state["video_result"] = str(out_path)
                    if stats["max_people_seen"] == 0:
                        st.warning("No persons were found in this video.")  # equivalent
                except InferenceError as e:
                    progress.empty()
                    st.error(f"Detection failed: {e}")
                    st.session_state.pop("video_result", None)

            if st.session_state.get("video_result"):
                st.video(st.session_state["video_result"], width=DISPLAY_MAX_W)  # native play/pause/scrub

elif mode == "Webcam":
    if WEBRTC_AVAILABLE:
        # Live mode: detection runs inside the video callback (webcam.py);
        # the component itself provides the Start/Stop control (REQ-19).
        render_live(show_attributes=st.session_state.show_attributes, max_width=DISPLAY_MAX_W)
    elif webcam_photo is None:
        st.markdown("*allow camera access and take a photo.*")
    else:
        image_np = np.array(Image.open(webcam_photo).convert("RGB"))
        with st.spinner("Running detection..."):
            try:
                people = run_inference(image_np)
            except InferenceError as e:
                people = None
                st.error(f"Detection failed: {e}")

        if people is not None:
            if len(people) == 0:
                st.warning("No persons were found.")
                st.image(fit_for_display(image_np))
            else:
                annotated = draw_detections(
                    image_np, people, show_attributes=st.session_state.show_attributes
                )
                st.image(fit_for_display(annotated))
        st.caption("Take another photo above to run detection again, or switch modes to stop the camera.")

elif uploaded_file is None:
    # placeholder state before first submitted input
    st.markdown(
        "*upload a photo to run detection.*"
    )

else:
    is_valid, error_msg = validate_image_file(uploaded_file)

    if not is_valid:
        st.error(error_msg)  
    else:
        st.caption(uploaded_file.name)  # source identifier

        pil_image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(pil_image)

        with st.spinner("Running detection..."):
            try:
                people = run_inference(image_np)
            except InferenceError as e:
                people = None
                st.error(f"Detection failed: {e}")

        if people is not None:
            if len(people) == 0:
                # no person detected
                st.warning("No persons were found in this image.")
                st.image(fit_for_display(image_np))
            else:
                annotated = draw_detections(
                    image_np, people, show_attributes=st.session_state.show_attributes
                )
                st.image(fit_for_display(annotated))  

                # Top-3 emotion categories per person (also drawn on the image)
                st.markdown("### Top 3 emotions")
                for i, person in enumerate(people):
                    top3 = " , ".join(f"{name} ({score:.0%})" for name, score in person.top_categories)
                    st.markdown(f"**Person {i + 1}:** {top3}")

                if st.session_state.show_attributes:
                    for i, person in enumerate(people):
                        st.markdown(f"**Person {i + 1}:** {person.age}, {person.gender}")
                    
                if st.session_state.show_breakdown:
                    # 26-category scores + VAD per person
                    st.divider()
                    st.markdown("### Emotion category breakdown")
                    if people and people[0].calibrated:
                        st.caption(
                            "Scores are calibrated against each category's training "
                            "frequency (calibration.json), so common categories like "
                            "Engagement don't win by default. Ranking above uses "
                            "calibrated scores; 'Raw' below is the uncalibrated model output."
                        )
                    else:
                        st.caption(
                            "No calibration.json found next to inference.py — showing raw, "
                            "uncalibrated model output. See tools/compute_priors.py to generate one."
                        )
                    for i, person in enumerate(people):
                        with st.expander(
                            f"Person {i + 1} — {person.generalized_emotion} "
                            f"(detection confidence {person.detection_confidence:.0%})",
                            expanded=(len(people) == 1),
                        ):
                            vad_cols = st.columns(len(VAD_DIMENSIONS))
                            for col, dim in zip(vad_cols, VAD_DIMENSIONS):
                                col.metric(dim, f"{person.vad[dim]:.1f} / 10")

                            if person.calibrated:
                                scores_df = (
                                    pd.DataFrame(
                                        {
                                            "Category": EMOTIC_CATEGORIES,
                                            "Calibrated": [person.calibrated_scores[c] for c in EMOTIC_CATEGORIES],
                                            "Raw": [person.category_scores[c] for c in EMOTIC_CATEGORIES],
                                        }
                                    )
                                    .sort_values("Calibrated", ascending=False)
                                    .reset_index(drop=True)
                                )
                            else:
                                scores_df = (
                                    pd.DataFrame(
                                        {
                                            "Category": EMOTIC_CATEGORIES,
                                            "Score": [person.category_scores[c] for c in EMOTIC_CATEGORIES],
                                        }
                                    )
                                    .sort_values("Score", ascending=False)
                                    .reset_index(drop=True)
                                )
                            st.dataframe(
                                scores_df, width="stretch", hide_index=True, height=280
                            )

                if st.session_state.show_raw:
                    st.divider()
                    st.markdown("### Raw model output")
                    st.caption(
                        "Values straight from the networks, before sigmoid / softmax / clipping. "
                        "Emotion logits become probabilities via sigmoid; age and gender logits via softmax."
                    )
                    for i, person in enumerate(people):
                        raw = person.raw_output
                        with st.expander(f"Person {i + 1} — raw output", expanded=(len(people) == 1)):
                            cx, cy, bw, bh, score = raw["detector_row_cx_cy_w_h_score"]
                            st.markdown("**Detector (YOLO) row** — 640×640 letterbox space")
                            st.dataframe(
                                pd.DataFrame([{"cx": cx, "cy": cy, "w": bw, "h": bh, "score": score}]),
                                width="stretch", hide_index=True,
                            )

                            st.markdown("**Classifier: 26 emotion logits** (canonical EMOTIC order)")
                            logit_df = pd.DataFrame(
                                {
                                    "Category": EMOTIC_CATEGORIES,
                                    "Logit (raw)": [raw["emotion_logits"][c] for c in EMOTIC_CATEGORIES],
                                    "Sigmoid (prob)": [person.category_scores[c] for c in EMOTIC_CATEGORIES],
                                }
                            )
                            st.dataframe(logit_df, width="stretch", hide_index=True, height=280)

                            c1, c2, c3 = st.columns(3)
                            c1.markdown("**VAD (unclipped)**")
                            c1.dataframe(
                                pd.DataFrame({"Dim": list(raw["vad_raw"]), "Value": list(raw["vad_raw"].values())}),
                                hide_index=True, width="stretch",
                            )
                            c2.markdown("**Age logits**")
                            c2.dataframe(
                                pd.DataFrame({"Class": list(raw["age_logits"]), "Logit": list(raw["age_logits"].values())}),
                                hide_index=True, width="stretch",
                            )
                            c3.markdown("**Gender logits**")
                            c3.dataframe(
                                pd.DataFrame({"Class": list(raw["gender_logits"]), "Logit": list(raw["gender_logits"].values())}),
                                hide_index=True, width="stretch",
                            )

                            with st.expander("Full raw JSON"):
                                st.json(raw)