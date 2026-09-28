"""
Emotion Detector — Streamlit front end.

Maps to the SRS as follows:
  - 3.1 Image Upload and Detection  -> fully implemented (High priority)
  - 3.2 Video Upload and Detection  -> mode selectable, marked "coming soon"
  - 3.3 Live Webcam Feed Detection  -> mode selectable, marked "coming soon"
  - 3.4 Emotion Category Breakdown  -> "Show Emotion category breakdown" toggle
  - 3.5 Detection Results Display   -> the persistent "Detection frame" panel
  - 4.1 User Interfaces             -> header + sidebar + main content layout
"""

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from inference import run_inference, InferenceError
from utils import validate_image_file, draw_detections
from emotion_categories import EMOTIC_CATEGORIES, VAD_DIMENSIONS

st.set_page_config(page_title="Emotion Detector", layout="wide")

# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------
if "show_breakdown" not in st.session_state:
    st.session_state.show_breakdown = False  # REQ-23: persists for the session

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Emotion Detector")

# --------------------------------------------------------------------------
# Sidebar — mode selector + upload area + detection settings
# --------------------------------------------------------------------------
with st.sidebar:
    mode = st.radio("Input mode", ["Image", "Video", "Webcam"], horizontal=True)  # REQ-1/9/15

    st.divider()

    uploaded_file = None
    if mode == "Image":
        uploaded_file = st.file_uploader(
            "Drag or upload photo here", type=["jpg", "jpeg", "png"]
        )  # REQ-2
    elif mode == "Video":
        st.file_uploader("Upload a video (MP4)", type=["mp4"], disabled=True)
    else:  # Webcam
        st.button("Start webcam", disabled=True)

    st.divider()
    st.subheader("Detection Settings")
    st.session_state.show_breakdown = st.toggle(
        "Show Emotion category breakdown",
        value=st.session_state.show_breakdown,
    )  # REQ-20

# --------------------------------------------------------------------------
# Main content — the persistent Detection frame panel (3.5)
# --------------------------------------------------------------------------
st.subheader("Detection frame")

if mode in ("Video", "Webcam"):
    st.info(f"{mode} mode is coming soon. Switch to **Image** mode to try detection now.")

elif uploaded_file is None:
    # REQ-27: placeholder state before first submitted input
    st.markdown(
        "*upload a photo to run detection.*"
    )

else:
    is_valid, error_msg = validate_image_file(uploaded_file)

    if not is_valid:
        st.error(error_msg)  # REQ-3
    else:
        st.caption(uploaded_file.name)  # REQ-7/REQ-25: source identifier

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
                # REQ-8: no person detected
                st.warning("No persons were found in this image.")
                st.image(image_np, use_container_width=True)
            else:
                annotated = draw_detections(image_np, people)
                st.image(annotated, use_container_width=True)  # REQ-5/REQ-6/REQ-26

                if st.session_state.show_breakdown:
                    # REQ-21: 26-category scores + VAD per person
                    st.divider()
                    st.markdown("### Emotion category breakdown")
                    for i, person in enumerate(people):
                        with st.expander(
                            f"Person {i + 1} — {person.generalized_emotion} "
                            f"(detection confidence {person.detection_confidence:.0%})",
                            expanded=(len(people) == 1),
                        ):
                            vad_cols = st.columns(len(VAD_DIMENSIONS))
                            for col, dim in zip(vad_cols, VAD_DIMENSIONS):
                                col.metric(dim, f"{person.vad[dim]:.1f} / 10")

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
                                scores_df, use_container_width=True, hide_index=True, height=280
                            )
