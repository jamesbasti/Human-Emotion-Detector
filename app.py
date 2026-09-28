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
from utils import validate_image_file, draw_detections, fit_for_display
from emotion_categories import EMOTIC_CATEGORIES, VAD_DIMENSIONS

st.set_page_config(page_title="Emotion Detector", layout="wide")

# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------
if "show_breakdown" not in st.session_state:
    st.session_state.show_breakdown = False  # REQ-23: persists for the session
if "show_attributes" not in st.session_state:
    st.session_state.show_attributes = False
if "show_raw" not in st.session_state:
    st.session_state.show_raw = False

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
    st.session_state.show_attributes = st.toggle(
        "Show age and gender",
        value=st.session_state.show_attributes,
    )
    st.session_state.show_raw = st.toggle(
        "Show raw model output",
        value=st.session_state.show_raw,
    )

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
                st.image(fit_for_display(image_np))
            else:
                annotated = draw_detections(
                    image_np, people, show_attributes=st.session_state.show_attributes
                )
                st.image(fit_for_display(annotated))  # REQ-5/REQ-6/REQ-26

                # Top-3 emotion categories per person (also drawn on the image)
                st.markdown("### Top 3 emotions")
                for i, person in enumerate(people):
                    top3 = " , ".join(f"{name} ({score:.0%})" for name, score in person.top_categories)
                    st.markdown(f"**Person {i + 1}:** {top3}")

                if st.session_state.show_attributes:
                    for i, person in enumerate(people):
                        st.markdown(f"**Person {i + 1}:** {person.age}, {person.gender}")
                    
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
                                use_container_width=True, hide_index=True,
                            )

                            st.markdown("**Classifier: 26 emotion logits** (canonical EMOTIC order)")
                            logit_df = pd.DataFrame(
                                {
                                    "Category": EMOTIC_CATEGORIES,
                                    "Logit (raw)": [raw["emotion_logits"][c] for c in EMOTIC_CATEGORIES],
                                    "Sigmoid (prob)": [person.category_scores[c] for c in EMOTIC_CATEGORIES],
                                }
                            )
                            st.dataframe(logit_df, use_container_width=True, hide_index=True, height=280)

                            c1, c2, c3 = st.columns(3)
                            c1.markdown("**VAD (unclipped)**")
                            c1.dataframe(
                                pd.DataFrame({"Dim": list(raw["vad_raw"]), "Value": list(raw["vad_raw"].values())}),
                                hide_index=True, use_container_width=True,
                            )
                            c2.markdown("**Age logits**")
                            c2.dataframe(
                                pd.DataFrame({"Class": list(raw["age_logits"]), "Logit": list(raw["age_logits"].values())}),
                                hide_index=True, use_container_width=True,
                            )
                            c3.markdown("**Gender logits**")
                            c3.dataframe(
                                pd.DataFrame({"Class": list(raw["gender_logits"]), "Logit": list(raw["gender_logits"].values())}),
                                hide_index=True, use_container_width=True,
                            )

                            with st.expander("Full raw JSON"):
                                st.json(raw)