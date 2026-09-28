# Human Emotion Detector

Streamlit app that detects people in an image or video and predicts their
emotions (EMOTIC's 26 categories + Valence/Arousal/Dominance), plus a
webcam snapshot mode.

Pipeline: YOLO person detector (ONNX) -> ResNet18 emotion classifier (ONNX).

## Setup

### 1. Install Python

Python 3.11, 3.12, or 3.13. **Avoid 3.14** — `onnxruntime` may not have a
build for it yet, which will make step 2 fail.

Check your version:
```
py --version
```

### 2. Install dependencies

From the project folder:
```
py -m pip install -r requirements.txt
```

This installs everything below. If you ever get a `ModuleNotFoundError`,
it means this step wasn't run (or wasn't run inside the same Python that
`streamlit run` uses) — rerun it.

| Package | Used for |
|---|---|
| `streamlit` | The web app framework — sidebar, image/video display, all UI |
| `opencv-python-headless` | Resizing frames, drawing boxes/labels, letterboxing for the detector |
| `numpy` | Array math for image data and model inputs/outputs |
| `pandas` | The emotion-category breakdown and raw-output tables |
| `pillow` | Reading uploaded images and webcam photos |
| `onnxruntime` | Runs the two `.onnx` models (detector + classifier) |
| `imageio` | Writes the annotated output video |
| `imageio-ffmpeg` | Bundles an ffmpeg binary so `imageio` can encode real H.264 — without this, output video won't play in the browser |

If you want to install them one at a time instead of using
`requirements.txt`:
```
py -m pip install streamlit opencv-python-headless numpy pandas pillow onnxruntime imageio imageio-ffmpeg
```

### 3. Add the model files

Create a `models/` folder next to `app.py` and put these two files in it:

- `models/detector.onnx`
- `models/emotic_attribute_model.onnx`

(If you still have an older `emotic_attribute_model_1.onnx` or `_2.onnx`
lying around, the app will fall back to those — see
`CLASSIFIER_CANDIDATES` in `inference.py` — but the plain name above is
what it looks for first.)

## Run

```
py -m streamlit run app.py
```

## Status

- Image mode: working
- Video mode: working (uploads an MP4, processes the whole clip with a
  progress bar, then plays back the annotated result). Every 5th frame is
  run through the model and reused for frames in between, to keep CPU
  processing time reasonable — see SAMPLE_EVERY_N_FRAMES in video.py.
- Webcam mode: working, but as a snapshot loop rather than a continuous
  live feed — click "Take Photo" to capture and run detection on-demand.
  True live streaming would need streamlit-webrtc, which pulls in aiortc
  (a heavier, more failure-prone install on Windows); this was a deliberate
  trade-off for reliability. Swap it in later if continuous streaming
  becomes a hard requirement.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `inference.py` | Runs the ONNX detector + classifier on a single image/frame |
| `video.py` | Video upload validation + full-video processing pipeline |
| `utils.py` | File validation and box drawing |
| `emotion_categories.py` | EMOTIC categories and generalized groups |
