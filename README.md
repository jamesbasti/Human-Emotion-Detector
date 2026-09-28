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

Every package this project needs is listed in `requirements.txt` — this
one command installs all of them, including the optional live-webcam
packages at the bottom of that file. It's the only install step most
people need; the rest of this section is for when something goes wrong.

**Verify it worked** before doing anything else:
```
py -c "import streamlit, cv2, numpy, pandas, PIL, onnxruntime, imageio; print('core deps OK')"
py -c "import streamlit_webrtc, av; print('webcam deps OK')"
```
The second line is allowed to fail — that just means live webcam won't be
available and the app will use the snapshot fallback instead (see
"Webcam mode" under Status, below). The first line failing means
something in the main install didn't complete; scroll down to that
package in the table and re-run its individual install command.

If you ever see `ModuleNotFoundError: No module named 'x'` when running
the app, it means the matching package below either wasn't installed, or
was installed into a *different* Python than the one running
`streamlit run` (common on Windows when both `python` and `py` are on
your machine — stick to `py -m pip install` and `py -m streamlit run`
throughout so both commands use the same Python).

| Package | Used for | Install on its own |
|---|---|---|
| `streamlit` | The web app framework — sidebar, image/video display, all UI | `py -m pip install streamlit` |
| `opencv-python-headless` | Resizing frames, drawing boxes/labels, letterboxing for the detector | `py -m pip install opencv-python-headless` |
| `numpy` | Array math for image data and model inputs/outputs | `py -m pip install numpy` |
| `pandas` | The emotion-category breakdown and raw-output tables | `py -m pip install pandas` |
| `pillow` | Reading uploaded images and webcam photos | `py -m pip install pillow` |
| `onnxruntime` | Runs the two `.onnx` models (detector + classifier). **Needs Python 3.11–3.13** — see step 1 | `py -m pip install onnxruntime` |
| `imageio` | Writes the annotated output video | `py -m pip install imageio` |
| `imageio-ffmpeg` | Bundles an ffmpeg binary so `imageio` can encode real H.264 — without this, output video won't play in the browser | `py -m pip install imageio-ffmpeg` |
| `streamlit-webrtc` *(optional)* | Enables live webcam video in Webcam mode | `py -m pip install streamlit-webrtc` |
| `av` *(optional)* | FFmpeg bindings `streamlit-webrtc` needs to decode/encode the live video stream. **Least reliable install in this project** — see "If live webcam doesn't work" below | `py -m pip install av` |

All ten in one line, if you want it:
```
py -m pip install streamlit opencv-python-headless numpy pandas pillow onnxruntime imageio imageio-ffmpeg streamlit-webrtc av
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
- Webcam mode: live video when `streamlit-webrtc` + `av` are installed and
  import successfully. If either isn't available for any reason, the app
  automatically falls back to a snapshot loop ("Take Photo" -> see result ->
  take another) using Streamlit's built-in camera widget instead — no
  crash, no manual switch needed. See "If live webcam doesn't work" below.

## If live webcam doesn't work

`streamlit-webrtc` depends on `aiortc`, which needs a working `av` (FFmpeg
bindings) install. This is the least predictable dependency in this
project -- more likely than anything else here to fail to install, or to
install but fail to connect (usually a firewall blocking the WebRTC
connection). If that happens:

- The app already handles the *unable to install* case for you: it
  detects the missing import and silently uses the snapshot fallback
  instead. You don't need to do anything.
- If it installs but the video never connects (spinner that never loads,
  or a permissions-looking error in the browser), just uninstall it --
  `py -m pip uninstall streamlit-webrtc av` -- and rerun the app. It will
  fall back automatically.
- Camera permission is requested by the browser either way (live or
  snapshot), so denying it produces the same "no camera access" behavior
  in both modes.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `inference.py` | Runs the ONNX detector + classifier on a single image/frame |
| `video.py` | Video upload validation + full-video processing pipeline |
| `webcam.py` | Live webcam video (optional) with automatic fallback detection |
| `utils.py` | File validation and box drawing |
| `emotion_categories.py` | EMOTIC categories and generalized groups |
