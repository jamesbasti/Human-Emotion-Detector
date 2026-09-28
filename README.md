# Human Emotion Detector

Streamlit app that detects people in an image and predicts their emotions
(EMOTIC's 26 categories + Valence/Arousal/Dominance).

Pipeline: YOLO11n person detector (ONNX) -> ResNet18 emotion classifier (ONNX).

## Setup

```
pip install -r requirements.txt
```

Put the model files in a `models/` folder:

- `models/detector.onnx`
- `models/emotic_attribute_model.onnx`

## Run

```
py -m streamlit run app.py
```

## Status

- Image mode: working
- Video / Webcam modes: not implemented yet

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `inference.py` | Runs the ONNX detector + classifier |
| `utils.py` | File validation and box drawing |
| `emotion_categories.py` | EMOTIC categories and generalized groups |
