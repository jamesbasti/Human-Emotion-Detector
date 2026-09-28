"""
EMOTIC category definitions.

This module is pure data / config. It has no dependency on the inference
pipeline, so it's safe to import from both `inference.py` and `app.py`.
"""

EMOTIC_CATEGORIES = [
    "Peace", "Affection", "Esteem", "Anticipation", "Engagement",
    "Confidence", "Happiness", "Pleasure", "Excitement", "Surprise",
    "Sympathy", "Doubt/Confusion", "Disconnection", "Fatigue",
    "Embarrassment", "Yearning", "Disapproval", "Aversion", "Annoyance",
    "Anger", "Sensitivity", "Sadness", "Disquietment", "Fear", "Pain",
    "Suffering",
]

# Valence / Arousal / Dominance are continuous dimensions in EMOTIC,
# each annotated on an integer 1-10 scale.
VAD_DIMENSIONS = ["Valence", "Arousal", "Dominance"]
VAD_RANGE = (1, 10)

GENERALIZED_GROUPS = {
    "Happiness":       ["Affection", "Excitement", "Confidence", "Anticipation", "Happiness", "Peace", "Pleasure", "Esteem", "Engagement", "Sympathy"],
    "Sadness":        ["Sadness", "Yearning", "Suffering", "Fatigue", "Sensitivity"],
    "Anger":        ["Anger", "Disapproval", "Doubt/Confusion", "Annoyance", "Pain"],
    "Fear":        ["Fear", "Disquietment", "Embarrassment",],
    "Disgust":        ["Aversion"],
    "Surprise":        ["Surprise"],
    "Neutral":        ["Disconnection"]
}

CATEGORY_TO_GROUP = {
    category: group
    for group, categories in GENERALIZED_GROUPS.items()
    for category in categories
}
