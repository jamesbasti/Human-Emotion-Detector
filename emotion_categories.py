"""
EMOTIC category definitions.

Source: Kosti et al., "EMOTIC: Emotions in Context Dataset" (CVPR 2017 Workshops).
https://s3.sunai.uoc.edu/emotic/annotations.html

This module is pure data / config. It has no dependency on the inference
pipeline, so it's safe to import from both `inference.py` and `app.py`.
"""

# The 26 fine-grained EMOTIC categories, in the dataset's canonical order.
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

# --------------------------------------------------------------------------
# Generalized emotion groups.
#
# The SRS calls for a coarse "generalized emotion group" label shown by
# default, with the 26-category/VAD detail available behind a toggle.
# EMOTIC itself does not define an official coarse grouping, so this is a
# reasonable placeholder mapping for the UI/demo. Replace GENERALIZED_GROUPS
# below with however your trained classifier actually buckets its coarse
# output once that's defined — the UI only depends on the dict shape.
# --------------------------------------------------------------------------
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
