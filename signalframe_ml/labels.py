"""Canonical label spaces for MELD-style multimodal affect classification."""

EMOTION_NAMES = (
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
)

SENTIMENT_NAMES = ("negative", "neutral", "positive")

EMOTION_ID_TO_NAME = {i: name for i, name in enumerate(EMOTION_NAMES)}
SENTIMENT_ID_TO_NAME = {i: name for i, name in enumerate(SENTIMENT_NAMES)}
