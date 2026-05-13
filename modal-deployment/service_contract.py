from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from os import environ
from typing import Any


DEFAULT_MODAL_MODEL_VERSION = "meld-multimodal-v1"
NEUTRAL_EMOTION = "neutral"
NEUTRAL_SENTIMENT = "neutral"


@dataclass(frozen=True)
class ModalServiceSettings:
    auth_token: str | None
    model_version: str

    @classmethod
    def from_env(cls) -> "ModalServiceSettings":
        raw_token = environ.get("MODAL_INFERENCE_AUTH_TOKEN", "").strip()
        raw_model_version = environ.get("MODAL_MODEL_VERSION", "").strip()
        return cls(
            auth_token=raw_token or None,
            model_version=raw_model_version or DEFAULT_MODAL_MODEL_VERSION,
        )


def normalize_source_range(source_start_ms: int, source_end_ms: int) -> tuple[int, int]:
    normalized_start = max(0, int(source_start_ms))
    normalized_end = max(normalized_start + 1, int(source_end_ms))
    return normalized_start, normalized_end


def _extract_bearer_token(authorization_header: str | None) -> str | None:
    if not authorization_header:
        return None

    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None

    return token.strip()


def ensure_authorized(
    settings: ModalServiceSettings,
    authorization_header: str | None,
) -> None:
    if not settings.auth_token:
        return

    provided_token = _extract_bearer_token(authorization_header)
    if provided_token != settings.auth_token:
        raise PermissionError("Missing or invalid bearer token.")


def _coerce_predictions(predictions: Any) -> list[dict[str, Any]]:
    if not isinstance(predictions, list):
        return []

    normalized: list[dict[str, Any]] = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue

        label = prediction.get("label")
        confidence = prediction.get("confidence")
        if not isinstance(label, str):
            continue
        if not isinstance(confidence, (int, float)):
            continue

        normalized.append(
            {
                "label": label,
                "confidence": max(0.0, min(1.0, float(confidence))),
            }
        )

    return normalized


def _coerce_utterances(raw_utterances: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_utterances, list):
        return []

    utterances: list[dict[str, Any]] = []
    for raw_utterance in raw_utterances:
        if not isinstance(raw_utterance, dict):
            continue

        emotions = _coerce_predictions(raw_utterance.get("emotions"))
        sentiments = _coerce_predictions(raw_utterance.get("sentiments"))
        if not emotions or not sentiments:
            continue

        utterance: dict[str, Any] = {
            "emotions": emotions,
            "sentiments": sentiments,
        }

        if isinstance(raw_utterance.get("start_time"), (int, float)):
            utterance["start_time"] = float(raw_utterance["start_time"])
        if isinstance(raw_utterance.get("end_time"), (int, float)):
            utterance["end_time"] = float(raw_utterance["end_time"])
        if isinstance(raw_utterance.get("text"), str):
            utterance["text"] = raw_utterance["text"].strip()

        utterances.append(utterance)

    return utterances


def _collapse_transcript(utterances: list[dict[str, Any]]) -> str | None:
    parts = [
        utterance["text"]
        for utterance in utterances
        if isinstance(utterance.get("text"), str) and utterance["text"].strip()
    ]
    if not parts:
        return None

    return " ".join(parts)


def _aggregate_predictions(
    utterances: list[dict[str, Any]],
    key: str,
) -> list[dict[str, float | str]]:
    aggregate: dict[str, list[float]] = defaultdict(list)

    for utterance in utterances:
        for prediction in utterance.get(key, []):
            label = prediction["label"]
            confidence = prediction["confidence"]
            aggregate[label].append(confidence)

    scored_predictions = [
        {
            "label": label,
            "confidence": sum(confidences) / len(confidences),
        }
        for label, confidences in aggregate.items()
        if confidences
    ]
    scored_predictions.sort(key=lambda prediction: prediction["confidence"], reverse=True)
    return scored_predictions[:3]


def _average_confidence(utterances: list[dict[str, Any]]) -> float:
    utterance_confidences: list[float] = []
    for utterance in utterances:
        emotion_confidence = utterance["emotions"][0]["confidence"]
        sentiment_confidence = utterance["sentiments"][0]["confidence"]
        utterance_confidences.append((emotion_confidence + sentiment_confidence) / 2)

    if not utterance_confidences:
        return 0.0

    return sum(utterance_confidences) / len(utterance_confidences)


def build_timeline_chunk_response(
    *,
    chunk_id: str,
    model_version: str,
    inference_result: dict[str, Any],
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    utterances = _coerce_utterances(inference_result.get("utterances"))
    emotions = _coerce_predictions(inference_result.get("emotions")) or _aggregate_predictions(
        utterances, "emotions"
    )
    sentiments = _coerce_predictions(
        inference_result.get("sentiments")
    ) or _aggregate_predictions(utterances, "sentiments")
    transcript = (
        inference_result.get("transcript")
        if isinstance(inference_result.get("transcript"), str)
        else None
    ) or _collapse_transcript(utterances)

    debug_payload = {}
    if isinstance(inference_result.get("debug"), dict):
        debug_payload.update(inference_result["debug"])
    if debug:
        debug_payload.update(debug)
    debug_payload["modelVersion"] = model_version
    debug_payload["source"] = "modal"
    debug_payload["utteranceCount"] = len(utterances)

    dominant_emotion = emotions[0]["label"] if emotions else NEUTRAL_EMOTION
    dominant_sentiment = sentiments[0]["label"] if sentiments else NEUTRAL_SENTIMENT

    response: dict[str, Any] = {
        "chunkId": chunk_id,
        "emotions": emotions,
        "sentiments": sentiments,
        "dominantEmotion": dominant_emotion,
        "dominantSentiment": dominant_sentiment,
        "confidence": _average_confidence(utterances),
        "utterances": utterances,
        "debug": debug_payload,
        "modelVersion": model_version,
    }

    if transcript:
        response["transcript"] = transcript

    return response
