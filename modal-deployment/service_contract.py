from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from os import environ
from typing import Any


DEFAULT_MODAL_MODEL_VERSION = "meld-multimodal-v1"
MODAL_SCHEMA_VERSION = 1
NEUTRAL_EMOTION = "neutral"
NEUTRAL_SENTIMENT = "neutral"

SAFE_ERROR_MESSAGES = {
    "DOWNLOAD_FAILED": "We couldn't access the source media for analysis.",
    "FFMPEG_EXTRACT_FAILED": "We couldn't prepare this clip for analysis.",
    "INFERENCE_FAILED": "The AI model couldn't finish this analysis.",
    "VALIDATION_FAILED": "This selection couldn't be analyzed.",
    "TIMEOUT": "The AI analysis took too long to finish.",
    "HTTP_ERROR": "The AI analysis service returned an unexpected response.",
}


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


def _build_debug_payload(
    *,
    inference_result: dict[str, Any],
    model_version: str,
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    debug_payload: dict[str, Any] = {}
    if isinstance(inference_result.get("debug"), dict):
        debug_payload.update(inference_result["debug"])
    if debug:
        debug_payload.update(debug)
    debug_payload["modelVersion"] = model_version
    debug_payload["source"] = "modal"
    return debug_payload


def build_modal_error_payload(
    *,
    code: str,
    detail: str,
    retryable: bool,
    message: str | None = None,
) -> dict[str, Any]:
    return {
        "schemaVersion": MODAL_SCHEMA_VERSION,
        "code": code,
        "retryable": retryable,
        "message": message or SAFE_ERROR_MESSAGES.get(code, SAFE_ERROR_MESSAGES["INFERENCE_FAILED"]),
        "detail": detail,
    }


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

    debug_payload = _build_debug_payload(
        inference_result=inference_result,
        model_version=model_version,
        debug=debug,
    )
    debug_payload["utteranceCount"] = len(utterances)

    dominant_emotion = emotions[0]["label"] if emotions else NEUTRAL_EMOTION
    dominant_sentiment = sentiments[0]["label"] if sentiments else NEUTRAL_SENTIMENT

    response: dict[str, Any] = {
        "schemaVersion": MODAL_SCHEMA_VERSION,
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


def _clip_utterance_to_chunk_window(
    utterance: dict[str, Any],
    *,
    chunk_start_s: float,
    chunk_end_s: float,
) -> dict[str, Any] | None:
    start_time = utterance.get("start_time")
    end_time = utterance.get("end_time")
    if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
        return None

    clipped_start = max(float(start_time), chunk_start_s)
    clipped_end = min(float(end_time), chunk_end_s)
    if clipped_end <= clipped_start:
        return None

    clipped = {
        "start_time": round(clipped_start - chunk_start_s, 3),
        "end_time": round(clipped_end - chunk_start_s, 3),
        "emotions": utterance["emotions"],
        "sentiments": utterance["sentiments"],
    }
    if isinstance(utterance.get("text"), str) and utterance["text"].strip():
        clipped["text"] = utterance["text"].strip()

    return clipped


def build_timeline_batch_response(
    *,
    model_version: str,
    selection_source_start_ms: int,
    chunks: list[dict[str, Any]],
    inference_result: dict[str, Any],
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    utterances = _coerce_utterances(inference_result.get("utterances"))
    results: list[dict[str, Any]] = []

    for chunk in chunks:
        chunk_id = chunk.get("chunkId")
        source_start_ms = chunk.get("sourceStartMs")
        source_end_ms = chunk.get("sourceEndMs")
        if not isinstance(chunk_id, str):
            continue
        if not isinstance(source_start_ms, (int, float)) or not isinstance(
            source_end_ms, (int, float)
        ):
            continue

        chunk_start_s = (float(source_start_ms) - float(selection_source_start_ms)) / 1000
        chunk_end_s = (float(source_end_ms) - float(selection_source_start_ms)) / 1000
        chunk_utterances = [
            clipped
            for utterance in utterances
            if (clipped := _clip_utterance_to_chunk_window(
                utterance,
                chunk_start_s=chunk_start_s,
                chunk_end_s=chunk_end_s,
            ))
            is not None
        ]

        results.append(
            build_timeline_chunk_response(
                chunk_id=chunk_id,
                model_version=model_version,
                inference_result={"utterances": chunk_utterances},
                debug=debug,
            )
        )

    top_level_debug = _build_debug_payload(
        inference_result=inference_result,
        model_version=model_version,
        debug=debug,
    )
    top_level_debug["chunkCount"] = len(results)

    return {
        "schemaVersion": MODAL_SCHEMA_VERSION,
        "results": results,
        "debug": top_level_debug,
        "modelVersion": model_version,
    }
