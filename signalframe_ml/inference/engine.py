"""Utterance-level timeline inference: Whisper segmentation + multimodal classifier."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer

from signalframe_ml.checkpoint import assert_compatible, load_weights_into_model
from signalframe_ml.labels import EMOTION_ID_TO_NAME, SENTIMENT_ID_TO_NAME
from signalframe_ml.model import MultimodalSentimentModel
from signalframe_ml.preprocessing import AudioMelExtractor, UtteranceSegmentExtractor, VideoFrameTensorizer

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Hyperparameters for the inference path (defaults match historical training)."""

    text_model_name: str = "bert-base-uncased"
    max_text_length: int = 128
    whisper_model_size: str = "base"
    video_frames: int = 30


class UtteranceTimelineEngine:
    """
    End-to-end engine:

    1. Run faster-whisper with word timestamps.
    2. For each segment, FFmpeg-extract sub-clip, tensorize video/audio/text.
    3. Forward fused model; emit top-k logits as human-readable labels.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: torch.device | None = None,
        engine_config: EngineConfig | None = None,
    ) -> None:
        self.cfg = engine_config or EngineConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MultimodalSentimentModel(text_model_name=self.cfg.text_model_name).to(self.device)
        missing, unexpected = load_weights_into_model(
            self.model,
            model_path,
            map_location=self.device,
            strict=False,
        )
        assert_compatible(missing, unexpected)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.text_model_name)
        whisper_device = "cuda" if self.device.type == "cuda" else "cpu"
        compute_type = "float16" if whisper_device == "cuda" else "int8"
        self.transcriber = WhisperModel(
            self.cfg.whisper_model_size,
            device=whisper_device,
            compute_type=compute_type,
        )

        self._video = VideoFrameTensorizer(max_frames=self.cfg.video_frames)
        self._audio = AudioMelExtractor()
        self._segment = UtteranceSegmentExtractor()

    def predict(self, video_path: str) -> dict[str, Any]:
        started_at = time.perf_counter()
        transcription_started_at = time.perf_counter()
        segments, _ = self.transcriber.transcribe(video_path, word_timestamps=True)
        transcription_segments = list(segments)
        transcription_ms = round((time.perf_counter() - transcription_started_at) * 1000)

        predictions: list[dict[str, Any]] = []
        failed_segments = 0

        for segment in transcription_segments:
            segment_path: str | None = None
            try:
                segment_path = self._segment.extract(
                    video_path,
                    float(segment.start),
                    float(segment.end),
                )
                video_frames = self._video.from_file(segment_path)
                audio_features = self._audio.from_video_path(segment_path)
                text_inputs = self.tokenizer(
                    segment.text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.cfg.max_text_length,
                    return_tensors="pt",
                )
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                video_b = video_frames.unsqueeze(0).to(self.device)
                audio_b = audio_features.unsqueeze(0).to(self.device)

                with torch.inference_mode():
                    outputs = self.model(text_inputs, video_b, audio_b)
                    emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
                    sentiment_probs = torch.softmax(outputs["sentiments"], dim=1)[0]
                    emotion_values, emotion_indices = torch.topk(emotion_probs, 3)
                    sentiment_values, sentiment_indices = torch.topk(sentiment_probs, 3)

                predictions.append(
                    {
                        "start_time": segment.start,
                        "end_time": segment.end,
                        "text": segment.text.strip(),
                        "emotions": [
                            {"label": EMOTION_ID_TO_NAME[int(idx)], "confidence": float(conf)}
                            for idx, conf in zip(emotion_indices, emotion_values)
                        ],
                        "sentiments": [
                            {"label": SENTIMENT_ID_TO_NAME[int(idx)], "confidence": float(conf)}
                            for idx, conf in zip(sentiment_indices, sentiment_values)
                        ],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                failed_segments += 1
                logger.warning(
                    "Utterance inference failed",
                    extra={
                        "start_time": getattr(segment, "start", None),
                        "end_time": getattr(segment, "end", None),
                        "error": str(exc),
                    },
                )
            finally:
                if segment_path and os.path.exists(segment_path):
                    os.remove(segment_path)

        total_ms = round((time.perf_counter() - started_at) * 1000)
        return {
            "utterances": predictions,
            "debug": {
                "preprocessingMs": transcription_ms,
                "inferenceMs": max(0, total_ms - transcription_ms),
                "framesUsed": self.cfg.video_frames if predictions else 0,
                "audioUsed": True,
                "textUsed": True,
                "videoUsed": True,
                "failedUtteranceCount": failed_segments,
            },
        }
