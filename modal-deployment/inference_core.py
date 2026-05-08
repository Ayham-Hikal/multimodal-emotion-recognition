import os
import subprocess
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torchaudio
from faster_whisper import WhisperModel
from transformers import AutoTokenizer

from models import MultimodalSentimentModel

EMOTION_MAP = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
    6: "surprise",
}
SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def _load_compatible_state_dict(model: torch.nn.Module, model_path: str, device: torch.device) -> None:
    """Support multiple checkpoint layouts and text encoder naming drift."""
    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    if "model_state_dict" in state_dict and isinstance(state_dict["model_state_dict"], dict):
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format at {model_path}")

    model_keys = set(model.state_dict().keys())
    remapped_state_dict = {}

    for key, value in state_dict.items():
        remapped_key = key
        if key.startswith("text_encoder.bert."):
            candidate = key.replace("text_encoder.bert.", "text_encoder.transformer.", 1)
            if candidate in model_keys:
                remapped_key = candidate
        elif key.startswith("text_encoder.transformer."):
            candidate = key.replace("text_encoder.transformer.", "text_encoder.bert.", 1)
            if candidate in model_keys:
                remapped_key = candidate
        remapped_state_dict[remapped_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
    allowed_prefixes = ("text_encoder.bert.", "text_encoder.transformer.")
    disallowed_missing = [k for k in missing_keys if not k.startswith(allowed_prefixes)]
    disallowed_unexpected = [k for k in unexpected_keys if not k.startswith(allowed_prefixes)]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible with current model architecture. "
            f"Missing keys: {disallowed_missing}. Unexpected keys: {disallowed_unexpected}."
        )


class VideoProcessor:
    def process_video(self, video_path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)
        finally:
            cap.release()

        if not frames:
            raise ValueError(f"No frames extracted from: {video_path}")

        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)


class AudioProcessor:
    def extract_features(self, video_path: str, max_length: int = 300) -> torch.Tensor:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            audio_path = wav_file.name

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    audio_path,
                    "-y",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512,
            )
            mel_spec = mel_spectrogram(waveform)
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < max_length:
                padding = max_length - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :max_length]

            return mel_spec
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)


class VideoUtteranceProcessor:
    def __init__(self) -> None:
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()

    def extract_segment(self, video_path: str, start_time: float, end_time: float) -> str:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as segment_file:
            segment_path = segment_file.name

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-ss",
                str(start_time),
                "-to",
                str(end_time),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-y",
                segment_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if os.path.getsize(segment_path) == 0:
            raise ValueError("Segment extraction produced empty file")
        return segment_path


@dataclass
class InferenceService:
    model_path: str

    def __post_init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultimodalSentimentModel().to(self.device)
        _load_compatible_state_dict(self.model, self.model_path, self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        whisper_device = "cuda" if self.device.type == "cuda" else "cpu"
        compute_type = "float16" if whisper_device == "cuda" else "int8"
        self.transcriber = WhisperModel(
            "base",
            device=whisper_device,
            compute_type=compute_type,
        )

    def predict(self, video_path: str) -> dict:
        segments, _ = self.transcriber.transcribe(video_path, word_timestamps=True)
        transcription_segments = list(segments)
        utterance_processor = VideoUtteranceProcessor()
        predictions = []

        for segment in transcription_segments:
            segment_path = None
            try:
                segment_path = utterance_processor.extract_segment(
                    video_path=video_path,
                    start_time=segment.start,
                    end_time=segment.end,
                )
                video_frames = utterance_processor.video_processor.process_video(segment_path)
                audio_features = utterance_processor.audio_processor.extract_features(segment_path)
                text_inputs = self.tokenizer(
                    segment.text,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )

                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                video_frames = video_frames.unsqueeze(0).to(self.device)
                audio_features = audio_features.unsqueeze(0).to(self.device)

                with torch.inference_mode():
                    outputs = self.model(text_inputs, video_frames, audio_features)
                    emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
                    sentiment_probs = torch.softmax(outputs["sentiments"], dim=1)[0]
                    emotion_values, emotion_indices = torch.topk(emotion_probs, 3)
                    sentiment_values, sentiment_indices = torch.topk(sentiment_probs, 3)

                predictions.append(
                    {
                        "start_time": segment.start,
                        "end_time": segment.end,
                        "text": segment.text,
                        "emotions": [
                            {"label": EMOTION_MAP[idx.item()], "confidence": conf.item()}
                            for idx, conf in zip(emotion_indices, emotion_values)
                        ],
                        "sentiments": [
                            {"label": SENTIMENT_MAP[idx.item()], "confidence": conf.item()}
                            for idx, conf in zip(sentiment_indices, sentiment_values)
                        ],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                predictions.append(
                    {
                        "start_time": getattr(segment, "start", None),
                        "end_time": getattr(segment, "end", None),
                        "text": getattr(segment, "text", ""),
                        "error": str(exc),
                    }
                )
            finally:
                if segment_path and os.path.exists(segment_path):
                    os.remove(segment_path)

        return {"utterances": predictions}
