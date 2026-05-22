"""Log-mel features aligned with training (64 mels, hop 512, 16 kHz mono)."""

from __future__ import annotations

import os
import subprocess
import tempfile

import torch
import torchaudio


class AudioMelExtractor:
    def __init__(self, *, sample_rate: int = 16000, n_mels: int = 64, n_fft: int = 1024, hop_length: int = 512) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )

    def from_video_path(self, video_path: str, *, max_time_steps: int = 300) -> torch.Tensor:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            audio_path = wav_file.name

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    video_path,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    str(self.sample_rate),
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
            if sample_rate != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(waveform)

            mel_spec = self._mel(waveform)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

            if mel_spec.size(2) < max_time_steps:
                mel_spec = torch.nn.functional.pad(mel_spec, (0, max_time_steps - mel_spec.size(2)))
            else:
                mel_spec = mel_spec[:, :, :max_time_steps]

            return mel_spec
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
