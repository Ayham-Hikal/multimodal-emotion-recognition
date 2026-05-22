"""RGB frame sampling for the 3D video tower."""

from __future__ import annotations

import cv2
import numpy as np
import torch


class VideoFrameTensorizer:
    """Sample up to ``max_frames`` resized RGB frames → ``[T, 3, H, W]`` float tensor in ``[0, 1]``."""

    def __init__(self, *, height: int = 224, width: int = 224, max_frames: int = 30) -> None:
        self.height = height
        self.width = width
        self.max_frames = max_frames

    def from_file(self, video_path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        frames: list[np.ndarray] = []
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            while len(frames) < self.max_frames and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (self.width, self.height))
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
        finally:
            cap.release()

        if not frames:
            raise ValueError(f"No frames extracted from: {video_path}")

        if len(frames) < self.max_frames:
            frames.extend([np.zeros_like(frames[0])] * (self.max_frames - len(frames)))
        else:
            frames = frames[: self.max_frames]

        return torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).float()
