"""FFmpeg helpers for cutting utterance-level sub-clips."""

from __future__ import annotations

import os
import subprocess
import tempfile


class UtteranceSegmentExtractor:
    def extract(self, video_path: str, start_time: float, end_time: float) -> str:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as segment_file:
            segment_path = segment_file.name

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
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

        if not os.path.isfile(segment_path) or os.path.getsize(segment_path) == 0:
            raise ValueError("Segment extraction produced an empty file")
        return segment_path
