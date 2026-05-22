"""
SageMaker inference handlers (`model_fn` / `input_fn` / `predict_fn` / `output_fn`).

The implementation delegates utterance segmentation + multimodal scoring to
:class:`signalframe_ml.inference.UtteranceTimelineEngine` (faster-whisper + shared weights).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import boto3
import torch

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from signalframe_ml.inference import UtteranceTimelineEngine


def install_ffmpeg() -> bool:
    """Best-effort FFmpeg resolution for stripped SageMaker images."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg"):
        return True
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    try:
        import subprocess as sp

        sp.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    except sp.CalledProcessError:
        return False
    return bool(shutil.which("ffmpeg"))


def download_from_s3(s3_uri: str) -> str:
    s3_client = boto3.client("s3")
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        s3_client.download_file(bucket, key, temp_file.name)
        return temp_file.name


def input_fn(request_body: bytes | str, request_content_type: str) -> dict[str, str]:
    if request_content_type == "application/json":
        raw = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
        input_data = json.loads(raw)
        s3_uri = input_data["video_path"]
        return {"video_path": download_from_s3(s3_uri)}
    raise ValueError(f"Unsupported content type: {request_content_type}")


def output_fn(prediction: object, response_content_type: str) -> str:
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")


def model_fn(model_dir: str) -> dict[str, object]:
    if not install_ffmpeg():
        raise RuntimeError("FFmpeg is required for audio/video preprocessing.")

    candidates = [
        os.path.join(model_dir, "model.pth"),
        os.path.join(model_dir, "model", "model.pth"),
    ]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.extend(
        [
            os.path.join(script_dir, model_dir, "model.pth"),
            os.path.join(script_dir, model_dir, "model", "model.pth"),
            os.path.join(script_dir, "model", "model.pth"),
        ]
    )
    model_path = next((p for p in candidates if os.path.exists(p)), None)
    if not model_path:
        raise FileNotFoundError("model.pth not found; tried: " + ", ".join(candidates))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = UtteranceTimelineEngine(model_path, device=device)
    return {"engine": engine}


def predict_fn(input_data: dict[str, str], model_dict: dict[str, object]) -> dict[str, object]:
    engine = model_dict["engine"]
    assert isinstance(engine, UtteranceTimelineEngine)
    video_path = input_data["video_path"]
    try:
        return engine.predict(video_path)
    finally:
        if os.path.isfile(video_path):
            os.remove(video_path)
