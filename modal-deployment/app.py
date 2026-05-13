import json
import os
import subprocess
import tempfile
from pathlib import Path

import modal
from fastapi import Header, HTTPException
from pydantic import BaseModel

from service_contract import (
    ModalServiceSettings,
    build_timeline_chunk_response,
    ensure_authorized,
    normalize_source_range,
)

APP_NAME = "multimodal-sentiment-inference"
MODEL_VOLUME_NAME = "multimodal-sentiment-model"
MODEL_MOUNT_DIR = "/vol/model"
DEFAULT_GPU = "A10G"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
BASE_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .apt_install("ffmpeg", "libgl1")
    .pip_install_from_requirements(str(BASE_DIR / "requirements.txt"))
    .pip_install("fastapi[standard]")
    .add_local_file(str(BASE_DIR / "inference_core.py"), "/root/inference_core.py")
    .add_local_file(str(BASE_DIR / "models.py"), "/root/models.py")
    .add_local_file(str(BASE_DIR / "service_contract.py"), "/root/service_contract.py")
)


def _resolve_local_path(raw_path: str) -> str:
    candidate = Path(raw_path)
    if candidate.is_file():
        return str(candidate.resolve())

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.is_file():
        return str(cwd_candidate)

    repo_root_candidate = (Path(__file__).resolve().parent.parent / raw_path).resolve()
    if repo_root_candidate.is_file():
        return str(repo_root_candidate)

    raise FileNotFoundError(
        "File not found. Tried: "
        f"{candidate.resolve()}, {cwd_candidate}, {repo_root_candidate}"
    )


@app.function(
    image=image,
    volumes={MODEL_MOUNT_DIR: model_volume},
    timeout=600,
)
def upload_model(model_bytes: bytes, filename: str = "model.pth") -> str:
    os.makedirs(MODEL_MOUNT_DIR, exist_ok=True)
    target_path = os.path.join(MODEL_MOUNT_DIR, filename)
    with open(target_path, "wb") as f:
        f.write(model_bytes)
    model_volume.commit()
    return target_path


@app.cls(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={MODEL_MOUNT_DIR: model_volume},
    timeout=900,
    scaledown_window=120,
)
class InferenceEndpoint:
    @modal.enter()
    def load(self) -> None:
        from inference_core import InferenceService

        model_path = os.path.join(MODEL_MOUNT_DIR, "model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights missing at {model_path}. "
                "Upload first with `modal run app.py --model-path /path/to/model.pth --video-path /path/to/video.mp4`."
            )
        self.service = InferenceService(model_path=model_path)

    @modal.method()
    def predict_video_bytes(self, video_bytes: bytes, filename: str = "input.mp4") -> dict:
        suffix = Path(filename).suffix if Path(filename).suffix else ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_video:
            tmp_video.write(video_bytes)
            tmp_video_path = tmp_video.name

        try:
            return self.service.predict(tmp_video_path)
        finally:
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)

    @modal.method()
    def predict_url_range(
        self,
        media_url: str,
        source_start_ms: int,
        source_end_ms: int,
    ) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            tmp_video_path = tmp_video.name

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-ss",
                    str(max(0, source_start_ms) / 1000),
                    "-to",
                    str(max(source_start_ms + 1, source_end_ms) / 1000),
                    "-i",
                    media_url,
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    "-movflags",
                    "+faststart",
                    "-y",
                    tmp_video_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return self.service.predict(tmp_video_path)
        finally:
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)


class TimelineChunkRequest(BaseModel):
    mediaUrl: str
    sourceStartMs: int
    sourceEndMs: int
    chunkId: str
    mimeType: str | None = None


@app.function(
    image=image,
    timeout=30,
)
@modal.fastapi_endpoint(method="GET")
def healthcheck() -> dict:
    settings = ModalServiceSettings.from_env()
    return {
        "ok": True,
        "service": APP_NAME,
        "modelVersion": settings.model_version,
        "authEnabled": bool(settings.auth_token),
        "modelVolume": MODEL_VOLUME_NAME,
    }


@app.function(
    image=image,
    timeout=900,
)
@modal.fastapi_endpoint(method="POST")
def infer_timeline_chunk(
    payload: TimelineChunkRequest,
    authorization: str | None = Header(default=None),
) -> dict:
    settings = ModalServiceSettings.from_env()
    try:
        ensure_authorized(settings, authorization)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    source_start_ms, source_end_ms = normalize_source_range(
        payload.sourceStartMs,
        payload.sourceEndMs,
    )
    result = InferenceEndpoint().predict_url_range.remote(
        media_url=payload.mediaUrl,
        source_start_ms=source_start_ms,
        source_end_ms=source_end_ms,
    )

    return build_timeline_chunk_response(
        chunk_id=payload.chunkId,
        model_version=settings.model_version,
        inference_result=result,
    )


@app.local_entrypoint()
def run(video_path: str, model_path: str = "") -> None:
    """
    Example:
      modal run app.py --model-path ../deployment/model/model.pth --video-path /path/to/video.mp4
      modal run app.py --video-path /path/to/video.mp4
    """
    if model_path:
        resolved_model_path = _resolve_local_path(model_path)
        with open(resolved_model_path, "rb") as f:
            target = upload_model.remote(f.read(), filename="model.pth")
        print(f"Uploaded model to: {target}")

    resolved_video_path = _resolve_local_path(video_path)
    with open(resolved_video_path, "rb") as f:
        result = InferenceEndpoint().predict_video_bytes.remote(
            video_bytes=f.read(),
            filename=os.path.basename(resolved_video_path),
        )

    print(json.dumps(result, indent=2))
