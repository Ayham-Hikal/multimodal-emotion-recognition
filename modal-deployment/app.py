"""
Modal serverless GPU deployment for utterance-level multimodal affect inference.

Architecture
------------

* **HTTP layer** — thin FastAPI-style endpoints (Modal ``@modal.fastapi_endpoint``) for
  health checks and timeline-shaped JSON used by the SaaS control plane.
* **GPU worker** — ``InferenceWorker`` loads PyTorch weights once per container
  (``@modal.enter``) and runs FFmpeg + faster-whisper + the fused multimodal model.
* **Artifacts** — weights live on a Modal ``Volume`` mounted at ``MODEL_MOUNT_DIR``.

Environment
-----------

``MODAL_INFERENCE_AUTH_TOKEN``
    When set, ``infer_timeline_chunk`` requires ``Authorization: Bearer …``.
``MODAL_MODEL_VERSION``
    Logical version string echoed back in API payloads (defaults to ``meld-multimodal-v1``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import modal
from fastapi import Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

_BACKEND_ROOT = Path(__file__).resolve().parents[1]

if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from signalframe_ml.api.service_contract import (
    ModalServiceSettings,
    build_modal_error_payload,
    build_timeline_batch_response,
    build_timeline_chunk_response,
    ensure_authorized,
    normalize_source_range,
)
from signalframe_ml.inference import UtteranceTimelineEngine

APP_NAME = "multimodal-sentiment-inference"
MODEL_VOLUME_NAME = "multimodal-sentiment-model"
MODEL_MOUNT_DIR = "/vol/model"
DEFAULT_GPU = "A10G"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
_MODAL_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .apt_install("ffmpeg", "libgl1")
    .pip_install(
        "modal",
        "setuptools<81",
        "wheel",
        "transformers==4.47.1",
        "faster-whisper==1.1.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "opencv-python-headless==4.10.0.84",
        "numpy==1.26.4",
        "soundfile==0.12.1",
        "ffmpeg-python==0.2.0",
        "fastapi[standard]",
    )
    .add_local_dir(str(_BACKEND_ROOT / "signalframe_ml"), "/pkg/signalframe_ml")
    .env({"PYTHONPATH": "/pkg"})
)


class TimelineBatchChunkRequest(BaseModel):
    chunkId: str
    sourceStartMs: int
    sourceEndMs: int


class TimelineInferenceRequest(BaseModel):
    schemaVersion: int | None = None
    mode: str | None = None
    mediaUrl: str
    sourceStartMs: int | None = None
    sourceEndMs: int | None = None
    selectionStartMs: int | None = None
    selectionEndMs: int | None = None
    chunkId: str | None = None
    mimeType: str | None = None
    chunks: list[TimelineBatchChunkRequest] | None = None


def _resolve_local_path(raw_path: str) -> str:
    candidate = Path(raw_path)
    if candidate.is_file():
        return str(candidate.resolve())

    cwd_candidate = (Path.cwd() / raw_path).resolve()
    if cwd_candidate.is_file():
        return str(cwd_candidate)

    repo_root_candidate = (_MODAL_DIR.parent / raw_path).resolve()
    if repo_root_candidate.is_file():
        return str(repo_root_candidate)

    raise FileNotFoundError(
        "File not found. Tried: " f"{candidate.resolve()}, {cwd_candidate}, {repo_root_candidate}"
    )


def _redact_urls(value: str) -> str:
    return re.sub(r"https?://\S+", "<redacted-url>", value)


def _error_detail_from_exception(exc: Exception) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    return _redact_urls(detail)


def _error_status_code(code: str) -> int:
    if code == "UNAUTHORIZED":
        return 401
    if code == "VALIDATION_FAILED":
        return 400
    if code == "DOWNLOAD_FAILED":
        return 502
    if code == "FFMPEG_EXTRACT_FAILED":
        return 422
    if code == "SERVICE_UNAVAILABLE":
        return 503
    return 500


def _json_error_response(*, code: str, detail: str, retryable: bool) -> JSONResponse:
    return JSONResponse(
        status_code=_error_status_code(code),
        content=build_modal_error_payload(code=code, detail=detail, retryable=retryable),
    )


def _download_media_to_path(media_url: str, destination_path: str) -> None:
    request = urllib.request.Request(
        media_url,
        headers={"User-Agent": "signalframe-modal-inference/2.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, open(destination_path, "wb") as output_file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output_file.write(chunk)


def _extract_time_range_to_path(
    input_path: str,
    output_path: str,
    source_start_ms: int,
    source_end_ms: int,
) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(max(0, source_start_ms) / 1000),
        "-to",
        str(max(source_start_ms + 1, source_end_ms) / 1000),
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-y",
        output_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr or stdout or "ffmpeg exited with no diagnostic output."
        raise RuntimeError(f"ffmpeg failed while extracting the media range: {details}")


@app.function(image=image, volumes={MODEL_MOUNT_DIR: model_volume}, timeout=600)
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
class InferenceWorker:
    @modal.enter()
    def load(self) -> None:
        model_path = os.path.join(MODEL_MOUNT_DIR, "model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights missing at {model_path}. "
                "Upload with `modal run modal-deployment/app.py --model-path … --video-path …`."
            )
        self.engine = UtteranceTimelineEngine(model_path)

    @modal.method()
    def predict_video_bytes(self, video_bytes: bytes, filename: str = "input.mp4") -> dict:
        suffix = Path(filename).suffix if Path(filename).suffix else ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_video:
            tmp_video.write(video_bytes)
            tmp_video_path = tmp_video.name
        try:
            return self.engine.predict(tmp_video_path)
        finally:
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)

    @modal.method()
    def predict_url_selection(
        self,
        media_url: str,
        selection_start_ms: int,
        selection_end_ms: int,
    ) -> dict:
        tmp_input = tempfile.NamedTemporaryFile(suffix=".source.mp4", delete=False)
        tmp_input_path = tmp_input.name
        tmp_input.close()
        tmp_selection = tempfile.NamedTemporaryFile(suffix=".selection.mp4", delete=False)
        tmp_selection_path = tmp_selection.name
        tmp_selection.close()
        try:
            download_started_at = time.perf_counter()
            try:
                _download_media_to_path(media_url, tmp_input_path)
            except Exception as exc:  # noqa: BLE001
                return {
                    "error": build_modal_error_payload(
                        code="DOWNLOAD_FAILED",
                        detail=_error_detail_from_exception(exc),
                        retryable=True,
                    )
                }
            download_ms = round((time.perf_counter() - download_started_at) * 1000)

            ffmpeg_started_at = time.perf_counter()
            try:
                _extract_time_range_to_path(
                    input_path=tmp_input_path,
                    output_path=tmp_selection_path,
                    source_start_ms=selection_start_ms,
                    source_end_ms=selection_end_ms,
                )
            except Exception as exc:  # noqa: BLE001
                return {
                    "error": build_modal_error_payload(
                        code="FFMPEG_EXTRACT_FAILED",
                        detail=_error_detail_from_exception(exc),
                        retryable=False,
                    )
                }
            ffmpeg_ms = round((time.perf_counter() - ffmpeg_started_at) * 1000)

            try:
                result = self.engine.predict(tmp_selection_path)
            except Exception as exc:  # noqa: BLE001
                return {
                    "error": build_modal_error_payload(
                        code="INFERENCE_FAILED",
                        detail=_error_detail_from_exception(exc),
                        retryable=True,
                    )
                }

            debug = result.setdefault("debug", {})
            if isinstance(debug, dict):
                debug["downloadMs"] = download_ms
                debug["ffmpegMs"] = ffmpeg_ms
                debug["selectionDurationMs"] = max(0, selection_end_ms - selection_start_ms)

            return {"result": result}
        finally:
            if os.path.exists(tmp_input_path):
                os.remove(tmp_input_path)
            if os.path.exists(tmp_selection_path):
                os.remove(tmp_selection_path)


def _normalize_batch_chunks(
    *,
    chunks: list[TimelineBatchChunkRequest],
    selection_start_ms: int,
    selection_end_ms: int,
) -> tuple[list[dict[str, int | str]] | None, JSONResponse | None]:
    normalized_chunks: list[dict[str, int | str]] = []
    for chunk in chunks:
        chunk_start_ms, chunk_end_ms = normalize_source_range(chunk.sourceStartMs, chunk.sourceEndMs)
        if chunk_start_ms < selection_start_ms or chunk_end_ms > selection_end_ms:
            return None, _json_error_response(
                code="VALIDATION_FAILED",
                detail="Chunk ranges must stay within the requested selection window.",
                retryable=False,
            )
        normalized_chunks.append(
            {
                "chunkId": chunk.chunkId,
                "sourceStartMs": chunk_start_ms,
                "sourceEndMs": chunk_end_ms,
            }
        )
    return normalized_chunks, None


@app.function(image=image, timeout=30)
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


@app.function(image=image, timeout=900)
@modal.fastapi_endpoint(method="POST")
def infer_timeline_chunk(
    payload: TimelineInferenceRequest,
    authorization: str | None = Header(default=None),
) -> dict | JSONResponse:
    settings = ModalServiceSettings.from_env()
    try:
        ensure_authorized(settings, authorization)
    except PermissionError as exc:
        return _json_error_response(
            code="UNAUTHORIZED",
            detail=_error_detail_from_exception(exc),
            retryable=False,
        )

    if payload.chunks:
        if payload.selectionStartMs is None or payload.selectionEndMs is None:
            return _json_error_response(
                code="VALIDATION_FAILED",
                detail="Batch inference requests must include selectionStartMs and selectionEndMs.",
                retryable=False,
            )
        selection_start_ms, selection_end_ms = normalize_source_range(
            payload.selectionStartMs,
            payload.selectionEndMs,
        )
        normalized_chunks, validation_error = _normalize_batch_chunks(
            chunks=payload.chunks,
            selection_start_ms=selection_start_ms,
            selection_end_ms=selection_end_ms,
        )
        if validation_error is not None:
            return validation_error

        remote_response = InferenceWorker().predict_url_selection.remote(
            media_url=payload.mediaUrl,
            selection_start_ms=selection_start_ms,
            selection_end_ms=selection_end_ms,
        )
        if isinstance(remote_response, dict) and isinstance(remote_response.get("error"), dict):
            error_payload = remote_response["error"]
            return JSONResponse(
                status_code=_error_status_code(str(error_payload.get("code"))),
                content=error_payload,
            )

        inference_result = remote_response.get("result", {}) if isinstance(remote_response, dict) else {}
        return build_timeline_batch_response(
            model_version=settings.model_version,
            selection_source_start_ms=selection_start_ms,
            chunks=normalized_chunks or [],
            inference_result=inference_result,
            debug=inference_result.get("debug") if isinstance(inference_result, dict) else None,
        )

    if payload.chunkId is None or payload.sourceStartMs is None or payload.sourceEndMs is None:
        return _json_error_response(
            code="VALIDATION_FAILED",
            detail="Single-chunk inference requests must include chunkId, sourceStartMs, and sourceEndMs.",
            retryable=False,
        )

    source_start_ms, source_end_ms = normalize_source_range(payload.sourceStartMs, payload.sourceEndMs)
    remote_response = InferenceWorker().predict_url_selection.remote(
        media_url=payload.mediaUrl,
        selection_start_ms=source_start_ms,
        selection_end_ms=source_end_ms,
    )
    if isinstance(remote_response, dict) and isinstance(remote_response.get("error"), dict):
        error_payload = remote_response["error"]
        return JSONResponse(
            status_code=_error_status_code(str(error_payload.get("code"))),
            content=error_payload,
        )

    inference_result = remote_response.get("result", {}) if isinstance(remote_response, dict) else {}
    return build_timeline_chunk_response(
        chunk_id=payload.chunkId,
        model_version=settings.model_version,
        inference_result=inference_result,
    )


@app.local_entrypoint()
def run(video_path: str, model_path: str = "") -> None:
    """
    Local smoke test::

        modal run modal-deployment/app.py --model-path ../outputs/model.pth --video-path clip.mp4
    """
    if model_path:
        resolved_model_path = _resolve_local_path(model_path)
        with open(resolved_model_path, "rb") as f:
            target = upload_model.remote(f.read(), filename="model.pth")
        print(f"Uploaded model to: {target}")

    resolved_video_path = _resolve_local_path(video_path)
    with open(resolved_video_path, "rb") as f:
        result = InferenceWorker().predict_video_bytes.remote(
            video_bytes=f.read(),
            filename=os.path.basename(resolved_video_path),
        )

    print(json.dumps(result, indent=2))
