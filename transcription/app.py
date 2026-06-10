"""
GPU Whisper transcription on Modal (faster-whisper).

Deploy:
  cd infrastructure/modal-transcribe
  pip install modal
  modal deploy app.py

Then set Cloudflare Pages env TRANSCRIBE_UPSTREAM_URL to the printed https URL
(ends with .modal.run/) so /api/transcribe proxies to this service.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi[standard]==0.115.6",
        "faster-whisper==1.1.0",
        "python-multipart==0.0.9",
    )
)

app = modal.App("signalsframe-transcribe")


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    scaledown_window=120,
    max_containers=10,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def transcribe_api() -> Any:
    from fastapi import FastAPI, File, HTTPException, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from faster_whisper import WhisperModel

        app.state.model = WhisperModel("base", device="cuda", compute_type="float16")
        yield
        app.state.model = None

    web_app = FastAPI(lifespan=lifespan)
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.post("/")
    async def transcribe(
        request: Request,
        audio: UploadFile = File(..., description="WAV (or other ffmpeg-readable) clip"),
    ) -> dict[str, Any]:
        body = await audio.read()
        if len(body) < 64:
            return {
                "text": "",
                "word_count": 0,
                "words": [],
                "language": "en",
                "target_language": None,
                "duration": 0.0,
            }

        model = getattr(request.app.state, "model", None)
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
        tmp_path: str | None = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.write(fd, body)
            os.close(fd)

            segments, info = model.transcribe(
                tmp_path,
                word_timestamps=True,
                vad_filter=True,
            )

            words_out: list[dict[str, Any]] = []
            text_chunks: list[str] = []
            for seg in segments:
                t = (seg.text or "").strip()
                if t:
                    text_chunks.append(t)
                if seg.words:
                    for w in seg.words:
                        token = (w.word or "").strip()
                        if not token:
                            continue
                        words_out.append(
                            {
                                "word": token,
                                "start": float(w.start),
                                "end": float(w.end),
                            }
                        )

            full_text = " ".join(text_chunks).strip()
            return {
                "text": full_text,
                "word_count": len(words_out),
                "words": words_out,
                "language": getattr(info, "language", None) or "en",
                "target_language": None,
                "duration": float(getattr(info, "duration", 0.0) or 0.0),
            }
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return web_app
