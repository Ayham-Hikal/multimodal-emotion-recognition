# Modal GPU deployment

Serverless inference lives in `app.py`. It packages the shared library
`signalframe_ml/` into the container image (`PYTHONPATH=/pkg`) so HTTP
wrappers and the GPU worker stay in sync with local training.

## Prerequisites

* Modal account + CLI (`pip install modal`, `modal token new`)
* From repo root or `backend/`, ensure Python can import `signalframe_ml`
  (`pip install -e ./backend` or run Modal commands with `PYTHONPATH` including
  `backend/`).

## Deploy / smoke

```bash
cd backend/modal-deployment
modal deploy app.py          # production
modal run app.py --model-path /path/to/model.pth --video-path /path/to/clip.mp4
```

Redeploy after changes to `app.py` **or** any file under `signalframe_ml/`.

## Environment

| Variable | Purpose |
|----------|---------|
| `MODAL_INFERENCE_AUTH_TOKEN` | When set, `infer_timeline_chunk` requires `Authorization: Bearer …` |
| `MODAL_MODEL_VERSION` | Logical version echoed in JSON (default `meld-multimodal-v1`) |

## Layout

| Component | Role |
|-----------|------|
| `InferenceWorker` | GPU class: loads `model.pth` from the Modal Volume, runs `UtteranceTimelineEngine` |
| `infer_timeline_chunk` | HTTP POST: selection + optional batch chunking (see `signalframe_ml.api.service_contract`) |
| `healthcheck` | Liveness / build identity |
