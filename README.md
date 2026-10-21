# Backend: Multimodal emotion & sentiment (MELD)

- **Capstone project:** Bahçeşehir University, Faculty of Engineering (2025–2026)
- Team: Khaireddine Arbouch, Ayham Hikal (AI Engineering) · Nasar Mahmud, Mohammad Siyam (Software Engineering)
- Advisors: Prof. Murat Gökşin Bakır, Prof. Duygu Çakır Yenidoğan
- Original repo: [khaireddine-arbouch/Video-Intelligence-Engine](https://github.com/khaireddine-arbouch/Video-Intelligence-Engine)
>
> **My contributions**
> - Deployed the multimodal model as a real-time AWS SageMaker endpoint (inference handlers, endpoint deployment)
> - Set up the cloud infrastructure and S3 storage for training data and uploaded videos
> - Supported experiment tracking, model evaluation, system integration and performance testing
> - Co-authored the project documentation and final report

This directory is the **machine-learning backend** for the capstone: a single
installable Python package (**`signalframe_ml`**) plus thin entrypoints for
**training**, **SageMaker**, and **Modal** inference.

Design goals:

1. **One source of truth** for the model, preprocessing, checkpoint loading,
   HTTP/timeline contracts, and metrics—no duplicated `models.py` trees.
2. **Production-shaped pipelines**: mixed precision, modern optimizers/schedulers,
   optional EMA, structured metrics, and GPU-friendly inference.
3. **Operational clarity**: every stage is documented, env-driven, and testable
   in isolation.

---

## Repository map

| Path | Role |
|------|------|
| `pyproject.toml` | Package metadata for `pip install -e ".[train,infer]"` |
| `signalframe_ml/` | Library code (model, preprocessing, inference engine, training, API contracts) |
| `training/` | MELD dataset loaders, `train.py` CLI, utilities |
| `deployment/` | SageMaker `model_fn` / `predict_fn` handlers + endpoint deploy helper |
| `modal-deployment/` | Modal app (`app.py`) + contract tests |
| `train_sagemaker.py` | SageMaker **training** job launcher |

---

## Quick install

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -U pip setuptools wheel
pip install -e ".[train,infer]"
```

Training extras pull **pandas, tqdm, tensorboard, scikit-learn**. Inference extras
pull **faster-whisper** (and shared deps). FFmpeg must exist on `PATH` for both
training and inference (Modal / SageMaker images install it explicitly).

---

## Model architecture

`signalframe_ml.model.MultimodalSentimentModel` is a **late-fusion** classifier:

| Tower | Backbone | Output dim |
|-------|-----------|------------|
| Text | `transformers.AutoModel` (default `bert-base-uncased`), optional last-layer unfreeze | 128 |
| Video | `torchvision.models.video.r3d_18` (Kinetics-pretrained) | 128 |
| Audio | 1D CNN over 64-bin log-mel (16 kHz mono, hop 512) | 128 |

Concatenated **384-d** features → `Linear → BN → ReLU → Dropout` (**256-d**) →
two heads: **7-way emotion**, **3-way sentiment**.

Training and inference **must** use the same tensorization (30 RGB frames at
224², mel length 300). The shared `signalframe_ml.preprocessing` package enforces
that for inference; the MELD `Dataset` in `training/meld_dataset.py` mirrors it
for training.

---

## Training pipeline (CLI)

Entrypoint: `training/train.py`.

### Reproducibility & throughput

| Feature | Flag / config | Notes |
|---------|----------------|-------|
| Fixed seed | `--seed` | Sets Python, NumPy, torch, CuDNN deterministic |
| TF32 matmul | on when CUDA | Faster matmul on Ampere+ without changing numerics much |
| Mixed precision | default on CUDA; `--no-amp` to disable | `torch.cuda.amp` + `GradScaler` |
| AdamW | optimizer | Weight decay decoupled (vs legacy Adam in older revisions) |
| LR schedule | `--scheduler cosine` (default) or `plateau` | Cosine decay + linear warmup (`--warmup-ratio`) |
| Gradient accumulation | `--gradient-accumulation-steps` | Effective larger batch on limited VRAM |
| EMA weights | `--ema-decay 0.999` | Polyak averaging; **best checkpoint stores EMA weights** when enabled |
| W&B | `--wandb` | Requires `WANDB_PROJECT` (and login) |

### Imbalance & loss

| Feature | CLI |
|---------|-----|
| Focal loss | `--loss-type focal --focal-gamma 2` |
| Class weights (emotion / sentiment) | `--emotion-class-weight-strategy effective_num` (etc.) |
| Weighted sampler | `--sampler-strategy weighted --sampler-task emotion` |
| Label smoothing | `--label-smoothing 0.05` |

### Model selection

`--selection-metric` chooses the validation key for checkpointing (default
`emotion_macro_f1`). `val_loss` is supported (minimization).

### Example local run

```bash
cd training
python train.py \
  --epochs 20 \
  --batch-size 16 \
  --scheduler cosine \
  --warmup-ratio 0.05 \
  --gradient-accumulation-steps 1 \
  --ema-decay 0.999 \
  --loss-type focal \
  --focal-gamma 2.0 \
  --selection-metric emotion_macro_f1 \
  --train-dir ../dataset/train \
  --val-dir ../dataset/dev \
  --test-dir ../dataset/test \
  --model-dir ./outputs
```

Artifacts in `--model-dir`:

* `model.pth` — best checkpoint (`model_state_dict`, metrics, full CLI snapshot)
* `run_config.json` — argparse dump
* `metrics.json` — per-epoch curves + test evaluation
* TensorBoard logs under `runs/` (or SageMaker `/opt/ml/output/tensorboard`)

### SageMaker training

`train_sagemaker.py` uploads the **entire `backend/` tree** so `signalframe_ml` is
available in-container. Configure dataset channels via `S3_*` env vars and run:

```bash
cd backend
python train_sagemaker.py
```

---

## Inference pipeline

High-level stages (identical semantics for Modal + SageMaker):

1. **Segment speech** — faster-whisper with word timestamps.
2. **Cut utterances** — FFmpeg per `[start, end]`.
3. **Tensorize** — shared video / audio / BERT tokenization.
4. **Forward** — fused model, softmax, top-3 labels + confidences per utterance.

Implementation: `signalframe_ml.inference.UtteranceTimelineEngine`.

### SageMaker realtime

* Handler: `deployment/inference.py` (`model_fn` caches `UtteranceTimelineEngine`).
* Package layout: deploy `source_dir` = **`backend/`** so `signalframe_ml` is on
  the import path (see `deployment/deploy_endpoint.py`).
* Request JSON: `{"video_path": "s3://bucket/key.mp4"}` (`Content-Type: application/json`).

### Modal serverless GPU

See `modal-deployment/README.md`. `app.py` bakes `signalframe_ml` into the image
and mounts weights from a Modal Volume.

### HTTP contract (timeline integration)

`signalframe_ml.api.service_contract` normalizes chunk/batch responses and safe
user-facing error messages. **Unit tests:** `modal-deployment/test_service_contract.py`.

---

## Checkpoints & compatibility

`signalframe_ml.checkpoint` understands:

* Raw `state_dict`
* `{ "model_state_dict": … }` (trainer export)
* `{ "state_dict": … }` (legacy)
* BERT key-prefix drift (`text_encoder.bert.*` ↔ `text_encoder.transformer.*`)

---

## Development & tests

```bash
cd backend
pip install -e ".[train,infer]"
pytest modal-deployment/test_service_contract.py -q
cd training && python test_logging.py
```

Full GPU integration tests are intentionally **not** committed here; validate on
Modal or SageMaker with a short public clip.

---

## Operational checklist

1. **Pin versions** — align `torch` / `torchvision` / `torchaudio` / `transformers`
   across training and inference images.
2. **Version the model** — set `MODAL_MODEL_VERSION` (or your SaaS metadata) to the
   same string you log in `run_config.json`.
3. **Calibrate in-domain** — MELD is *Friends* dialogue; macro-F1 matters more than
   headline accuracy for imbalanced emotions.
4. **Watch failure modes** — `failedUtteranceCount` in inference `debug` indicates
   FFmpeg or per-utterance faults; alert if it spikes.

---

## License

Academic / project use. Verify MELD and upstream weight licenses before
redistributing checkpoints or merged binaries.
