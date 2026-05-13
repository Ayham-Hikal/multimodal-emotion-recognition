# Multimodal Sentiment & Emotion Recognition (MELD)

A multimodal deep learning system that watches, listens to, and reads a person speaking in a video, and predicts:

- **Emotion** out of 7 classes: `anger, disgust, fear, joy, neutral, sadness, surprise`
- **Sentiment** out of 3 classes: `negative, neutral, positive`

It works at the **utterance level** (one prediction per spoken sentence in a video), not per frame and not per full video. A full long video is automatically chopped into utterances using **OpenAI Whisper / faster-whisper**, then each utterance gets its own emotion and sentiment scores.

---

## Table of Contents

1. [What the model actually does](#1-what-the-model-actually-does)
2. [Architecture](#2-architecture)
3. [Inputs and outputs](#3-inputs-and-outputs)
4. [Dataset (MELD)](#4-dataset-meld)
5. [Training pipeline](#5-training-pipeline)
6. [Inference pipeline](#6-inference-pipeline)
7. [Deployment options shipped in this repo](#7-deployment-options-shipped-in-this-repo)
8. [Current performance and known limitations](#8-current-performance-and-known-limitations)
9. [Production wrappers you can build on top](#9-production-wrappers-you-can-build-on-top)
10. [Strong product ideas for studios and agencies](#10-strong-product-ideas-for-studios-and-agencies)
11. [Repository layout](#11-repository-layout)
12. [Quick start](#12-quick-start)

---

## 1. What the model actually does

Given a short video clip of someone speaking, the model jointly answers two questions:

- **What emotion is the speaker expressing right now?** (one of 7 emotions)
- **What is the overall sentiment of what they just said?** (negative / neutral / positive)

It does this by combining three signals at once:

- **What was said** (text from the transcript)
- **How it was said** (audio: prosody, intonation, volume from the mel spectrogram)
- **How they looked while saying it** (video: facial expressions, body language)

Then it fuses those three signals and predicts both labels in one forward pass. This is meaningful because in real conversation, the same sentence ("oh, great") can be sarcasm, anger, or genuine joy depending on tone and face — text alone is not enough.

For long videos (interviews, episodes, podcasts, customer calls), the pipeline:

1. Transcribes the video with Whisper and gets per-utterance start/end timestamps.
2. Cuts each utterance into its own sub-clip with FFmpeg.
3. Runs the multimodal model on each sub-clip.
4. Returns a list of `{start_time, end_time, text, top-3 emotions, top-3 sentiments}` objects.

That output structure is what makes this useful in production — it's a timeline of affect, not a single label for the whole video.

---

## 2. Architecture

`MultimodalSentimentModel` (defined in `training/models.py`) is a three-encoder + late-fusion + two-head network.

### Encoders

| Modality | Backbone | Pretrained on | Default state |
|---|---|---|---|
| Text | `bert-base-uncased` (HuggingFace) | English text | Frozen, optional top-N layer unfreeze |
| Video | `r3d_18` (3D ResNet, torchvision) | Kinetics-400 | Frozen, optional unfreeze |
| Audio | Custom 1D CNN over mel spectrogram | Trained from scratch | Trainable |

Each encoder projects its features to a common **128-d** vector.

### Fusion

The three 128-d vectors are concatenated into a 384-d vector, then passed through:

`Linear(384 → 256) → BatchNorm → ReLU → Dropout(0.3)`

### Heads

Two MLP heads sit on the fused 256-d representation:

- **Emotion head** → `Linear(256 → 64) → ReLU → Dropout(0.2) → Linear(64 → 7)`
- **Sentiment head** → `Linear(256 → 64) → ReLU → Dropout(0.2) → Linear(64 → 3)`

Both heads are trained jointly with a weighted sum of two cross-entropy losses (focal loss is also supported), with discriminative learning rates per parameter group (lower LR for the BERT layers, higher LR for fusion/heads).

---

## 3. Inputs and outputs

### Per-utterance input tensors

| Input | Shape | Notes |
|---|---|---|
| `text_inputs.input_ids` | `[B, 128]` | BERT tokenized, padded/truncated to 128 |
| `text_inputs.attention_mask` | `[B, 128]` | |
| `video_frames` | `[B, 30, 3, 224, 224]` | 30 sampled frames, RGB, normalized to `[0, 1]` |
| `audio_features` | `[B, 1, 64, 300]` | Log-mel spectrogram, 64 mels, 300 time steps, z-normalized |

### Output

```json
{
  "emotions":   [B, 7],   // raw logits
  "sentiments": [B, 3]    // raw logits
}
```

### Production-facing inference output (per utterance)

```json
{
  "start_time": 12.34,
  "end_time": 14.10,
  "text": "I can't believe you did that.",
  "emotions": [
    {"label": "anger",    "confidence": 0.71},
    {"label": "surprise", "confidence": 0.18},
    {"label": "sadness",  "confidence": 0.06}
  ],
  "sentiments": [
    {"label": "negative", "confidence": 0.84},
    {"label": "neutral",  "confidence": 0.12},
    {"label": "positive", "confidence": 0.04}
  ]
}
```

A whole-video prediction is just `{"utterances": [ ... ]}` — a timeline of these objects.

---

## 4. Dataset (MELD)

The model is trained on the **Multimodal EmotionLines Dataset (MELD)** — utterance-level clips from the *Friends* TV series, each with text, audio, video, an emotion label, and a sentiment label.

- ~13,000 utterances across ~1,300 dialogues
- 7 emotion classes, 3 sentiment classes
- Splits: `train`, `dev`, `test` CSVs (`dataset/{split}/{split}_sent_emo.csv`) plus matching `.mp4` clips
- Filename convention: `dia{Dialogue_ID}_utt{Utterance_ID}.mp4`

The dataset is **heavily imbalanced**: `neutral` dominates emotions, and minority classes (`fear`, `disgust`) have very low support. This is the central challenge of the project and the reason imbalance handling (class weighting strategies, focal loss, weighted samplers, label smoothing) is configurable from the training CLI.

For full data details see `DATA_PIPELINE.md` and `dataset/README.txt`.

---

## 5. Training pipeline

Entrypoint: `training/train.py`

Key features:

- **Reproducibility**: fixed seed, deterministic CuDNN flags, full run config saved to `run_config.json`.
- **Imbalance handling** (configurable from CLI):
  - Class weight strategies: `none`, `inverse`, `balanced`, `sqrt_inverse`, `effective_num`
  - Loss types: `cross_entropy` or `focal` (with adjustable `gamma`)
  - Label smoothing
  - `WeightedRandomSampler` driven by either emotion or sentiment counts
- **Discriminative learning rates**: separate LRs for BERT, video/audio encoders, fusion, and heads.
- **Selective unfreezing**: top-N BERT layers, full video backbone, audio CNN — all toggleable.
- **Robust evaluation** (in `models.compute_classification_metrics`):
  - accuracy, balanced accuracy, macro-F1, weighted-F1
  - per-class precision / recall / F1 / support
  - confusion matrix
- **Model selection** by configurable validation metric (default `emotion_macro_f1`), not test set.
- **Logging**: TensorBoard + structured `metrics.json` per run, plus SageMaker-format metric lines for cloud training.

Launch a local run:

```bash
cd training
pip install -r requirements.txt
python train.py \
  --epochs 20 \
  --batch-size 16 \
  --loss-type focal \
  --focal-gamma 2.0 \
  --emotion-class-weight-strategy effective_num \
  --selection-metric emotion_macro_f1 \
  --train-dir ../dataset/train \
  --val-dir ../dataset/dev \
  --test-dir ../dataset/test \
  --model-dir ./outputs
```

Launch a SageMaker training job: `train_sagemaker.py`.

---

## 6. Inference pipeline

There are two inference code paths in this repo:

### `deployment/inference.py` — SageMaker-style

- Implements `model_fn`, `input_fn`, `predict_fn`, `output_fn` for SageMaker endpoints.
- Pulls the input video from S3.
- Uses `openai-whisper` to transcribe + segment.
- Per segment: extracts a sub-clip with FFmpeg, builds the three modality tensors, runs the model, returns top-3 emotion and sentiment labels with confidences.
- Includes a self-installing FFmpeg fallback (handy in stripped containers).

### `modal-deployment/` — Modal serverless GPU

- `app.py` defines a Modal app with:
  - A persistent Modal Volume to hold the model weights.
  - An `upload_model` function so you can push your local `model.pth` into the volume.
  - An `InferenceEndpoint` class (`@modal.cls` with `gpu="A10G"`) that loads the model once on container start (`@modal.enter`) and exposes `predict_video_bytes`.
- Uses `faster-whisper` (faster + lower memory than `openai-whisper`).
- One-line entry from your laptop:
  ```bash
  modal run app.py --model-path ../deployment/model/model.pth --video-path /path/to/clip.mp4
  ```

Both paths produce the same per-utterance JSON shape described in section 3.

---

## 7. Deployment options shipped in this repo

| Path | Best for | Notes |
|---|---|---|
| `deployment/` (SageMaker) | Enterprise AWS shops, VPC requirements, batch jobs | Standard SageMaker handlers, S3 video ingest |
| `modal-deployment/` (Modal) | Fast iteration, pay-per-second GPU, demos, MVPs | Auto-scales to zero; cold starts mitigated by `@modal.enter` |

Both are GPU-friendly (they fall back to CPU if no CUDA device is found).

---

## 8. Current performance and known limitations

From `README_FINDINGS_AND_PLAN.md`:

| Task | Accuracy (no imbalance fix) | Accuracy (current imbalance fix) |
|---|---|---|
| Emotion (7-class) | 62% | 49% |
| Sentiment (3-class) | 68% | 65% |

There is an active, documented improvement plan — the headline issues being:

1. The current class-weighting strategy is too aggressive and over-corrects on noisy minority classes.
2. Backbones are mostly frozen, limiting task-specific adaptation.
3. There is no dialogue/context modeling — every utterance is judged in isolation, which hurts MELD-style conversational emotion.
4. Evaluation has only recently been upgraded to macro-F1 / per-class metrics; older numbers were accuracy-only and hide minority-class collapse.

**What this means for production usage:**

- Treat the model as a **strong signal generator**, not a final verdict. Use top-3 outputs and confidences, not just argmax.
- Calibrate thresholds on your domain data before shipping.
- For best results outside the *Friends* domain (TV interviews, sales calls, social videos), plan a small fine-tuning pass on in-domain labeled clips.

---

## 9. Production wrappers you can build on top

These are concrete API/SDK shapes you can ship around the existing model. Each one is small (days, not months) because the heavy lifting is already done by the inference pipeline.

### 9.1 REST API service

A FastAPI/Flask service in front of `InferenceService`:

- `POST /v1/analyze` — upload a video file → get per-utterance timeline JSON.
- `POST /v1/analyze-url` — pass an S3 / GCS / public URL → server downloads → analyzes.
- `GET /v1/jobs/{id}` — async job polling for long videos.
- `GET /v1/healthz`, `GET /v1/version` — for orchestration.
- Auth via API keys + per-key rate limits.
- Return both raw probabilities and top-3 labels so downstream apps can pick their own thresholds.

### 9.2 Python SDK

A thin `pip install yourbrand-affect` package:

```python
from yourbrand import AffectClient

client = AffectClient(api_key="...")
timeline = client.analyze("interview.mp4")
for u in timeline.utterances:
    print(u.text, u.top_emotion, u.top_sentiment)
timeline.to_srt("interview.affect.srt")   # export
timeline.to_csv("interview.affect.csv")
```

### 9.3 Webhook / async batch worker

For studios with archives:

- Drop a video in an S3 bucket → S3 event triggers → results land in a sibling bucket as JSON + CSV + an SRT subtitle track tagged with emotions.
- Slack/Email notification when a batch completes.
- Idempotent (hash the video; skip re-analysis).

### 9.4 Streaming / near-real-time wrapper

- Sliding-window inference on a live RTMP/WebRTC stream.
- Buffer ~5–10 seconds, run Whisper-stream + the multimodal model, push results over WebSocket.
- Useful for live shows, QA monitoring, and live customer support coaching.

### 9.5 Editor / NLE plugins

- **Adobe Premiere / DaVinci Resolve / Final Cut** extensions that call the API and drop **affect markers** onto the timeline so editors can jump to "all the angry beats" or "all the joyful moments" instantly.

### 9.6 Browser extensions

- Right-click any video on YouTube / Vimeo / Loom → "Analyze emotional arc" → renders an inline emotion timeline.

### 9.7 CLI tool

- `affect analyze input.mp4 --out timeline.json --srt subtitles.srt`
- Useful for power users, studios with custom pipelines, and CI checks for content QA.

### 9.8 No-code dashboard

- Drag-and-drop video uploader.
- Renders:
  - Transcript with each utterance color-coded by emotion.
  - An emotional arc chart (positive vs. negative over time).
  - A pie chart of emotion distribution.
  - "Hot moments" — top-K highest-confidence non-neutral utterances.
- Export PDF report.

### 9.9 Confidence-aware ensemble layer

Wrap the model with:

- **Calibration**: temperature scaling on the validation set so confidence scores are usable as probabilities.
- **Smoothing**: HMM / simple temporal smoothing across consecutive utterances (rare to flip from `joy → fear → joy` in one second).
- **Speaker-conditioning**: use diarization (e.g. pyannote) so each speaker gets their own affect track.
- **Fallback rules**: if all three modalities disagree strongly, surface "uncertain" instead of forcing a label.

### 9.10 MLOps surface

- A model registry entry (MLflow / SageMaker Model Registry) with the metrics from `metrics.json`.
- An A/B routing layer so you can ship new checkpoints behind a flag.
- Drift monitoring: log emotion distribution per customer per week and alert on big shifts (often a sign the model is degrading on new content types).

---

## 10. Repository layout

```
.
├── dataset/                 # MELD CSVs and video clip splits
├── training/
│   ├── train.py             # Training entrypoint with full CLI
│   ├── models.py            # MultimodalSentimentModel + MultimodalTrainer + metrics
│   ├── meld_dataset.py      # Dataset, preprocessing, dataloaders, samplers
│   ├── experiment_matrix.json
│   ├── install_ffmpeg.py
│   └── requirements.txt
├── deployment/              # SageMaker-style inference handlers
│   ├── deploy_endpoint.py
│   ├── inference.py
│   ├── models.py
│   └── requirements.txt
├── modal-deployment/        # Serverless GPU inference on Modal
│   ├── app.py
│   ├── inference_core.py
│   ├── models.py
│   └── requirements.txt
├── train_sagemaker.py       # SageMaker job launcher
├── DATA_PIPELINE.md         # Per-modality preprocessing details
├── README_FINDINGS_AND_PLAN.md  # Audit + improvement roadmap
└── README.md                # (this file)
```