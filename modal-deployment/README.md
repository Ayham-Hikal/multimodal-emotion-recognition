# Modal Inference Service

This folder contains the production-facing Modal HTTP service for timeline AI
analysis. It is designed to match the Convex caller in
`frontend/convex/inngest/modalClient.ts`.

## What It Exposes

- `POST infer_timeline_chunk`
  Accepts `schemaVersion: 1` requests in two modes:
  - Single chunk:
    - `mediaUrl`
    - `sourceStartMs`
    - `sourceEndMs`
    - `chunkId`
    - `mimeType`
  - Batched selection:
    - `mediaUrl`
    - `selectionStartMs`
    - `selectionEndMs`
    - `mimeType`
    - `chunks[]` with `chunkId`, `sourceStartMs`, `sourceEndMs`
- `GET healthcheck`
  Returns a lightweight config and deployment health payload.

## Environment

- `MODAL_INFERENCE_AUTH_TOKEN`
  Optional shared bearer token. If set here, Convex must send the same token in
  the `Authorization` header.
- `MODAL_MODEL_VERSION`
  Optional model version label returned in the response debug payload. Defaults
  to `meld-multimodal-v1`.

## Deploy

From this directory:

```bash
modal deploy app.py
```

Redeploy after every change to `app.py`, `inference_core.py`, `models.py`, or
`service_contract.py`. Modal will not pick up local edits until you deploy
again. Treat each deploy as a release artifact: capture the deployed endpoint
URL and the Modal version/build identifier shown by the CLI in the PR or
release checklist so Convex and the runbook always point at the same build.

If your model weights are not already present in the Modal volume, upload them
first or as part of a validation run:

```bash
modal run app.py --model-path ../deployment/model/model.pth --video-path /path/to/sample.mp4
```

## Convex Wiring

After deploy, copy the `infer_timeline_chunk` HTTPS URL into Convex:

```bash
npx convex env set MODAL_INFERENCE_URL <modal-endpoint-url>
```

If auth is enabled on Modal, also mirror the token:

```bash
npx convex env set MODAL_INFERENCE_AUTH_TOKEN <shared-token>
```

After any deploy that changes the response contract or route behavior, re-run:

```bash
npx convex env set MODAL_INFERENCE_URL <modal-endpoint-url>
```

if the URL changed, then trigger one end-to-end analysis job from Convex/Inngest
before calling the release ready.

## Smoke Check

Run these checks after deploy:

```bash
curl <modal-healthcheck-url>
```

Then run one timeline analysis job against a known H.264/AAC test asset and
confirm the workflow reaches `complete-job` in Inngest and stores timeline
segments in Convex. If the response contract changed, include one single-chunk
request and one batch request in your manual verification notes.

## Expected Response Shape

Single-chunk responses return:

- `schemaVersion`
- `chunkId`
- `transcript`
- `emotions`
- `sentiments`
- `dominantEmotion`
- `dominantSentiment`
- `confidence`
- `utterances`
- `debug`
- `modelVersion`

Batch responses return:

- `schemaVersion`
- `results[]` containing the same per-chunk shape
- `debug`
- `modelVersion`

Failures return structured JSON with:

- `schemaVersion`
- `code`
- `retryable`
- `message`
- `detail`

The `message` field is safe for UI copy. Keep operational details in `detail`
and logs, not in user-facing strings.
