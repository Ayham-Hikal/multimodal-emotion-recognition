# Modal Inference Service

This folder contains the production-facing Modal HTTP service for timeline AI
analysis. It is designed to match the Convex caller in
`frontend/convex/inngest/modalClient.ts`.

## What It Exposes

- `POST infer_timeline_chunk`
  Accepts a JSON payload with:
  - `mediaUrl`
  - `sourceStartMs`
  - `sourceEndMs`
  - `chunkId`
  - `mimeType`
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

## Expected Response Shape

The service returns:

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

That keeps the response aligned with Convex normalization while still carrying
extra observability fields for production debugging.
