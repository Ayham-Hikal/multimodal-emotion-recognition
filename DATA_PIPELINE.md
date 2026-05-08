# Data Overview and Processing Pipeline

This project uses the MELD-style multimodal dataset with one CSV row per utterance and a matching utterance-level video clip.

## Dataset Structure

- Main training metadata file: `dataset/train/train_sent_emo.csv`
- Video clips for each utterance: `dataset/train/train_splits/`
- Key CSV columns include:
  - `Utterance` (text)
  - `Emotion` (7 classes: anger, disgust, fear, joy, neutral, sadness, surprise)
  - `Sentiment` (3 classes: negative, neutral, positive)
  - `Dialogue_ID`, `Utterance_ID` (used to build video filename: `dia{Dialogue_ID}_utt{Utterance_ID}.mp4`)

## DataLoader Processing (`training /meld_dataset.py`)

For each sample, `MELDDataset.__getitem__` does:

1. **Text**
   - Tokenizes `Utterance` with `bert-base-uncased` tokenizer
   - Uses `max_length=128`, truncation, and max-length padding
   - Returns `input_ids` and `attention_mask`

2. **Video**
   - Loads video with OpenCV
   - Reads up to 30 frames, resizes each to `224x224`, normalizes pixel values to `[0, 1]`
   - Pads with zero-frames if fewer than 30 frames
   - Final tensor shape per sample: `[30, 3, 224, 224]`

3. **Audio**
   - Extracts mono audio at `16 kHz` via `ffmpeg`
   - Computes mel spectrogram (`n_mels=64`, `n_fft=1024`, `hop_length=512`)
   - Normalizes spectrogram
   - Pads/truncates time dimension to 300
   - Final tensor shape per sample: `[1, 64, 300]`

4. **Labels**
   - Maps emotion and sentiment strings to integer class IDs

`collate_fn` filters out failed samples (`None`) and uses default PyTorch collation for batching.

## Model Processing (`training /models.py`)

`MultimodalSentimentModel` has three frozen encoders, each projected to 128-d features:

- **TextEncoder**: frozen BERT (`bert-base-uncased`) -> `pooler_output` -> linear projection to 128
- **VideoEncoder**: frozen pretrained `r3d_18` -> modified FC head to 128  
  (input is transposed from `[B, T, C, H, W]` to `[B, C, T, H, W]`)
- **AudioEncoder**: frozen 1D CNN stack over mel features -> projection to 128

The three 128-d vectors are concatenated (`384-d`), passed through a fusion MLP, then fed into two heads:

- **Emotion classifier**: outputs 7 logits
- **Sentiment classifier**: outputs 3 logits

So the end-to-end flow is:
**CSV row + video clip -> modality-specific preprocessing -> batched tensors -> multimodal fusion -> emotion and sentiment predictions**.
