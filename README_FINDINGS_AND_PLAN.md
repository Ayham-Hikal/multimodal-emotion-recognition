# MELD Multimodal Project Audit and Improvement Plan

## Scope

This document captures the end-to-end audit findings and an implementation roadmap to improve performance for:

- **Task 1:** Emotion classification (priority)
- **Task 2:** Sentiment classification

The goals are robust, reproducible gains with strong imbalance-aware evaluation, not cosmetic refactors.

## Current Snapshot

- Reported results:
  - Emotion accuracy: **62%** (without imbalance handling) -> **49%** (with current imbalance fix)
  - Sentiment accuracy: **68%** (without imbalance handling) -> **65%** (with current imbalance fix)
- Dataset is imbalanced, with **neutral** dominating emotion labels.
- Existing imbalance strategy appears to degrade performance.

## Repository Audit (What Exists)

### Core pipeline files

- Training entrypoint: `training/train.py`
- Model and trainer logic: `training/models.py`
- Dataset loading and preprocessing: `training/meld_dataset.py`
- SageMaker launcher: `train_sagemaker.py`
- Pipeline docs: `DATA_PIPELINE.md`
- Dataset metadata docs: `dataset/README.txt`

### Data assets discovered

- CSV label files:
  - `dataset/train/train_sent_emo.csv`
  - `dataset/dev/dev_sent_emo.csv`
  - `dataset/test/test_sent_emo.csv`
- No Excel label files were found in this repo snapshot.

### Notable engineering observations

- Multimodal architecture combines text/audio/video features.
- BERT and major visual/audio backbones are largely frozen.
- Loss uses weighted cross-entropy + label smoothing for both tasks.
- Evaluation emphasizes accuracy and weighted precision, but misses key imbalance-safe metrics.

## Data Flow (Current)

1. **Raw labels:** split CSV files (`train/dev/test`).
2. **Input loading:** sample-level multimodal read from dataset paths.
3. **Preprocessing:**
   - Text tokenized with BERT tokenizer.
   - Video frames sampled/resized/padded.
   - Audio extracted via ffmpeg and converted to MelSpectrogram.
4. **Label mapping:** hard-coded emotion/sentiment integer maps.
5. **Batching:** custom collate drops invalid samples (`None`).
6. **Model:** feature extractors -> fusion -> two heads (emotion/sentiment).
7. **Training:** sum of emotion + sentiment CE losses.
8. **Imbalance handling:** class weights computed and passed to CE.
9. **Validation/Test:** accuracy + weighted precision and losses.

## Dataset and Label Findings

### Class imbalance

- Emotion labels are heavily skewed toward **neutral**.
- Minority emotions (for example `fear`, `disgust`) have very low support.
- Sentiment is less skewed than emotion but still imbalanced.

### Data quality and split risks

High-risk areas that require explicit verification in implementation:

- Missing or inconsistent labels in CSVs.
- Duplicate utterances or near-duplicates.
- Potential leakage/overlap patterns across train/dev/test.
- Split distribution drift and non-stratified behavior.
- Silent sample drops in preprocessing (`None` returns), which may bias effective training distribution.

## Why Current Imbalance Fix Likely Hurts

The current drop in performance (emotion especially) is plausibly caused by a combination of:

1. **Aggressive inverse-frequency weighting** over-amplifying noisy/minority classes.
2. **Weighted CE + label smoothing interaction** causing over-regularized targets.
3. **Joint multitask loss coupling** where one task's reweighting destabilizes shared features for both tasks.
4. **Optimization/selection mismatch** (training with one objective, selecting with metrics that do not reflect minority performance well).
5. **Potential bias in weight estimation** when class counts are inferred through sample loading paths that may fail/drop examples.

## Evaluation Gaps (Critical)

To claim meaningful progress on imbalanced data, the current metrics are insufficient.

Missing or not primary today:

- **Macro-F1** (must-have for emotion)
- **Weighted-F1**
- **Balanced accuracy**
- **Per-class precision/recall/F1**
- **Confusion matrix**
- Seed variance / stability reporting

## Root-Cause Hypotheses (Ranked)

1. Imbalance strategy is too blunt and likely over-correcting.
2. Backbone freezing limits adaptation to task-specific cues.
3. No dialogue/context modeling for MELD-style conversational emotion.
4. Preprocessing pipeline has fragile IO and potential sample-loss bias.
5. Experiment tracking is too weak for disciplined model selection.

## Implementation Plan

## Phase 1 - Reliability and Measurement (No architecture risk)

1. Add robust evaluation outputs:
   - macro-F1, weighted-F1, balanced accuracy, per-class metrics, confusion matrix, accuracy.
2. Save structured run artifacts:
   - config, seed, split, metrics JSON, confusion matrix image/table, checkpoint metadata.
3. Compute class distributions directly from labels (not loader side effects).
4. Add deterministic run controls:
   - fixed seeds, deterministic flags where feasible, logging of package/version/hardware.

**Acceptance criteria:** identical config + seed reproduces within expected tolerance; full metric suite available per run.

## Phase 2 - Emotion-first Imbalance Experiments

Run controlled experiments (same data/splits, no test-set model selection):

1. Baseline: no class weighting, CE.
2. CE + class weights (current style) as explicit ablation.
3. CE + label smoothing off/on sweep.
4. Focal loss variants (gamma/alpha sweeps).
5. WeightedRandomSampler without class-weighted loss.
6. Oversampling minority classes with cap controls.
7. Mild undersampling of dominant neutral class (ablation only).
8. Threshold/logit calibration on validation for macro-F1 optimization (where applicable).

**Primary model-selection metric for emotion:** macro-F1 (with per-class recall guardrails).  
**Secondary:** weighted-F1, balanced accuracy, accuracy.

## Phase 3 - Stronger Model Baselines

Emotion first, then sentiment:

1. Compare stronger pretrained text encoders (e.g., RoBERTa/DeBERTa class).
2. Unfreeze top transformer layers with discriminative learning rates.
3. Tune fusion dropout and head capacity.
4. Tune max sequence length, LR schedule, warmup ratio, weight decay, grad clipping.
5. Run seed sweeps to confirm stability.

## Phase 4 - Sentiment Optimization

After emotion strategy stabilizes:

1. Re-check whether sentiment needs any imbalance correction at all.
2. Run at least 3 serious sentiment experiments:
   - baseline CE
   - best emotion-aligned shared setup
   - sentiment-specific objective/sampler adjustment
3. Select by sentiment macro-F1/weighted-F1 with class-level recall checks.

## Phase 5 - Data-Centric Improvements

1. Label audit pass:
   - suspicious samples, potential noise, inconsistent mappings.
2. Duplicate and near-duplicate detection.
3. Leakage checks across splits by robust text similarity + metadata keys.
4. Preprocessing integrity checks:
   - verify dropped-sample rates by class and split.

## Mandatory Experiment Protocol

- Do **not** use test set for model selection.
- Keep train/val/test split boundaries strict.
- Log every failed run and regression with reason.
- Track all runs in a single comparison table.

Recommended experiment table columns:

- experiment_id
- task (emotion/sentiment)
- model/backbone
- loss
- sampler
- class-weight mode
- label smoothing
- lr / batch / epochs / warmup / wd / max_len / dropout / grad_clip
- seed
- val macro-F1 / weighted-F1 / balanced-acc / acc
- test metrics (only for final selected setups)
- notes (failures, instability, data issues)

## Immediate Next Actions (Execution Order)

1. Upgrade evaluation and run logging.
2. Reproduce baseline and current imbalance-fix runs with new metrics.
3. Execute at least 5 emotion experiments (controlled matrix).
4. Pick best emotion configuration by validation macro-F1.
5. Execute at least 3 sentiment experiments.
6. Finalize best joint setup and run locked test evaluation once.

## Expected Deliverables

1. Repository audit summary (this document + linked code pointers).
2. Data imbalance and label quality report with counts and integrity checks.
3. Updated training pipeline code with reproducible configs.
4. Updated evaluation with macro-F1, weighted-F1, balanced accuracy, confusion matrix, per-class metrics.
5. Emotion experiment suite (>= 5 serious experiments).
6. Sentiment experiment suite (>= 3 serious experiments).
7. Best-model recommendation and rationale.
8. Ranked next steps if SOTA-level target remains unmet.

## Notes on "SOTA-level" Ambition

SOTA-like progress on MELD-style emotion tasks typically requires:

- strong pretrained backbones with selective fine-tuning,
- robust imbalance-aware objectives,
- context-aware modeling,
- disciplined experiment tracking and reproducibility.

This plan prioritizes high-leverage, evidence-driven improvements first to avoid overfitting to noisy gains.
