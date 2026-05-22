"""Training loop: AMP, discriminative LRs, plateau or cosine+warmup, optional EMA, optional W&B."""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from signalframe_ml.labels import EMOTION_NAMES, SENTIMENT_NAMES
from signalframe_ml.training.metrics import (
    FocalLoss,
    compute_classification_metrics,
    compute_class_weights,
)


class ModelEMA:
    """Exponential moving average of floating tensors in ``state_dict``."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            if torch.is_tensor(v) and v.dtype in (torch.float32, torch.float16, torch.bfloat16):
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}


class MultimodalTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        *,
        config: dict[str, Any] | None = None,
        total_epochs: int = 1,
    ) -> None:
        self.config = config or {}
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        print("\nDataset sizes:")
        print(f"Training samples: {train_size:,}")
        print(f"Validation samples: {val_size:,}")
        print(f"Batches per epoch: {len(train_loader):,}")

        timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
        base_dir = "/opt/ml/output/tensorboard" if "SM_MODEL_DIR" in os.environ else "runs"
        log_dir = f"{base_dir}/run_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        text_lr = float(self.config.get("text_lr", 8e-6))
        encoder_lr = float(self.config.get("encoder_lr", 8e-5))
        head_lr = float(self.config.get("head_lr", 5e-4))
        weight_decay = float(self.config.get("weight_decay", 1e-5))

        param_groups = [
            {"params": [p for p in model.text_encoder.parameters() if p.requires_grad], "lr": text_lr},
            {"params": [p for p in model.video_encoder.parameters() if p.requires_grad], "lr": encoder_lr},
            {"params": [p for p in model.audio_encoder.parameters() if p.requires_grad], "lr": encoder_lr},
            {"params": list(model.fusion_layer.parameters()), "lr": head_lr},
            {"params": list(model.emotion_classifier.parameters()), "lr": head_lr},
            {"params": list(model.sentiment_classifier.parameters()), "lr": head_lr},
        ]
        param_groups = [g for g in param_groups if len(g["params"]) > 0]
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        scheduler_type = str(self.config.get("scheduler", "cosine")).lower()
        accum = max(1, int(self.config.get("gradient_accumulation_steps", 1)))
        batches_per_epoch = max(1, len(train_loader))
        optimizer_steps_per_epoch = max(1, (batches_per_epoch + accum - 1) // accum)
        total_steps = max(1, int(total_epochs) * optimizer_steps_per_epoch)
        warmup_ratio = float(self.config.get("warmup_ratio", 0.05))
        warmup_steps = max(1, int(total_steps * warmup_ratio))

        if scheduler_type == "cosine":

            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
            self._scheduler_step_batch = True
        else:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=2)
            self._scheduler_step_batch = False

        self.current_train_losses: dict[str, float] | None = None

        print("\nCalculating class weights...")
        emotion_weights, sentiment_weights = compute_class_weights(
            train_loader.dataset,
            emotion_strategy=self.config.get("emotion_class_weight_strategy", "none"),
            sentiment_strategy=self.config.get("sentiment_class_weight_strategy", "none"),
            effective_beta=float(self.config.get("effective_num_beta", 0.999)),
        )

        device = next(model.parameters()).device
        self.emotion_weights = emotion_weights.to(device) if emotion_weights is not None else None
        self.sentiment_weights = sentiment_weights.to(device) if sentiment_weights is not None else None

        label_smoothing = float(self.config.get("label_smoothing", 0.0))
        loss_type = self.config.get("loss_type", "ce")
        focal_gamma = float(self.config.get("focal_gamma", 2.0))

        if loss_type == "ce":
            self.emotion_criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                weight=self.emotion_weights,
            )
            self.sentiment_criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                weight=self.sentiment_weights,
            )
        elif loss_type == "focal":
            self.emotion_criterion = FocalLoss(
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
                weight=self.emotion_weights,
            )
            self.sentiment_criterion = FocalLoss(
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
                weight=self.sentiment_weights,
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        self.emotion_loss_weight = float(self.config.get("emotion_loss_weight", 1.0))
        self.sentiment_loss_weight = float(self.config.get("sentiment_loss_weight", 1.0))
        self.grad_clip = float(self.config.get("grad_clip", 1.0))
        self.accum_steps = max(1, int(self.config.get("gradient_accumulation_steps", 1)))
        self.use_amp = bool(self.config.get("use_amp", torch.cuda.is_available()))
        self.scaler = GradScaler(enabled=self.use_amp)

        ema_decay = float(self.config.get("ema_decay", 0.0))
        self.ema: ModelEMA | None = ModelEMA(model, ema_decay) if ema_decay > 0.0 else None

        self._wandb = None
        if self.config.get("wandb", False):
            try:
                import wandb  # type: ignore[import-not-found]

                wandb.init(project=os.environ.get("WANDB_PROJECT", "signalframe-ml"), config=self.config)
                self._wandb = wandb
            except Exception as exc:  # noqa: BLE001
                print(f"W&B init skipped: {exc}")

    def log_metrics(self, losses: dict[str, float], metrics: dict[str, Any] | None = None, phase: str = "train") -> None:
        if phase == "train":
            self.current_train_losses = losses
        else:
            assert self.current_train_losses is not None
            self.writer.add_scalar("loss/total/train", self.current_train_losses["total"], self.global_step)
            self.writer.add_scalar("loss/total/val", losses["total"], self.global_step)
            self.writer.add_scalar("loss/emotion/train", self.current_train_losses["emotion"], self.global_step)
            self.writer.add_scalar("loss/emotion/val", losses["emotion"], self.global_step)
            self.writer.add_scalar("loss/sentiment/train", self.current_train_losses["sentiment"], self.global_step)
            self.writer.add_scalar("loss/sentiment/val", losses["sentiment"], self.global_step)

        if metrics:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{phase}/{name}", value, self.global_step)

        if self._wandb is not None and metrics and phase == "val":
            self._wandb.log({f"val/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}, step=self.global_step)

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        running_loss = {"total": 0.0, "emotion": 0.0, "sentiment": 0.0}
        batches = 0
        device = next(self.model.parameters()).device

        self.optimizer.zero_grad(set_to_none=True)
        accum_counter = 0

        for batch in self.train_loader:
            if batch is None:
                continue

            text_inputs = {
                "input_ids": batch["text_inputs"]["input_ids"].to(device, non_blocking=True),
                "attention_mask": batch["text_inputs"]["attention_mask"].to(device, non_blocking=True),
            }
            video_frames = batch["video_frames"].to(device, non_blocking=True)
            audio_features = batch["audio_features"].to(device, non_blocking=True)
            emotion_labels = batch["emotion_label"].to(device, non_blocking=True)
            sentiment_labels = batch["sentiment_label"].to(device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(text_inputs, video_frames, audio_features)
                emotion_loss = self.emotion_criterion(outputs["emotions"], emotion_labels)
                sentiment_loss = self.sentiment_criterion(outputs["sentiments"], sentiment_labels)
                total_loss = (
                    self.emotion_loss_weight * emotion_loss + self.sentiment_loss_weight * sentiment_loss
                ) / self.accum_steps

            self.scaler.scale(total_loss).backward()
            accum_counter += 1

            if accum_counter % self.accum_steps == 0:
                if self.grad_clip and self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model)
                if self._scheduler_step_batch:
                    self.scheduler.step()

            running_loss["total"] += float(total_loss.item()) * self.accum_steps
            running_loss["emotion"] += float(emotion_loss.item())
            running_loss["sentiment"] += float(sentiment_loss.item())
            batches += 1

            self.log_metrics(
                {
                    "total": float(total_loss.item()) * self.accum_steps,
                    "emotion": float(emotion_loss.item()),
                    "sentiment": float(sentiment_loss.item()),
                }
            )
            self.global_step += 1

        if accum_counter % self.accum_steps != 0:
            if self.grad_clip and self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update(self.model)
            if self._scheduler_step_batch:
                self.scheduler.step()

        return {k: v / max(batches, 1) for k, v in running_loss.items()}

    def evaluate(self, data_loader: Any, *, phase: str = "val") -> tuple[dict[str, float], dict[str, Any]]:
        self.model.eval()
        losses = {"total": 0.0, "emotion": 0.0, "sentiment": 0.0}
        batches = 0
        device = next(self.model.parameters()).device

        all_emotion_preds: list[int] = []
        all_emotion_labels: list[int] = []
        all_sentiment_preds: list[int] = []
        all_sentiment_labels: list[int] = []

        with torch.inference_mode():
            for batch in data_loader:
                if batch is None:
                    continue
                text_inputs = {
                    "input_ids": batch["text_inputs"]["input_ids"].to(device, non_blocking=True),
                    "attention_mask": batch["text_inputs"]["attention_mask"].to(device, non_blocking=True),
                }
                video_frames = batch["video_frames"].to(device, non_blocking=True)
                audio_features = batch["audio_features"].to(device, non_blocking=True)
                emotion_labels = batch["emotion_label"].to(device, non_blocking=True)
                sentiment_labels = batch["sentiment_label"].to(device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(text_inputs, video_frames, audio_features)
                    emotion_loss = self.emotion_criterion(outputs["emotions"], emotion_labels)
                    sentiment_loss = self.sentiment_criterion(outputs["sentiments"], sentiment_labels)
                    total_loss = (
                        self.emotion_loss_weight * emotion_loss + self.sentiment_loss_weight * sentiment_loss
                    )

                all_emotion_preds.extend(outputs["emotions"].argmax(dim=1).cpu().numpy().tolist())
                all_emotion_labels.extend(emotion_labels.cpu().numpy().tolist())
                all_sentiment_preds.extend(outputs["sentiments"].argmax(dim=1).cpu().numpy().tolist())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy().tolist())

                losses["total"] += float(total_loss.item())
                losses["emotion"] += float(emotion_loss.item())
                losses["sentiment"] += float(sentiment_loss.item())
                batches += 1

        if batches == 0:
            raise RuntimeError(f"No valid batches found during {phase} evaluation.")

        avg_loss = {k: v / batches for k, v in losses.items()}

        emotion_metrics = compute_classification_metrics(all_emotion_labels, all_emotion_preds, EMOTION_NAMES)
        sentiment_metrics = compute_classification_metrics(
            all_sentiment_labels, all_sentiment_preds, SENTIMENT_NAMES
        )
        flat_metrics: dict[str, Any] = {}
        for task, task_metrics in [("emotion", emotion_metrics), ("sentiment", sentiment_metrics)]:
            for key, value in task_metrics.items():
                flat_metrics[f"{task}_{key}"] = value

        self.log_metrics(avg_loss, flat_metrics, phase=phase)

        if phase == "val" and not self._scheduler_step_batch:
            self.scheduler.step(avg_loss["total"])

        return avg_loss, flat_metrics

    def export_ema_state_dict(self) -> dict[str, torch.Tensor]:
        if self.ema is None:
            return {k: v.cpu() for k, v in self.model.state_dict().items()}
        base = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        base.update({k: v.cpu().clone() for k, v in self.ema.state_dict().items()})
        return base
