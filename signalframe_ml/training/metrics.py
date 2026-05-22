"""Losses, class rebalancing utilities, and sklearn-backed classification metrics."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from signalframe_ml.labels import EMOTION_NAMES, SENTIMENT_NAMES


def _weights_from_counts(counts: torch.Tensor, strategy: str, effective_beta: float) -> torch.Tensor | None:
    counts = counts.float().clamp_min(1.0)
    if strategy == "none":
        return None
    if strategy == "inverse":
        weights = 1.0 / counts
        return weights / weights.mean()
    if strategy == "balanced":
        weights = counts.sum() / (len(counts) * counts)
        return weights / weights.mean()
    if strategy == "sqrt_inverse":
        weights = torch.sqrt(counts.sum() / counts)
        return weights / weights.mean()
    if strategy == "effective_num":
        beta = effective_beta
        weights = (1.0 - beta) / (1.0 - torch.pow(torch.full_like(counts, beta), counts))
        return weights / weights.mean()
    raise ValueError(f"Unsupported class weight strategy: {strategy}")


def compute_class_weights(
    dataset: Any,
    emotion_strategy: str = "none",
    sentiment_strategy: str = "none",
    effective_beta: float = 0.999,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not hasattr(dataset, "get_class_counts"):
        if emotion_strategy == "none" and sentiment_strategy == "none":
            return None, None
        raise ValueError("Dataset must expose get_class_counts() for class weights.")

    counts = dataset.get_class_counts()
    emotion_counts = counts["emotion"]
    sentiment_counts = counts["sentiment"]

    print("\nClass distribution from label CSV")
    print("Emotions:")
    for name, count in zip(EMOTION_NAMES, emotion_counts):
        print(f"{name}: {int(count)}")

    print("\nSentiments:")
    for name, count in zip(SENTIMENT_NAMES, sentiment_counts):
        print(f"{name}: {int(count)}")

    emotion_weights = _weights_from_counts(emotion_counts, emotion_strategy, effective_beta)
    sentiment_weights = _weights_from_counts(sentiment_counts, sentiment_strategy, effective_beta)

    return emotion_weights, sentiment_weights


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        return ((1.0 - pt) ** self.gamma * ce_loss).mean()


def compute_classification_metrics(labels: list[int], preds: list[int], class_names: tuple[str, ...]) -> dict[str, Any]:
    labels = list(labels)
    preds = list(preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)

    per_class: dict[str, Any] = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(class_names)))).tolist(),
    }
