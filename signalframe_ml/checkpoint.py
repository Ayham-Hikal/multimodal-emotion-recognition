"""Checkpoint loading with legacy key remapping and multiple on-disk layouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def load_checkpoint_blob(model_path: str | Path, *, map_location: torch.device | str) -> dict[str, Any]:
    """Load raw checkpoint dict from disk (supports full training checkpoints)."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # weights_only=True fails for dict checkpoints that include non-tensor metadata.
    return torch.load(str(path), map_location=map_location, weights_only=False)


def extract_state_dict(blob: dict[str, Any]) -> dict[str, Any]:
    if isinstance(blob.get("model_state_dict"), dict):
        return blob["model_state_dict"]
    if isinstance(blob.get("state_dict"), dict):
        return blob["state_dict"]
    # Raw weights-only dict (legacy exports)
    tensorish = sum(1 for v in blob.values() if torch.is_tensor(v))
    if tensorish >= max(1, len(blob) // 2):
        return blob
    raise ValueError("Unsupported checkpoint format: expected model_state_dict, state_dict, or tensor state.")


def remap_text_encoder_keys(state_dict: dict[str, Any], model_keys: set[str]) -> dict[str, Any]:
    """Map legacy BERT module prefixes to HuggingFace AutoModel `transformer.*` keys."""
    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        rk = key
        if key.startswith("text_encoder.bert."):
            candidate = key.replace("text_encoder.bert.", "text_encoder.transformer.", 1)
            if candidate in model_keys:
                rk = candidate
        elif key.startswith("text_encoder.transformer."):
            candidate = key.replace("text_encoder.transformer.", "text_encoder.bert.", 1)
            if candidate in model_keys:
                rk = candidate
        remapped[rk] = value
    return remapped


def load_weights_into_model(
    model: nn.Module,
    model_path: str | Path,
    *,
    map_location: torch.device | str,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Load weights into ``model`` in-place.

    Returns ``(missing_keys, unexpected_keys)`` from ``load_state_dict``.
    When ``strict`` is False, callers may still validate prefixes via ``assert_compatible``."""
    blob = load_checkpoint_blob(model_path, map_location=map_location)
    state_dict = extract_state_dict(blob)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format at {model_path}")

    model_keys = set(model.state_dict().keys())
    remapped = remap_text_encoder_keys(state_dict, model_keys)
    return model.load_state_dict(remapped, strict=strict)


def assert_compatible(
    missing_keys: list[str],
    unexpected_keys: list[str],
    *,
    allowed_prefixes: tuple[str, ...] = ("text_encoder.bert.", "text_encoder.transformer."),
) -> None:
    """Fail if incompatibility goes beyond known text-encoder naming drift."""
    bad_missing = [k for k in missing_keys if not k.startswith(allowed_prefixes)]
    bad_unexpected = [k for k in unexpected_keys if not k.startswith(allowed_prefixes)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible with the current architecture. "
            f"Missing keys: {bad_missing}. Unexpected keys: {bad_unexpected}."
        )
