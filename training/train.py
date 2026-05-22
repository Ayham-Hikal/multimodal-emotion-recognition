"""
MELD multimodal training entrypoint.

Install once::

    cd backend && pip install -e ".[train]"

Run::

    cd backend/training && python train.py --epochs 20 --model-dir ./outputs
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from meld_dataset import prepare_dataloaders
from signalframe_ml.model import MultimodalSentimentModel
from signalframe_ml.training import MultimodalTrainer

from install_ffmpeg import install_ffmpeg

SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", ".")
SM_CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
SM_CHANNEL_VALIDATION = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
SM_CHANNEL_TEST = os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multimodal emotion + sentiment on MELD.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--text-lr", type=float, default=8e-6)
    p.add_argument("--encoder-lr", type=float, default=8e-5)
    p.add_argument("--head-lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--text-model-name", type=str, default="bert-base-uncased")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--text-unfreeze-layers", type=int, default=0)
    p.add_argument("--video-unfreeze-backbone", action="store_true")
    p.add_argument("--audio-freeze-conv", action="store_true")
    p.add_argument("--fusion-dropout", type=float, default=0.3)
    p.add_argument("--head-dropout", type=float, default=0.2)

    p.add_argument("--loss-type", choices=["ce", "focal"], default="ce")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument(
        "--emotion-class-weight-strategy",
        choices=["none", "inverse", "balanced", "sqrt_inverse", "effective_num"],
        default="none",
    )
    p.add_argument(
        "--sentiment-class-weight-strategy",
        choices=["none", "inverse", "balanced", "sqrt_inverse", "effective_num"],
        default="none",
    )
    p.add_argument("--effective-num-beta", type=float, default=0.999)
    p.add_argument("--sampler-strategy", choices=["none", "weighted"], default="none")
    p.add_argument("--sampler-task", choices=["emotion", "sentiment"], default="emotion")
    p.add_argument("--emotion-loss-weight", type=float, default=1.0)
    p.add_argument("--sentiment-loss-weight", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--selection-metric", type=str, default="emotion_macro_f1")

    p.add_argument("--scheduler", choices=["cosine", "plateau"], default="cosine")
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision on CUDA.")
    p.add_argument("--ema-decay", type=float, default=0.0, help="0 disables EMA (e.g. 0.999 for Polyak averaging).")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases (WANDB_PROJECT env).")

    p.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)
    p.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
    p.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
    p.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def metric_is_better(metric_name: str, candidate: float, current_best: float | None) -> bool:
    if current_best is None:
        return True
    if metric_name.endswith("loss") or metric_name == "total":
        return candidate < current_best
    return candidate > current_best


def to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    if not install_ffmpeg():
        print("Error: FFmpeg installation failed. Cannot continue training.")
        sys.exit(1)

    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.model_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"Initial GPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, "train_sent_emo.csv"),
        train_video_dir=os.path.join(args.train_dir, "train_splits"),
        dev_csv=os.path.join(args.val_dir, "dev_sent_emo.csv"),
        dev_video_dir=os.path.join(args.val_dir, "dev_splits_complete"),
        test_csv=os.path.join(args.test_dir, "test_sent_emo.csv"),
        test_video_dir=os.path.join(args.test_dir, "output_repeated_splits_test"),
        batch_size=args.batch_size,
        text_model_name=args.text_model_name,
        max_length=args.max_length,
        sampler_strategy=args.sampler_strategy,
        sampler_task=args.sampler_task,
        num_workers=args.num_workers,
    )

    model = MultimodalSentimentModel(
        text_model_name=args.text_model_name,
        text_unfreeze_layers=args.text_unfreeze_layers,
        video_unfreeze_backbone=args.video_unfreeze_backbone,
        audio_freeze_conv=args.audio_freeze_conv,
        fusion_dropout=args.fusion_dropout,
        head_dropout=args.head_dropout,
    ).to(device)

    trainer_config = {
        "text_lr": args.text_lr,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "loss_type": args.loss_type,
        "focal_gamma": args.focal_gamma,
        "label_smoothing": args.label_smoothing,
        "emotion_class_weight_strategy": args.emotion_class_weight_strategy,
        "sentiment_class_weight_strategy": args.sentiment_class_weight_strategy,
        "effective_num_beta": args.effective_num_beta,
        "emotion_loss_weight": args.emotion_loss_weight,
        "sentiment_loss_weight": args.sentiment_loss_weight,
        "grad_clip": args.grad_clip,
        "scheduler": args.scheduler,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "use_amp": not args.no_amp,
        "ema_decay": args.ema_decay,
        "wandb": args.wandb,
    }

    trainer = MultimodalTrainer(
        model,
        train_loader,
        val_loader,
        config=trainer_config,
        total_epochs=args.epochs,
    )

    best_selection_value: float | None = None
    best_epoch: int | None = None

    metrics_data: dict[str, object] = {
        "config": vars(args),
        "train_losses": [],
        "val_losses": [],
        "epochs": [],
        "val_metrics": [],
        "best_epoch": None,
        "best_selection_metric": args.selection_metric,
        "best_selection_value": None,
    }

    with open(os.path.join(args.model_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_loss = trainer.train_epoch()
        val_loss, val_metrics = trainer.evaluate(val_loader, phase="val")

        metrics_data["train_losses"].append(to_jsonable(train_loss))
        metrics_data["val_losses"].append(to_jsonable(val_loss))
        metrics_data["val_metrics"].append(to_jsonable(val_metrics))
        metrics_data["epochs"].append(epoch)

        sm_metrics = [
            {"Name": "train:loss", "Value": train_loss["total"]},
            {"Name": "validation:loss", "Value": val_loss["total"]},
        ]
        for name, value in val_metrics.items():
            if isinstance(value, (int, float)):
                sm_metrics.append({"Name": f"validation:{name}", "Value": value})
        print(json.dumps({"metrics": sm_metrics}))

        if torch.cuda.is_available():
            print(f"Peak GPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

        selection_value = val_loss["total"] if args.selection_metric == "val_loss" else float(val_metrics[args.selection_metric])
        if metric_is_better(args.selection_metric, selection_value, best_selection_value):
            best_selection_value = selection_value
            best_epoch = epoch
            metrics_data["best_epoch"] = best_epoch
            metrics_data["best_selection_value"] = best_selection_value

            weights = trainer.export_ema_state_dict()
            torch.save(
                {
                    "model_state_dict": weights,
                    "config": vars(args),
                    "trainer_config": trainer_config,
                    "epoch": epoch,
                    "selection_metric": args.selection_metric,
                    "selection_value": best_selection_value,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                },
                os.path.join(args.model_dir, "model.pth"),
            )

        with open(os.path.join(args.model_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(to_jsonable(metrics_data), f, indent=2)

    checkpoint_path = os.path.join(args.model_dir, "model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    print("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
    metrics_data["test_loss"] = to_jsonable(test_loss)
    metrics_data["test_metrics"] = to_jsonable(test_metrics)

    sm_test_metrics = [{"Name": "test:loss", "Value": test_loss["total"]}]
    for name, value in test_metrics.items():
        if isinstance(value, (int, float)):
            sm_test_metrics.append({"Name": f"test:{name}", "Value": value})
    print(json.dumps({"metrics": sm_test_metrics}))

    with open(os.path.join(args.model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(metrics_data), f, indent=2)


if __name__ == "__main__":
    main()
