import os
import argparse
import torchaudio
import torch
from tqdm import tqdm
import json
import sys
import random
import numpy as np

from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer
from install_ffmpeg import install_ffmpeg

# AWS SageMaker
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', ".")
SM_CHANNEL_TRAINING = os.environ.get(
    'SM_CHANNEL_TRAINING', "/opt/ml/input/data/training")
SM_CHANNEL_VALIDATION = os.environ.get(
    'SM_CHANNEL_VALIDATION', "/opt/ml/input/data/validation")
SM_CHANNEL_TEST = os.environ.get(
    'SM_CHANNEL_TEST', "/opt/ml/input/data/test")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--text-lr", type=float, default=8e-6)
    parser.add_argument("--encoder-lr", type=float, default=8e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    # Model and preprocessing
    parser.add_argument("--text-model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--text-unfreeze-layers", type=int, default=0)
    parser.add_argument("--video-unfreeze-backbone", action="store_true")
    parser.add_argument("--audio-freeze-conv", action="store_true")
    parser.add_argument("--fusion-dropout", type=float, default=0.3)
    parser.add_argument("--head-dropout", type=float, default=0.2)

    # Loss, imbalance, and model selection
    parser.add_argument("--loss-type", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--emotion-class-weight-strategy",
                        choices=["none", "inverse", "balanced", "sqrt_inverse", "effective_num"],
                        default="none")
    parser.add_argument("--sentiment-class-weight-strategy",
                        choices=["none", "inverse", "balanced", "sqrt_inverse", "effective_num"],
                        default="none")
    parser.add_argument("--effective-num-beta", type=float, default=0.999)
    parser.add_argument("--sampler-strategy", choices=["none", "weighted"], default="none")
    parser.add_argument("--sampler-task", choices=["emotion", "sentiment"], default="emotion")
    parser.add_argument("--emotion-loss-weight", type=float, default=1.0)
    parser.add_argument("--sentiment-loss-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--selection-metric", type=str, default="emotion_macro_f1")

    # Data directories
    parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def metric_is_better(metric_name, candidate, current_best):
    if current_best is None:
        return True
    if metric_name.endswith("loss") or metric_name == "total":
        return candidate < current_best
    return candidate > current_best


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main():
    if not install_ffmpeg():
        print("Error: FFmpeg installation failed. Cannot continue training.")
        sys.exit(1)

    print("Available audio backends:")
    print(str(torchaudio.list_audio_backends()))

    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.model_dir, exist_ok=True)

    # Track initial GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Initial GPU memory used: {memory_used:.2f} GB")

    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),
        train_video_dir=os.path.join(args.train_dir, 'train_splits'),
        dev_csv=os.path.join(args.val_dir, 'dev_sent_emo.csv'),
        dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete'),
        test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'),
        test_video_dir=os.path.join(
            args.test_dir, 'output_repeated_splits_test'),
        batch_size=args.batch_size,
        text_model_name=args.text_model_name,
        max_length=args.max_length,
        sampler_strategy=args.sampler_strategy,
        sampler_task=args.sampler_task,
        num_workers=args.num_workers
    )

    print(f"""Training DSV path: {os.path.join(
        args.train_dir, 'train_sent_emo.csv')}""")
    print(f"""Training video directory: {
          os.path.join(args.train_dir, 'train_splits')}""")

    model = MultimodalSentimentModel(
        text_model_name=args.text_model_name,
        text_unfreeze_layers=args.text_unfreeze_layers,
        video_unfreeze_backbone=args.video_unfreeze_backbone,
        audio_freeze_conv=args.audio_freeze_conv,
        fusion_dropout=args.fusion_dropout,
        head_dropout=args.head_dropout
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
    }
    trainer = MultimodalTrainer(model, train_loader, val_loader, config=trainer_config)
    best_selection_value = None
    best_epoch = None

    metrics_data = {
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
        val_loss, val_metrics = trainer.evaluate(val_loader)

        # Track metrics
        metrics_data["train_losses"].append(to_jsonable(train_loss))
        metrics_data["val_losses"].append(to_jsonable(val_loss))
        metrics_data["val_metrics"].append(to_jsonable(val_metrics))
        metrics_data["epochs"].append(epoch)

        # Log metrics in SageMaker format
        sm_metrics = [
            {"Name": "train:loss", "Value": train_loss["total"]},
            {"Name": "validation:loss", "Value": val_loss["total"]},
        ]
        for name, value in val_metrics.items():
            if isinstance(value, (int, float)):
                sm_metrics.append({"Name": f"validation:{name}", "Value": value})
        print(json.dumps({
            "metrics": sm_metrics
        }))

        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory used: {memory_used:.2f} GB")

        selection_value = val_loss["total"] if args.selection_metric == "val_loss" else val_metrics[args.selection_metric]
        if metric_is_better(args.selection_metric, selection_value, best_selection_value):
            best_selection_value = selection_value
            best_epoch = epoch
            metrics_data["best_epoch"] = best_epoch
            metrics_data["best_selection_value"] = best_selection_value
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "trainer_config": trainer_config,
                "epoch": epoch,
                "selection_metric": args.selection_metric,
                "selection_value": best_selection_value,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }, os.path.join(args.model_dir, "model.pth"))

        with open(os.path.join(args.model_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(to_jsonable(metrics_data), f, indent=2)

    # After training is complete, evaluate on test set
    checkpoint_path = os.path.join(args.model_dir, "model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

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