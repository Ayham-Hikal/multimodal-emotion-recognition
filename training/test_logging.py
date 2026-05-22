import sys
from collections import namedtuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from signalframe_ml.model import MultimodalSentimentModel
from signalframe_ml.training import MultimodalTrainer


def test_logging() -> None:
    Batch = namedtuple("Batch", ["text_inputs", "video_frames", "audio_features"])
    mock_batch = Batch(
        text_inputs={"input_ids": torch.ones(1), "attention_mask": torch.ones(1)},
        video_frames=torch.ones(1),
        audio_features=torch.ones(1),
    )
    mock_loader = DataLoader([mock_batch])

    model = MultimodalSentimentModel()
    trainer = MultimodalTrainer(model, mock_loader, mock_loader, config={}, total_epochs=1)

    train_losses = {"total": 2.5, "emotion": 1.0, "sentiment": 1.5}
    trainer.log_metrics(train_losses, phase="train")

    val_losses = {"total": 1.5, "emotion": 0.5, "sentiment": 1.0}
    val_metrics = {
        "emotion_accuracy": 0.75,
        "emotion_macro_f1": 0.55,
        "emotion_weighted_f1": 0.70,
        "emotion_balanced_accuracy": 0.50,
        "sentiment_accuracy": 0.95,
        "sentiment_macro_f1": 0.90,
        "sentiment_precision": 0.85,
    }

    trainer.log_metrics(val_losses, val_metrics, phase="val")


if __name__ == "__main__":
    test_logging()
