import torch
import torch.nn as nn
from torchvision import models as vision_models
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", projection_dim: int = 128):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)

        for param in self.transformer.parameters():
            param.requires_grad = False

        hidden_size = self.transformer.config.hidden_size
        self.projection = nn.Linear(hidden_size, projection_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        if getattr(outputs, "pooler_output", None) is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]
        return self.projection(features)


class VideoEncoder(nn.Module):
    def __init__(self, projection_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        num_fts = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, frames, channels, height, width] -> [batch, channels, frames, height, width]
        x = x.transpose(1, 2)
        return self.backbone(x)


class AudioEncoder(nn.Module):
    def __init__(self, projection_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.projection = nn.Sequential(
            nn.Linear(128, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        features = self.conv_layers(x)
        return self.projection(features.squeeze(-1))


class MultimodalSentimentModel(nn.Module):
    def __init__(self, projection_dim: int = 128, fusion_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.text_encoder = TextEncoder(projection_dim=projection_dim)
        self.video_encoder = VideoEncoder(projection_dim=projection_dim, dropout=dropout)
        self.audio_encoder = AudioEncoder(projection_dim=projection_dim, dropout=dropout)

        self.fusion_layer = nn.Sequential(
            nn.Linear(projection_dim * 3, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.emotion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 7),
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )

    def forward(
        self,
        text_inputs: dict[str, torch.Tensor],
        video_frames: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_features = self.text_encoder(
            text_inputs["input_ids"],
            text_inputs["attention_mask"],
        )
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        combined_features = torch.cat([text_features, video_features, audio_features], dim=1)
        fused_features = self.fusion_layer(combined_features)

        return {
            "emotions": self.emotion_classifier(fused_features),
            "sentiments": self.sentiment_classifier(fused_features),
        }
