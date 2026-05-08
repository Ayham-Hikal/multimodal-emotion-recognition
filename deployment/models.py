import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import models as vision_models
from torch.utils.tensorboard import SummaryWriter
import json


EMOTION_NAMES = ['anger', 'disgust', 'fear',
                 'joy', 'neutral', 'sadness', 'surprise']
SENTIMENT_NAMES = ['negative', 'neutral', 'positive']


class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', projection_dim=128, unfreeze_layers=0):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)

        for param in self.transformer.parameters():
            param.requires_grad = False

        if unfreeze_layers > 0 and hasattr(self.transformer, 'encoder'):
            layers = getattr(self.transformer.encoder, 'layer', None)
            if layers is not None:
                for layer in layers[-unfreeze_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        hidden_size = self.transformer.config.hidden_size
        self.projection = nn.Linear(hidden_size, projection_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        if getattr(outputs, 'pooler_output', None) is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]
        return self.projection(features)


class VideoEncoder(nn.Module):
    def __init__(self, projection_dim=128, dropout=0.2, unfreeze_backbone=False):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        for param in self.backbone.parameters():
            param.requires_grad = unfreeze_backbone

        num_fts = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # [batch_size, frames, channels, height, width]->[batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)
        return self.backbone(x)


class AudioEncoder(nn.Module):
    def __init__(self, projection_dim=128, dropout=0.2, freeze_conv=False):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Lower level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # The audio CNN starts from random weights, so freezing it blocks learning.
        for param in self.conv_layers.parameters():
            param.requires_grad = not freeze_conv

        self.projection = nn.Sequential(
            nn.Linear(128, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x.squeeze(1)

        features = self.conv_layers(x)
        # Features output: [batch_size, 128, 1]

        return self.projection(features.squeeze(-1))


class MultimodalSentimentModel(nn.Module):
    def __init__(self,
                 text_model_name='bert-base-uncased',
                 text_unfreeze_layers=0,
                 video_unfreeze_backbone=False,
                 audio_freeze_conv=False,
                 projection_dim=128,
                 fusion_dim=256,
                 fusion_dropout=0.3,
                 head_dropout=0.2):
        super().__init__()

        # Encoders
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            projection_dim=projection_dim,
            unfreeze_layers=text_unfreeze_layers
        )
        self.video_encoder = VideoEncoder(
            projection_dim=projection_dim,
            dropout=head_dropout,
            unfreeze_backbone=video_unfreeze_backbone
        )
        self.audio_encoder = AudioEncoder(
            projection_dim=projection_dim,
            dropout=head_dropout,
            freeze_conv=audio_freeze_conv
        )

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(projection_dim * 3, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout)
        )

        # Classification heads
        self.emotion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(64, 7)  # Sadness, anger
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(64, 3)  # Negative, positive, neutral
        )

    def forward(self, text_inputs, video_frames, audio_features):
        text_features = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask'],
        )
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Concatenate multimodal features
        combined_features = torch.cat([
            text_features,
            video_features,
            audio_features
        ], dim=1)  # [batch_size, 128 * 3]

        fused_features = self.fusion_layer(combined_features)

        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)

        return {
            'emotions': emotion_output,
            'sentiments': sentiment_output
        }