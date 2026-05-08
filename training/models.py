import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import models as vision_models
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import json

from meld_dataset import MELDDataset


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


def _weights_from_counts(counts, strategy='none', effective_beta=0.999):
    counts = counts.float().clamp_min(1.0)
    if strategy == 'none':
        return None
    if strategy == 'inverse':
        weights = 1.0 / counts
        return weights / weights.mean()
    if strategy == 'balanced':
        weights = counts.sum() / (len(counts) * counts)
        return weights / weights.mean()
    if strategy == 'sqrt_inverse':
        weights = torch.sqrt(counts.sum() / counts)
        return weights / weights.mean()
    if strategy == 'effective_num':
        beta = effective_beta
        weights = (1.0 - beta) / (1.0 - torch.pow(torch.full_like(counts, beta), counts))
        return weights / weights.mean()
    raise ValueError(f"Unsupported class weight strategy: {strategy}")


def compute_class_weights(dataset, emotion_strategy='none', sentiment_strategy='none', effective_beta=0.999):
    if not hasattr(dataset, 'get_class_counts'):
        if emotion_strategy == 'none' and sentiment_strategy == 'none':
            return None, None
        raise ValueError("Dataset must expose get_class_counts() for unbiased class weights.")

    counts = dataset.get_class_counts()
    emotion_counts = counts['emotion']
    sentiment_counts = counts['sentiment']

    print("\nClass distribution from label CSV")
    print("Emotions:")
    for name, count in zip(EMOTION_NAMES, emotion_counts):
        print(f"{name}: {int(count)}")

    print("\nSentiments:")
    for name, count in zip(SENTIMENT_NAMES, sentiment_counts):
        print(f"{name}: {int(count)}")

    emotion_weights = _weights_from_counts(
        emotion_counts, emotion_strategy, effective_beta)
    sentiment_weights = _weights_from_counts(
        sentiment_counts, sentiment_strategy, effective_beta)

    return emotion_weights, sentiment_weights


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        return ((1.0 - pt) ** self.gamma * ce_loss).mean()


def compute_classification_metrics(labels, preds, class_names):
    labels = list(labels)
    preds = list(preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(len(class_names))),
        zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0)

    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1': float(f1[idx]),
            'support': int(support[idx]),
        }

    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, preds)),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class': per_class,
        'confusion_matrix': confusion_matrix(
            labels, preds, labels=list(range(len(class_names)))).tolist(),
    }


class MultimodalTrainer:
    def __init__(self, model, train_loader, val_loader, config=None):
        self.config = config or {}
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Log dataset sized
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        print("\nDataset sizes:")
        print(f"Training samples: {train_size:,}")
        print(f"Validation samples: {val_size:,}")
        print(f"Batches per epoch: {len(train_loader):,}")

        timestamp = datetime.now().strftime('%b%d_%H-%M-%S')  # Dec17_14-22-35
        base_dir = '/opt/ml/output/tensorboard' if 'SM_MODEL_DIR' in os.environ else 'runs'
        log_dir = f"{base_dir}/run_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        text_lr = self.config.get('text_lr', 8e-6)
        encoder_lr = self.config.get('encoder_lr', 8e-5)
        head_lr = self.config.get('head_lr', 5e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)

        param_groups = [
            {'params': [p for p in model.text_encoder.parameters()
                        if p.requires_grad], 'lr': text_lr},
            {'params': [p for p in model.video_encoder.parameters()
                        if p.requires_grad], 'lr': encoder_lr},
            {'params': [p for p in model.audio_encoder.parameters()
                        if p.requires_grad], 'lr': encoder_lr},
            {'params': list(model.fusion_layer.parameters()), 'lr': head_lr},
            {'params': list(model.emotion_classifier.parameters()), 'lr': head_lr},
            {'params': list(model.sentiment_classifier.parameters()), 'lr': head_lr}
        ]
        param_groups = [group for group in param_groups if len(group['params']) > 0]
        self.optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=2
        )

        self.current_train_losses = None

        # Calculate class weights from CSV labels, not multimodal __getitem__.
        print("\nCalculating class weights...")
        emotion_weights, sentiment_weights = compute_class_weights(
            train_loader.dataset,
            emotion_strategy=self.config.get('emotion_class_weight_strategy', 'none'),
            sentiment_strategy=self.config.get('sentiment_class_weight_strategy', 'none'),
            effective_beta=self.config.get('effective_num_beta', 0.999),
        )

        device = next(model.parameters()).device

        self.emotion_weights = emotion_weights.to(device) if emotion_weights is not None else None
        self.sentiment_weights = sentiment_weights.to(device) if sentiment_weights is not None else None

        print(f"Emotion class weight strategy: {self.config.get('emotion_class_weight_strategy', 'none')}")
        print(f"Sentiment class weight strategy: {self.config.get('sentiment_class_weight_strategy', 'none')}")

        label_smoothing = self.config.get('label_smoothing', 0.0)
        loss_type = self.config.get('loss_type', 'ce')
        focal_gamma = self.config.get('focal_gamma', 2.0)

        if loss_type == 'ce':
            criterion_cls = nn.CrossEntropyLoss
            self.emotion_criterion = criterion_cls(
                label_smoothing=label_smoothing,
                weight=self.emotion_weights
            )
            self.sentiment_criterion = criterion_cls(
                label_smoothing=label_smoothing,
                weight=self.sentiment_weights
            )
        elif loss_type == 'focal':
            self.emotion_criterion = FocalLoss(
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
                weight=self.emotion_weights
            )
            self.sentiment_criterion = FocalLoss(
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
                weight=self.sentiment_weights
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        self.emotion_loss_weight = self.config.get('emotion_loss_weight', 1.0)
        self.sentiment_loss_weight = self.config.get('sentiment_loss_weight', 1.0)
        self.grad_clip = self.config.get('grad_clip', 1.0)

    def log_metrics(self, losses, metrics=None, phase="train"):
        if phase == "train":
            self.current_train_losses = losses
        else:  # Validation phase
            self.writer.add_scalar(
                'loss/total/train', self.current_train_losses['total'], self.global_step)
            self.writer.add_scalar(
                'loss/total/val', losses['total'], self.global_step)

            self.writer.add_scalar(
                'loss/emotion/train', self.current_train_losses['emotion'], self.global_step)
            self.writer.add_scalar(
                'loss/emotion/val', losses['emotion'], self.global_step)

            self.writer.add_scalar(
                'loss/sentiment/train', self.current_train_losses['sentiment'], self.global_step)
            self.writer.add_scalar(
                'loss/sentiment/val', losses['sentiment'], self.global_step)

        if metrics:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'{phase}/{name}', value, self.global_step)

    def train_epoch(self):
        self.model.train()
        running_loss = {'total': 0, 'emotion': 0, 'sentiment': 0}
        batches = 0

        for batch in self.train_loader:
            if batch is None:
                continue
            device = next(self.model.parameters()).device
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }
            video_frames = batch['video_frames'].to(device)
            audio_features = batch['audio_features'].to(device)
            emotion_labels = batch['emotion_label'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)

            # Zero gradient
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(text_inputs, video_frames, audio_features)

            # Calculate losses using raw logits
            emotion_loss = self.emotion_criterion(
                outputs["emotions"], emotion_labels)
            sentiment_loss = self.sentiment_criterion(
                outputs["sentiments"], sentiment_labels)
            total_loss = (
                self.emotion_loss_weight * emotion_loss +
                self.sentiment_loss_weight * sentiment_loss
            )

            # Backward pass. Calculate gradients
            total_loss.backward()

            # Gradient clipping
            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip)

            self.optimizer.step()

            # Track losses
            running_loss['total'] += total_loss.item()
            running_loss['emotion'] += emotion_loss.item()
            running_loss['sentiment'] += sentiment_loss.item()
            batches += 1

            self.log_metrics({
                'total': total_loss.item(),
                'emotion': emotion_loss.item(),
                'sentiment': sentiment_loss.item()
            })

            self.global_step += 1

        return {k: v/max(batches, 1) for k, v in running_loss.items()}

    def evaluate(self, data_loader, phase="val"):
        self.model.eval()
        losses = {'total': 0, 'emotion': 0, 'sentiment': 0}
        batches = 0
        all_emotion_preds = []
        all_emotion_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []

        with torch.inference_mode():
            for batch in data_loader:
                if batch is None:
                    continue
                device = next(self.model.parameters()).device
                text_inputs = {
                    'input_ids': batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device)
                }
                video_frames = batch['video_frames'].to(device)
                audio_features = batch['audio_features'].to(device)
                emotion_labels = batch['emotion_label'].to(device)
                sentiment_labels = batch['sentiment_label'].to(device)

                outputs = self.model(text_inputs, video_frames, audio_features)

                emotion_loss = self.emotion_criterion(
                    outputs["emotions"], emotion_labels)
                sentiment_loss = self.sentiment_criterion(
                    outputs["sentiments"], sentiment_labels)
                total_loss = (
                    self.emotion_loss_weight * emotion_loss +
                    self.sentiment_loss_weight * sentiment_loss
                )

                all_emotion_preds.extend(
                    outputs["emotions"].argmax(dim=1).cpu().numpy())
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_sentiment_preds.extend(
                    outputs["sentiments"].argmax(dim=1).cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

                # Track losses
                losses['total'] += total_loss.item()
                losses['emotion'] += emotion_loss.item()
                losses['sentiment'] += sentiment_loss.item()
                batches += 1

        if batches == 0:
            raise RuntimeError(f"No valid batches found during {phase} evaluation.")

        avg_loss = {k: v/batches for k, v in losses.items()}

        emotion_metrics = compute_classification_metrics(
            all_emotion_labels, all_emotion_preds, EMOTION_NAMES)
        sentiment_metrics = compute_classification_metrics(
            all_sentiment_labels, all_sentiment_preds, SENTIMENT_NAMES)
        flat_metrics = {}
        for task, task_metrics in [('emotion', emotion_metrics), ('sentiment', sentiment_metrics)]:
            for key, value in task_metrics.items():
                flat_metrics[f'{task}_{key}'] = value

        self.log_metrics(avg_loss, flat_metrics, phase=phase)

        if phase == "val":
            self.scheduler.step(avg_loss['total'])

        return avg_loss, flat_metrics


if __name__ == "__main__":
    dataset = MELDDataset(
        '../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits')

    sample = dataset[0]

    model = MultimodalSentimentModel()
    model.eval()

    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
    }
    video_frames = sample['video_frames'].unsqueeze(0)
    audio_features = sample['audio_features'].unsqueeze(0)

    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)

        emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
        sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0]

    emotion_map = {
        0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'
    }

    sentiment_map = {
        0: 'negative', 1: 'neutral', 2: 'positive'
    }

    for i, prob in enumerate(emotion_probs):
        print(f"{emotion_map[i]}: {prob:.2f}")

    for i, prob in enumerate(sentiment_probs):
        print(f"{sentiment_map[i]}: {prob:.2f}")

    print("Predictions for utterance")