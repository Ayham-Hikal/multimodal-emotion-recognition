from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import torch.utils.data.dataloader
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import torchaudio
os.environ["TOKENIZERS_PARALLELISM"] = "false"

VIDEO_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
VIDEO_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir, text_model_name='bert-base-uncased', max_length=128):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        self.emotion_map = {
            'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6
        }

        self.sentiment_map = {
            'negative': 0, 'neutral': 1, 'positive': 2
        }
        self.emotion_names = ['anger', 'disgust', 'fear',
                              'joy', 'neutral', 'sadness', 'surprise']
        self.sentiment_names = ['negative', 'neutral', 'positive']
        self.emotion_labels = self.data['Emotion'].str.lower().map(
            self.emotion_map).astype(int).to_numpy()
        self.sentiment_labels = self.data['Sentiment'].str.lower().map(
            self.sentiment_map).astype(int).to_numpy()

    def get_class_counts(self):
        return {
            'emotion': torch.bincount(torch.tensor(self.emotion_labels), minlength=len(self.emotion_names)),
            'sentiment': torch.bincount(torch.tensor(self.sentiment_labels), minlength=len(self.sentiment_names)),
        }

    def get_sample_weights(self, task='emotion'):
        if task == 'emotion':
            labels = torch.tensor(self.emotion_labels)
            counts = torch.bincount(labels, minlength=len(self.emotion_names)).float()
        elif task == 'sentiment':
            labels = torch.tensor(self.sentiment_labels)
            counts = torch.bincount(labels, minlength=len(self.sentiment_names)).float()
        else:
            raise ValueError(f"Unsupported sampler task: {task}")

        class_weights = counts.sum() / counts.clamp_min(1.0)
        return class_weights[labels].double()

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")

            # Try and read first frame to validate video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")

            # Reset index to not skip first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
                frame = (frame - VIDEO_MEAN) / VIDEO_STD
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            cap.release()

        if (len(frames) == 0):
            raise ValueError("No frames could be extracted")

        # Pad or truncate frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        # Before permute: [frames, height, width, channels]
        # After permute: [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

    def _extract_audio_features(self, video_path):
        audio_path = video_path.replace('.mp4', '.wav')

        try:
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )

            mel_spec = mel_spectrogram(waveform)

            # Normalize per sample while avoiding NaNs on very short/silent clips.
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std().clamp_min(1e-6)

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        row = self.data.iloc[idx]

        try:
            video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"

            path = os.path.join(self.video_dir, video_filename)
            video_path_exists = os.path.exists(path)

            if not video_path_exists:
                raise FileNotFoundError(f"No video found for filename: {path}")

            text_inputs = self.tokenizer(row['Utterance'],
                                         padding='max_length',
                                         truncation=True,
                                         max_length=self.max_length,
                                         return_tensors='pt')

            video_frames = self._load_video_frames(path)
            audio_features = self._extract_audio_features(path)

            # Map sentiment and emotion labels
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]

            return {
                'text_inputs': {
                    'input_ids': text_inputs['input_ids'].squeeze(),
                    'attention_mask': text_inputs['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label': torch.tensor(sentiment_label)
            }
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            return None


def collate_fn(batch):
    # Filter oout None samples
    batch = list(filter(None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def prepare_dataloaders(train_csv, train_video_dir,
                        dev_csv, dev_video_dir,
                        test_csv, test_video_dir, batch_size=32,
                        text_model_name='bert-base-uncased',
                        max_length=128,
                        sampler_strategy='none',
                        sampler_task='emotion',
                        num_workers=0):
    train_dataset = MELDDataset(
        train_csv, train_video_dir, text_model_name=text_model_name, max_length=max_length)
    dev_dataset = MELDDataset(
        dev_csv, dev_video_dir, text_model_name=text_model_name, max_length=max_length)
    test_dataset = MELDDataset(
        test_csv, test_video_dir, text_model_name=text_model_name, max_length=max_length)

    sampler = None
    shuffle = True
    if sampler_strategy == 'weighted':
        sampler = WeightedRandomSampler(
            train_dataset.get_sample_weights(task=sampler_task),
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False
    elif sampler_strategy != 'none':
        raise ValueError(f"Unsupported sampler strategy: {sampler_strategy}")

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              sampler=sampler,
                              collate_fn=collate_fn,
                              num_workers=num_workers)

    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            num_workers=num_workers)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             collate_fn=collate_fn,
                             num_workers=num_workers)

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        '../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits',
        '../dataset/dev/dev_sent_emo.csv', '../dataset/dev/dev_splits_complete',
        '../dataset/test/test_sent_emo.csv', '../dataset/test/output_repeated_splits_test'
    )

    for batch in train_loader:
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break