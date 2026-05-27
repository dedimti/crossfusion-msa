"""
Dataset loaders for CrossFusion.

Supports:
  - MER 2023 (multilingual, 8 languages, 4,642 utterances)
  - MER 2024 (multilingual, 8 languages, 9,241 utterances)
  - CMU-MOSEI (English, 16,326 utterances)

Missing modalities are handled by returning zero tensors of the
correct shape. The model's gating mechanism suppresses zero inputs
automatically — no explicit missingness flags needed.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, Wav2Vec2Processor, ViTFeatureExtractor


class MultimodalSentimentDataset(Dataset):
    """
    Generic dataset for multimodal sentiment analysis.

    Expects a JSON manifest file with entries:
    {
        "utterance_id": "utt_001",
        "text": "This movie was amazing",
        "audio_path": "audio/utt_001.wav",
        "video_path": "video/utt_001",
        "polarity": 2,           // 0=Negative, 1=Neutral, 2=Positive
        "intensity": 0.85,       // float in [0, 1]
        "kappa": 0.74,           // per-utterance Fleiss kappa
        "missing_audio": false,  // optional: explicit missingness flag
        "missing_video": false
    }

    Args:
        manifest_path:  Path to JSON manifest file.
        data_root:      Root directory for audio/video files.
        split:          One of 'train', 'val', 'test'.
        max_text_len:   Maximum token sequence length for BERT.
        audio_max_len:  Maximum audio length in seconds (truncated/padded).
        n_video_frames: Number of video frames to sample per utterance.
        p_miss_audio:   Probability of dropping audio at training time (data augmentation).
        p_miss_video:   Probability of dropping video at training time.
    """

    POLARITY_MAP = {"negative": 0, "neutral": 1, "positive": 2}

    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        split: str = "train",
        max_text_len: int = 512,
        audio_max_len: float = 10.0,
        n_video_frames: int = 8,
        p_miss_audio: float = 0.0,
        p_miss_video: float = 0.0,
    ):
        self.data_root = data_root
        self.split = split
        self.max_text_len = max_text_len
        self.audio_max_len = audio_max_len
        self.n_video_frames = n_video_frames
        self.p_miss_audio = p_miss_audio if split == "train" else 0.0
        self.p_miss_video = p_miss_video if split == "train" else 0.0

        # Load manifest
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        # Filter by split
        self.samples = [s for s in self.samples if s.get("split", split) == split]

        # Initialize processors
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.video_processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

        # Audio sample rate
        self.audio_sr = 16000
        self.audio_max_samples = int(audio_max_len * self.audio_sr)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # --- Text ---
        text_encoding = self.tokenizer(
            sample["text"],
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_encoding["input_ids"].squeeze(0)
        attention_mask = text_encoding["attention_mask"].squeeze(0)

        # --- Audio ---
        missing_audio = sample.get("missing_audio", False)
        if not missing_audio and np.random.random() < self.p_miss_audio:
            missing_audio = True

        if missing_audio:
            audio_values = torch.zeros(self.audio_max_samples)
            audio_attention_mask = torch.zeros(self.audio_max_samples, dtype=torch.long)
        else:
            audio_values, audio_attention_mask = self._load_audio(
                os.path.join(self.data_root, sample["audio_path"])
            )

        # --- Video ---
        missing_video = sample.get("missing_video", False)
        if not missing_video and np.random.random() < self.p_miss_video:
            missing_video = True

        if missing_video:
            pixel_values = torch.zeros(self.n_video_frames, 3, 224, 224)
        else:
            pixel_values = self._load_video(
                os.path.join(self.data_root, sample["video_path"])
            )

        # --- Labels ---
        polarity = sample["polarity"]
        if isinstance(polarity, str):
            polarity = self.POLARITY_MAP[polarity.lower()]
        polarity = torch.tensor(polarity, dtype=torch.long)

        intensity = torch.tensor(sample.get("intensity", 0.5), dtype=torch.float32)
        kappa = torch.tensor(sample.get("kappa", 0.61), dtype=torch.float32)

        return {
            "utterance_id": sample["utterance_id"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_values": audio_values,
            "audio_attention_mask": audio_attention_mask,
            "pixel_values": pixel_values,
            "polarity": polarity,
            "intensity": intensity,
            "kappa": kappa,
            "missing_audio": torch.tensor(missing_audio, dtype=torch.bool),
            "missing_video": torch.tensor(missing_video, dtype=torch.bool),
        }

    def _load_audio(self, path: str):
        """Load and preprocess audio waveform."""
        import torchaudio
        waveform, sr = torchaudio.load(path)

        # Resample to 16kHz
        if sr != self.audio_sr:
            resampler = torchaudio.transforms.Resample(sr, self.audio_sr)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        waveform = waveform.squeeze(0)

        # Truncate or pad
        if waveform.shape[0] > self.audio_max_samples:
            waveform = waveform[:self.audio_max_samples]
            mask = torch.ones(self.audio_max_samples, dtype=torch.long)
        else:
            pad_len = self.audio_max_samples - waveform.shape[0]
            mask_len = waveform.shape[0]
            waveform = torch.cat([waveform, torch.zeros(pad_len)])
            mask = torch.cat([
                torch.ones(mask_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ])

        return waveform, mask

    def _load_video(self, frame_dir: str):
        """Load video frames from directory of JPEG/PNG images."""
        from PIL import Image
        import glob

        frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")) +
                        glob.glob(os.path.join(frame_dir, "*.png")))

        if not frames:
            return torch.zeros(self.n_video_frames, 3, 224, 224)

        # Sample n_video_frames evenly
        indices = np.linspace(0, len(frames) - 1, self.n_video_frames, dtype=int)
        selected = [frames[i] for i in indices]

        pixel_list = []
        for fp in selected:
            img = Image.open(fp).convert("RGB")
            processed = self.video_processor(images=img, return_tensors="pt")
            pixel_list.append(processed["pixel_values"])  # (1, 3, 224, 224)

        return torch.cat(pixel_list, dim=0)  # (n_video_frames, 3, 224, 224)


def build_dataloader(
    manifest_path: str,
    data_root: str,
    split: str,
    batch_size: int = 32,
    num_workers: int = 4,
    p_miss_audio: float = 0.0,
    p_miss_video: float = 0.20,
    **dataset_kwargs,
) -> DataLoader:
    """Build a DataLoader for a given split."""
    dataset = MultimodalSentimentDataset(
        manifest_path=manifest_path,
        data_root=data_root,
        split=split,
        p_miss_audio=p_miss_audio,
        p_miss_video=p_miss_video,
        **dataset_kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
