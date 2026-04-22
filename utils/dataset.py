import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from utils.audio_features import extract_mfcc, extract_temporal_features, extract_energy_features

class MFCCDataset(Dataset):
    """baseline：仅 MFCC，一通道"""
    def __init__(self, metadata, audio_dir, target_size=(128, 128), indices=None,
                 is_train=False, augment_prob=0.4, stats=None):
        self.metadata = metadata if indices is None else [metadata[i] for i in indices]
        self.audio_dir = audio_dir
        self.target_size = target_size
        self.is_train = is_train
        self.augment_prob = augment_prob

        self.labels = [item["label"] for item in self.metadata]
        if stats is None:
            self.mean = 0.0
            self.std = 1.0
        else:
            self.mean = stats["mfcc_mean"]
            self.std = stats["mfcc_std"] if stats["mfcc_std"] > 0 else 1.0

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        audio_path = os.path.join(self.audio_dir, item["file_path"].replace(".npy", ".wav"))

        mfcc = extract_mfcc(audio_path, n_mfcc=40)

        if mfcc.shape[1] > 100:
            mfcc = mfcc[:, :100]
        elif mfcc.shape[1] < 100:
            pad_width = 100 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")

        mfcc = (mfcc - self.mean) / self.std
        zoom_factor = (self.target_size[0] / mfcc.shape[0],
                       self.target_size[1] / mfcc.shape[1])
        mfcc = ndimage.zoom(mfcc, zoom_factor, order=1)

        feat = mfcc[np.newaxis, :, :].astype(np.float32)  # 1 x H x W

        if self.is_train and random.random() < self.augment_prob:
            feat = self._augment(feat)

        return torch.from_numpy(feat), torch.tensor(item["label"], dtype=torch.long)

    def _augment(self, features):
        augmented = features.copy()
        if random.random() < 0.6:
            shift = int(augmented.shape[2] * 0.15 * (random.random() - 0.5))
            augmented = np.roll(augmented, shift, axis=2)
            if shift > 0:
                augmented[:, :, :shift] = 0
            else:
                augmented[:, :, shift:] = 0
        if random.random() < 0.4:
            noise = np.random.normal(0, 0.03, augmented.shape).astype(np.float32)
            augmented += noise
        return np.clip(augmented, -3, 3).astype(np.float32)


class MFCCTemporalDataset(Dataset):
    """Model A：MFCC + Temporal，2 通道"""
    def __init__(self, metadata, audio_dir, target_size=(128, 128), indices=None,
                 is_train=False, augment_prob=0.4, stats=None):
        self.metadata = metadata if indices is None else [metadata[i] for i in indices]
        self.audio_dir = audio_dir
        self.target_size = target_size
        self.is_train = is_train
        self.augment_prob = augment_prob
        self.labels = [item["label"] for item in self.metadata]

        if stats is None:
            self.mfcc_mean = 0.0; self.mfcc_std = 1.0
            self.temp_mean = 0.0; self.temp_std = 1.0
        else:
            self.mfcc_mean = stats["mfcc_mean"]; self.mfcc_std = max(stats["mfcc_std"], 1e-6)
            self.temp_mean = stats["temporal_mean"]; self.temp_std = max(stats["temporal_std"], 1e-6)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        audio_path = os.path.join(self.audio_dir, item["file_path"].replace(".npy", ".wav"))

        mfcc = extract_mfcc(audio_path, n_mfcc=40)
        temporal = extract_temporal_features(audio_path)

        if mfcc.shape[1] > 100:
            mfcc = mfcc[:, :100]; temporal = temporal[:100]
        elif mfcc.shape[1] < 100:
            pad_width = 100 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
            temporal = np.pad(temporal, (0, pad_width), mode="constant")

        mfcc = (mfcc - self.mfcc_mean) / self.mfcc_std
        temporal = (temporal - self.temp_mean) / self.temp_std

        def _resize(feat):
            if feat.ndim == 1:
                feat = feat.reshape(1, -1)
            zoom_factor = (self.target_size[0] / feat.shape[0],
                           self.target_size[1] / feat.shape[1])
            return ndimage.zoom(feat, zoom_factor, order=1)

        mfcc = _resize(mfcc)
        temporal = _resize(temporal)

        if mfcc.shape != temporal.shape:
            temporal = np.resize(temporal, mfcc.shape)

        feats = np.stack([mfcc, temporal], axis=0).astype(np.float32)

        if self.is_train and random.random() < self.augment_prob:
            feats = self._augment(feats)

        return torch.from_numpy(feats), torch.tensor(item["label"], dtype=torch.long)

    def _augment(self, features):
        augmented = features.copy()
        if random.random() < 0.6:
            shift = int(augmented.shape[2] * 0.15 * (random.random() - 0.5))
            augmented = np.roll(augmented, shift, axis=2)
            if shift > 0:
                augmented[:, :, :shift] = 0
            else:
                augmented[:, :, shift:] = 0
        if random.random() < 0.4:
            noise = np.random.normal(0, 0.03, augmented.shape).astype(np.float32)
            augmented += noise
        return np.clip(augmented, -3, 3).astype(np.float32)


class MFCCEnergyDataset(Dataset):
    """Model B：MFCC + Energy(Mel)，2 通道"""
    def __init__(self, metadata, audio_dir, target_size=(128, 128), indices=None,
                 is_train=False, augment_prob=0.4, stats=None):
        self.metadata = metadata if indices is None else [metadata[i] for i in indices]
        self.audio_dir = audio_dir
        self.target_size = target_size
        self.is_train = is_train
        self.augment_prob = augment_prob
        self.labels = [item["label"] for item in self.metadata]

        if stats is None:
            self.mfcc_mean = 0.0; self.mfcc_std = 1.0
            self.energy_mean = 0.0; self.energy_std = 1.0
        else:
            self.mfcc_mean = stats["mfcc_mean"]; self.mfcc_std = max(stats["mfcc_std"], 1e-6)
            self.energy_mean = stats["energy_mean"]; self.energy_std = max(stats["energy_std"], 1e-6)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        audio_path = os.path.join(self.audio_dir, item["file_path"].replace(".npy", ".wav"))

        mfcc = extract_mfcc(audio_path, n_mfcc=40)
        energy = extract_energy_features(audio_path, n_mels=40)

        if mfcc.shape[1] > 100:
            mfcc = mfcc[:, :100]; energy = energy[:, :100]
        elif mfcc.shape[1] < 100:
            pad_width = 100 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
            energy = np.pad(energy, ((0, 0), (0, pad_width)), mode="constant")

        mfcc = (mfcc - self.mfcc_mean) / self.mfcc_std
        energy = (energy - self.energy_mean) / self.energy_std

        zoom_factor = (self.target_size[0] / mfcc.shape[0],
                       self.target_size[1] / mfcc.shape[1])
        mfcc = ndimage.zoom(mfcc, zoom_factor, order=1)
        energy = ndimage.zoom(energy, zoom_factor, order=1)

        if mfcc.shape != energy.shape:
            energy = np.resize(energy, mfcc.shape)

        feats = np.stack([mfcc, energy], axis=0).astype(np.float32)

        if self.is_train and random.random() < self.augment_prob:
            feats = self._augment(feats)

        return torch.from_numpy(feats), torch.tensor(item["label"], dtype=torch.long)

    def _augment(self, features):
        augmented = features.copy()
        if random.random() < 0.6:
            shift = int(augmented.shape[2] * 0.15 * (random.random() - 0.5))
            augmented = np.roll(augmented, shift, axis=2)
            if shift > 0:
                augmented[:, :, :shift] = 0
            else:
                augmented[:, :, shift:] = 0
        if random.random() < 0.4:
            noise = np.random.normal(0, 0.03, augmented.shape).astype(np.float32)
            augmented += noise
        return np.clip(augmented, -3, 3).astype(np.float32)


class ThreeFeatureDataset(Dataset):
    """Model C / C_CR：MFCC + Temporal + Energy，3 通道"""
    def __init__(self, metadata, audio_dir, target_size=(128, 128), indices=None,
                 is_train=False, augment_prob=0.4, stats=None):
        self.metadata = metadata if indices is None else [metadata[i] for i in indices]
        self.audio_dir = audio_dir
        self.target_size = target_size
        self.is_train = is_train
        self.augment_prob = augment_prob

        self.labels = [item["label"] for item in self.metadata]

        if stats is None:
            self.mfcc_mean = 0.0; self.mfcc_std = 1.0
            self.temporal_mean = 0.0; self.temporal_std = 1.0
            self.energy_mean = 0.0; self.energy_std = 1.0
        else:
            self.mfcc_mean = stats["mfcc_mean"]; self.mfcc_std = max(stats["mfcc_std"], 1e-6)
            self.temporal_mean = stats["temporal_mean"]; self.temporal_std = max(stats["temporal_std"], 1e-6)
            self.energy_mean = stats["energy_mean"]; self.energy_std = max(stats["energy_std"], 1e-6)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        audio_path = os.path.join(self.audio_dir, item["file_path"].replace(".npy", ".wav"))

        mfcc = extract_mfcc(audio_path, n_mfcc=40)
        temporal = extract_temporal_features(audio_path)
        energy = extract_energy_features(audio_path, n_mels=40)

        if mfcc.shape[1] > 100:
            mfcc = mfcc[:, :100]; temporal = temporal[:100]; energy = energy[:, :100]
        elif mfcc.shape[1] < 100:
            pad_width = 100 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
            temporal = np.pad(temporal, (0, pad_width), mode="constant")
            energy = np.pad(energy, ((0, 0), (0, pad_width)), mode="constant")

        mfcc = (mfcc - self.mfcc_mean) / self.mfcc_std
        temporal = (temporal - self.temporal_mean) / self.temporal_std
        energy = (energy - self.energy_mean) / self.energy_std

        def _resize(feat):
            if feat.ndim == 1:
                feat = feat.reshape(1, -1)
            zoom_factor = (self.target_size[0] / feat.shape[0],
                           self.target_size[1] / feat.shape[1])
            return ndimage.zoom(feat, zoom_factor, order=1)

        mfcc = _resize(mfcc)
        temporal = _resize(temporal)
        energy = _resize(energy)

        if mfcc.shape != temporal.shape:
            temporal = np.resize(temporal, mfcc.shape)
        if mfcc.shape != energy.shape:
            energy = np.resize(energy, mfcc.shape)

        feats = np.stack([mfcc, temporal, energy], axis=0).astype(np.float32)

        if self.is_train and random.random() < self.augment_prob:
            feats = self._augment(feats)

        return torch.from_numpy(feats), torch.tensor(item["label"], dtype=torch.long)

    def _augment(self, features):
        augmented = features.copy()
        if random.random() < 0.6:
            shift = int(augmented.shape[2] * 0.15 * (random.random() - 0.5))
            augmented = np.roll(augmented, shift, axis=2)
            if shift > 0:
                augmented[:, :, :shift] = 0
            else:
                augmented[:, :, shift:] = 0
        if random.random() < 0.4:
            noise = np.random.normal(0, 0.03, augmented.shape).astype(np.float32)
            augmented += noise
        return np.clip(augmented, -3, 3).astype(np.float32)

class MultiFeatureDataset(Dataset):
    """
    Generic multi-feature dataset for 3-channel combinations such as:
    - MFCC + Mel + PCEN
    - MFCC + Mel + Chroma 
    - MFCC + Mel + Spectral Contrast
    """
    def __init__(self, metadata, audio_dir, feature_names, target_size=(128, 128),
                 indices=None, is_train=False, augment_prob=0.4, stats=None):
        self.metadata = metadata if indices is None else [metadata[i] for i in indices]
        self.audio_dir = audio_dir
        self.feature_names = feature_names
        self.target_size = target_size
        self.is_train = is_train
        self.augment_prob = augment_prob
        self.stats = stats if stats is not None else {}

    def __len__(self):
        return len(self.metadata)

    def _extract_one_feature(self, feat_name, audio_path):
        if feat_name == "mfcc":
            feat = extract_mfcc(audio_path, n_mfcc=40)
        elif feat_name == "energy":
            feat = extract_energy_features(audio_path, n_mels=40)
        elif feat_name == "chroma":
            feat = extract_chroma_features(audio_path, n_chroma=12)
        elif feat_name == "pcen":
            feat = extract_pcen_features(audio_path, n_mels=40)
        elif feat_name == "spectral":
            feat = extract_spectral_contrast(audio_path, n_bands=6)
        else:
            raise ValueError(f"Unknown feature: {feat_name}")

        # 处理时长和尺寸
        feat = self._fix_time_length(feat, target_frames=100)
        feat = self._resize_feature(feat, self.target_size)
        return feat

    def __getitem__(self, idx):
        item = self.metadata[idx]
        audio_path = os.path.join(self.audio_dir, item["file_path"].replace(".npy", ".wav"))

        feats = []
        for feat_name in self.feature_names:
            feat = self._extract_one_feature(feat_name, audio_path)
            feats.append(feat)

        feats = np.stack(feats, axis=0).astype(np.float32)

        if self.is_train and random.random() < self.augment_prob:
            feats = self._augment(feats)

        return torch.from_numpy(feats), torch.tensor(item["label"], dtype=torch.long)

    def _augment(self, features):
        augmented = features.copy()
        if random.random() < 0.6:
            shift = int(augmented.shape[2] * 0.15 * (random.random() - 0.5))
            augmented = np.roll(augmented, shift, axis=2)
            if shift > 0:
                augmented[:, :, :shift] = 0
            else:
                augmented[:, :, shift:] = 0
        if random.random() < 0.4:
            noise = np.random.normal(0, 0.03, augmented.shape).astype(np.float32)
            augmented += noise
        return np.clip(augmented, -3, 3).astype(np.float32)

    def _fix_time_length(self, feat, target_frames=100):
        if feat.ndim == 1:
            if feat.shape[0] > target_frames:
                feat = feat[:target_frames]
            elif feat.shape[0] < target_frames:
                feat = np.pad(feat, (0, target_frames - feat.shape[0]), mode="constant")
            feat = feat.reshape(1, -1)
            return feat.astype(np.float32)

        if feat.shape[1] > target_frames:
            feat = feat[:, :target_frames]
        elif feat.shape[1] < target_frames:
            feat = np.pad(feat, ((0, 0), (0, target_frames - feat.shape[1])), mode="constant")
        return feat.astype(np.float32)

    def _resize_feature(self, feat, target_size=(128, 128)):
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)

        zoom_factor = (
            target_size[0] / feat.shape[0],
            target_size[1] / feat.shape[1]
        )
        resized = ndimage.zoom(feat, zoom_factor, order=1)
        return resized.astype(np.float32)
