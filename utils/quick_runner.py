# -*- coding: utf-8 -*-
"""
utils/quick_runner.py

Minimal runnable quick compare implementation.

- Fixed split 7/1/2 (train/val/test) with stratification
- Supports features: mfcc, energy(mel), chroma, pcen, spectral_contrast
- Supports no_cr / cr via models.baseline / models.attention_net
- Exports per-model artifacts:
  - config.json
  - history.json
  - result.json
  - best_val_report.txt / .json
  - test_report.txt / .json
- Exports summary:
  - quick_all_results.json (under output_root)

Repo dependencies:
- torch, librosa, numpy, scipy, scikit-learn, tqdm
"""

import os
import json
import random
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import librosa
from scipy import ndimage
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

from models.baseline import ImprovedBirdNet
from models.attention_net import ImprovedBirdNetWithAttention


# =========================
# 0) Repro
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 1) Audio / feature extract
# =========================
def safe_load_audio(audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(audio_path, sr=sr)
    if y is None or len(y) == 0:
        y = np.zeros(sr, dtype=np.float32)
    return y, sr


def resolve_audio_path(audio_dir: str, file_path: str) -> str:
    """
    metadata item["file_path"] could be xxx.npy or xxx.wav or without suffix.
    We always resolve to .wav under audio_dir.
    """
    fp = str(file_path)
    if fp.lower().endswith(".wav"):
        wav_name = fp
    elif fp.lower().endswith(".npy"):
        wav_name = fp[:-4] + ".wav"
    else:
        wav_name = fp + ".wav"
    return os.path.join(audio_dir, wav_name)


def extract_mfcc(audio_path: str, sr: int = 16000, n_mfcc: int = 40, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    try:
        y, sr = safe_load_audio(audio_path, sr=sr)
        pre_emphasis = 0.97
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
            hop_length=hop_length, fmax=sr // 2
        )
        return mfcc.astype(np.float32)
    except Exception as e:
        print(f"[WARN] MFCC failed: {audio_path} | {e}")
        return np.zeros((n_mfcc, 100), dtype=np.float32)


def extract_energy_mel_db(audio_path: str, sr: int = 16000, n_mels: int = 40, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    try:
        y, sr = safe_load_audio(audio_path, sr=sr)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length,
            fmax=sr // 2
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.astype(np.float32)
    except Exception as e:
        print(f"[WARN] Mel(Energy) failed: {audio_path} | {e}")
        return np.zeros((n_mels, 100), dtype=np.float32)


def extract_chroma(audio_path: str, sr: int = 16000, n_chroma: int = 12, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    try:
        y, sr = safe_load_audio(audio_path, sr=sr)
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_chroma=n_chroma,
            n_fft=n_fft, hop_length=hop_length
        )
        return chroma.astype(np.float32)
    except Exception as e:
        print(f"[WARN] Chroma failed: {audio_path} | {e}")
        return np.zeros((n_chroma, 100), dtype=np.float32)


def extract_pcen(audio_path: str, sr: int = 16000, n_mels: int = 40, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    try:
        y, sr = safe_load_audio(audio_path, sr=sr)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length,
            fmax=sr // 2
        )
        pcen = librosa.pcen(mel + 1e-6)
        return pcen.astype(np.float32)
    except Exception as e:
        print(f"[WARN] PCEN failed: {audio_path} | {e}")
        return np.zeros((n_mels, 100), dtype=np.float32)


def extract_spectral_contrast(audio_path: str, sr: int = 16000, n_bands: int = 6, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    try:
        y, sr = safe_load_audio(audio_path, sr=sr)
        sc = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_bands=n_bands,
            n_fft=n_fft, hop_length=hop_length
        )
        # librosa returns (n_bands+1, frames)
        return sc.astype(np.float32)
    except Exception as e:
        print(f"[WARN] SpectralContrast failed: {audio_path} | {e}")
        return np.zeros((n_bands + 1, 100), dtype=np.float32)


def fix_time_length(feat: np.ndarray, target_frames: int = 100) -> np.ndarray:
    """
    Ensure feat has shape [F, T] and T == target_frames
    """
    feat = np.asarray(feat)
    if feat.ndim == 1:
        # [T] -> [1, T]
        if feat.shape[0] > target_frames:
            feat = feat[:target_frames]
        elif feat.shape[0] < target_frames:
            feat = np.pad(feat, (0, target_frames - feat.shape[0]), mode="constant")
        return feat.reshape(1, -1).astype(np.float32)

    # [F, T]
    if feat.shape[1] > target_frames:
        feat = feat[:, :target_frames]
    elif feat.shape[1] < target_frames:
        feat = np.pad(feat, ((0, 0), (0, target_frames - feat.shape[1])), mode="constant")
    return feat.astype(np.float32)


def resize_feature(feat: np.ndarray, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Resize [F, T] -> [H, W] using ndimage.zoom
    """
    if feat.ndim == 1:
        feat = feat.reshape(1, -1)
    zoom_factor = (target_size[0] / feat.shape[0], target_size[1] / feat.shape[1])
    resized = ndimage.zoom(feat, zoom_factor, order=1)
    return resized.astype(np.float32)


# =========================
# 2) Dataset / model configs
# =========================
MODEL_CONFIGS: Dict[str, Dict] = {
    "baseline_mfcc": {"features": ["mfcc"], "use_attention": False},
    "model_b_no_cr": {"features": ["mfcc", "energy"], "use_attention": False},
    "model_b_cr": {"features": ["mfcc", "energy"], "use_attention": True},
    "model_b_chroma_no_cr": {"features": ["mfcc", "energy", "chroma"], "use_attention": False},
    "model_b_chroma_cr": {"features": ["mfcc", "energy", "chroma"], "use_attention": True},
    "model_b_pcen_no_cr": {"features": ["mfcc", "energy", "pcen"], "use_attention": False},
    "model_b_pcen_cr": {"features": ["mfcc", "energy", "pcen"], "use_attention": True},
    "model_b_spectral_no_cr": {"features": ["mfcc", "energy", "spectral"], "use_attention": False},
    "model_b_spectral_cr": {"features": ["mfcc", "energy", "spectral"], "use_attention": True},
}


class MultiFeatureDataset(Dataset):
    def __init__(
        self,
        metadata: List[dict],
        audio_dir: str,
        feature_names: List[str],
        indices: np.ndarray,
        is_train: bool,
        target_size: Tuple[int, int] = (128, 128),
        augment_prob: float = 0.4,
        stats: Optional[Dict[str, float]] = None,
        use_stats: bool = False,
    ):
        self.metadata = metadata
        self.audio_dir = audio_dir
        self.feature_names = feature_names
        self.indices = np.asarray(indices)
        self.is_train = bool(is_train)
        self.target_size = target_size
        self.augment_prob = float(augment_prob)
        self.use_stats = bool(use_stats)
        self.stats = stats or {}

    def __len__(self):
        return len(self.indices)

    def _extract_one(self, feat_name: str, audio_path: str) -> np.ndarray:
        if feat_name == "mfcc":
            feat = extract_mfcc(audio_path, n_mfcc=40)
        elif feat_name == "energy":
            feat = extract_energy_mel_db(audio_path, n_mels=40)
        elif feat_name == "chroma":
            feat = extract_chroma(audio_path, n_chroma=12)
        elif feat_name == "pcen":
            feat = extract_pcen(audio_path, n_mels=40)
        elif feat_name == "spectral":
            feat = extract_spectral_contrast(audio_path, n_bands=6)
        else:
            raise ValueError(f"Unknown feature: {feat_name}")

        feat = fix_time_length(feat, target_frames=100)

        if self.use_stats:
            mean = float(self.stats.get(f"{feat_name}_mean", 0.0))
            std = float(self.stats.get(f"{feat_name}_std", 1.0))
            std = max(std, 1e-6)
            feat = (feat - mean) / std

        feat = resize_feature(feat, self.target_size)
        return feat

    def _augment(self, x: np.ndarray) -> np.ndarray:
        # x: [C, H, W]
        augmented = x.copy()

        # time shift along W
        if random.random() < 0.6:
            shift = int(augmented.shape[2] * 0.15 * (random.random() - 0.5))
            augmented = np.roll(augmented, shift, axis=2)
            if shift > 0:
                augmented[:, :, :shift] = 0
            elif shift < 0:
                augmented[:, :, shift:] = 0

        # gaussian noise
        if random.random() < 0.4:
            noise = np.random.normal(0, 0.03, augmented.shape).astype(np.float32)
            augmented = augmented + noise

        return np.clip(augmented, -3, 3).astype(np.float32)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        item = self.metadata[idx]
        audio_path = resolve_audio_path(self.audio_dir, item["file_path"])
        label = int(item["label"])

        feats = []
        for fn in self.feature_names:
            feats.append(self._extract_one(fn, audio_path))

        x = np.stack(feats, axis=0).astype(np.float32)  # [C, H, W]

        if self.is_train and random.random() < self.augment_prob:
            x = self._augment(x)

        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


def build_model_and_dataset(
    model_name: str,
    metadata: List[dict],
    audio_dir: str,
    indices: np.ndarray,
    num_classes: int,
    stats: Optional[Dict[str, float]],
    is_train: bool,
    use_stats: bool,
):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_name: {model_name}")

    cfg = MODEL_CONFIGS[model_name]
    feature_names = cfg["features"]
    in_channels = len(feature_names)

    ds = MultiFeatureDataset(
        metadata=metadata,
        audio_dir=audio_dir,
        feature_names=feature_names,
        indices=indices,
        is_train=is_train,
        augment_prob=0.4 if is_train else 0.0,
        stats=stats,
        use_stats=use_stats,
    )

    if cfg["use_attention"]:
        model = ImprovedBirdNetWithAttention(num_classes=num_classes, in_channels=in_channels)
    else:
        model = ImprovedBirdNet(num_classes=num_classes, in_channels=in_channels)

    return model, ds


# =========================
# 3) Train / Eval
# =========================
def calculate_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    cnt = Counter(labels)
    total = len(labels)
    w = np.zeros(num_classes, dtype=np.float32)
    for l, c in cnt.items():
        w[int(l)] = total / (num_classes * c)
    w = w / (w.sum() + 1e-12)
    return torch.tensor(w, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += float(loss.item()) * x.size(0)
        preds = torch.argmax(logits, dim=1)
        all_labels.extend(y.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    f1w = f1_score(all_labels, all_preds, average="weighted", zero_division=0) if all_labels else 0.0
    return float(epoch_loss), float(acc), float(f1w)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, label_names: Optional[List[str]] = None) -> Tuple[float, float, float, float, str, dict]:
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = criterion(logits, y)

        running_loss += float(loss.item()) * x.size(0)
        preds = torch.argmax(logits, dim=1)
        all_labels.extend(y.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    f1w = f1_score(all_labels, all_preds, average="weighted", zero_division=0) if all_labels else 0.0
    f1m = f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0

    report_text = ""
    report_dict = {}

    if label_names is not None and all_labels:
        report_text = classification_report(
            all_labels, all_preds,
            labels=list(range(len(label_names))),
            target_names=label_names,
            digits=4, zero_division=0
        )
        report_dict = classification_report(
            all_labels, all_preds,
            labels=list(range(len(label_names))),
            target_names=label_names,
            output_dict=True,
            zero_division=0
        )

    return float(epoch_loss), float(acc), float(f1w), float(f1m), report_text, report_dict


# =========================
# 4) Split / Stats
# =========================
def make_quick_split(metadata: List[dict], seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_indices = np.arange(len(metadata))
    all_labels = np.array([int(item["label"]) for item in metadata])

    trainval_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=seed,
        stratify=all_labels
    )

    trainval_labels = all_labels[trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.125,  # 0.1 overall => 0.1 / 0.8 = 0.125 of trainval
        random_state=seed,
        stratify=trainval_labels
    )

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def calculate_dataset_stats(
    metadata: List[dict],
    audio_dir: str,
    indices: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Compute mean/std over train set only (avoid leakage).
    """
    stats: Dict[str, float] = {}
    for fn in feature_names:
        vals = []
        for i in tqdm(indices, desc=f"Stats-{fn}", leave=False):
            item = metadata[int(i)]
            audio_path = resolve_audio_path(audio_dir, item["file_path"])

            if fn == "mfcc":
                feat = extract_mfcc(audio_path, n_mfcc=40)
            elif fn == "energy":
                feat = extract_energy_mel_db(audio_path, n_mels=40)
            elif fn == "chroma":
                feat = extract_chroma(audio_path, n_chroma=12)
            elif fn == "pcen":
                feat = extract_pcen(audio_path, n_mels=40)
            elif fn == "spectral":
                feat = extract_spectral_contrast(audio_path, n_bands=6)
            else:
                raise ValueError(f"Unknown feature: {fn}")

            feat = fix_time_length(feat, 100)
            vals.append(feat.reshape(-1))

        if not vals:
            concat = np.array([0.0], dtype=np.float32)
        else:
            concat = np.concatenate(vals, axis=0).astype(np.float32)

        stats[f"{fn}_mean"] = float(np.mean(concat))
        stats[f"{fn}_std"] = float(np.std(concat) + 1e-6)

    return stats


# =========================
# 5) Quick runner (single model)
# =========================
@dataclass
class QuickArgs:
    model_name: str
    audio_dir: str
    metadata_path: str
    label_mapping_path: str
    output_root: str
    seed: int = 42
    epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    patience: int = 10
    use_stats: bool = False


def run_quick_one(args: QuickArgs) -> dict:
    set_seed(args.seed)

    if not os.path.isdir(args.audio_dir):
        raise FileNotFoundError(f"audio_dir not found: {args.audio_dir}")
    if not os.path.isfile(args.metadata_path):
        raise FileNotFoundError(f"metadata not found: {args.metadata_path}")
    if not os.path.isfile(args.label_mapping_path):
        raise FileNotFoundError(f"label_mapping not found: {args.label_mapping_path}")

    os.makedirs(args.output_root, exist_ok=True)
    out_dir = os.path.join(args.output_root, f"{args.model_name}_quick_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    # load mapping
    with open(args.label_mapping_path, "r", encoding="utf-8") as f:
        lm = json.load(f)
    label_to_name = lm.get("label_to_name", {})
    num_classes = len(label_to_name) if label_to_name else None
    label_names = [label_to_name[str(i)] for i in range(len(label_to_name))] if label_to_name else None

    # load metadata
    with open(args.metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, list) or len(metadata) == 0:
        raise ValueError("metadata.json must be a non-empty list")

    all_labels = [int(item["label"]) for item in metadata]
    if num_classes is None:
        num_classes = len(set(all_labels))

    # split
    train_idx, val_idx, test_idx = make_quick_split(metadata, seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # stats (optional)
    feature_names = MODEL_CONFIGS[args.model_name]["features"]
    stats = None
    if args.use_stats:
        stats = calculate_dataset_stats(metadata, args.audio_dir, train_idx, feature_names)
        with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    # build
    model, train_ds = build_model_and_dataset(
        args.model_name, metadata, args.audio_dir, train_idx, num_classes,
        stats, is_train=True, use_stats=args.use_stats
    )
    _, val_ds = build_model_and_dataset(
        args.model_name, metadata, args.audio_dir, val_idx, num_classes,
        stats, is_train=False, use_stats=args.use_stats
    )
    _, test_ds = build_model_and_dataset(
        args.model_name, metadata, args.audio_dir, test_idx, num_classes,
        stats, is_train=False, use_stats=args.use_stats
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = model.to(device)

    # class weights from train split
    train_labels_for_weight = [all_labels[int(i)] for i in train_idx]
    class_w = calculate_class_weights(train_labels_for_weight, num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # save config
    config = {
        "model_name": args.model_name,
        "mode": "quick",
        "seed": args.seed,
        "audio_dir": args.audio_dir,
        "metadata_path": args.metadata_path,
        "label_mapping_path": args.label_mapping_path,
        "output_root": args.output_root,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
        "use_stats": args.use_stats,
        "features": feature_names,
        "use_attention": MODEL_CONFIGS[args.model_name]["use_attention"],
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "device": str(device),
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    best_val_f1 = -1.0
    early_stop = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1_weighted": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_weighted": [],
        "val_f1_macro": [],
    }

    best_model_path = os.path.join(out_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1w = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1w, va_f1m, va_txt, va_dict = eval_epoch(model, val_loader, criterion, device, label_names)

        scheduler.step(va_f1w)

        history["train_loss"].append(float(tr_loss))
        history["train_acc"].append(float(tr_acc))
        history["train_f1_weighted"].append(float(tr_f1w))
        history["val_loss"].append(float(va_loss))
        history["val_acc"].append(float(va_acc))
        history["val_f1_weighted"].append(float(va_f1w))
        history["val_f1_macro"].append(float(va_f1m))

        print(
            f"[{args.model_name}] Epoch {epoch}/{args.epochs} | "
            f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} f1w={tr_f1w:.4f} | "
            f"Val: loss={va_loss:.4f} acc={va_acc:.4f} f1w={va_f1w:.4f} f1m={va_f1m:.4f}"
        )

        if va_f1w > best_val_f1:
            best_val_f1 = float(va_f1w)
            early_stop = 0

            torch.save(model.state_dict(), best_model_path)

            with open(os.path.join(out_dir, "best_val_report.txt"), "w", encoding="utf-8") as f:
                f.write(va_txt or "")
            with open(os.path.join(out_dir, "best_val_report.json"), "w", encoding="utf-8") as f:
                json.dump(va_dict or {}, f, ensure_ascii=False, indent=2)

            print(f"[{args.model_name}] Save best model (val f1w={best_val_f1:.4f})")
        else:
            early_stop += 1
            print(f"[{args.model_name}] EarlyStop {early_stop}/{args.patience}")
            if early_stop >= args.patience:
                print(f"[{args.model_name}] Early stop triggered.")
                break

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # test with best model
    if not os.path.isfile(best_model_path):
        # fallback: no improvement at all (rare), save current
        torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    te_loss, te_acc, te_f1w, te_f1m, te_txt, te_dict = eval_epoch(model, test_loader, criterion, device, label_names)

    with open(os.path.join(out_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(te_txt or "")
    with open(os.path.join(out_dir, "test_report.json"), "w", encoding="utf-8") as f:
        json.dump(te_dict or {}, f, ensure_ascii=False, indent=2)

    result = {
        "model_name": args.model_name,
        "mode": "quick",
        "seed": int(args.seed),
        "use_stats": bool(args.use_stats),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "best_val_f1_weighted": float(best_val_f1),
        "test_loss": float(te_loss),
        "test_acc": float(te_acc),
        "test_f1_weighted": float(te_f1w),
        "test_f1_macro": float(te_f1m),
        "device": str(device),
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"[{args.model_name}] Quick TEST | "
        f"acc={result['test_acc']:.4f} f1w={result['test_f1_weighted']:.4f} f1m={result['test_f1_macro']:.4f}"
    )

    return result


# =========================
# 6) Public API for entry script
# =========================
def run_quick_compare(
    models_to_run: List[str],
    audio_dir: str,
    metadata_path: str,
    label_mapping_path: str,
    output_root: str,
    seed: int = 42,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    use_stats: bool = False,
) -> List[dict]:
    """
    Called by scripts/train_quick_compare_models.py
    """
    os.makedirs(output_root, exist_ok=True)

    all_results: List[dict] = []
    print("[INFO] Quick compare started")
    print(f"[INFO] models_to_run = {models_to_run}")
    print(f"[INFO] output_root   = {output_root}")
    print(f"[INFO] seed={seed} epochs={epochs} bs={batch_size} lr={lr} patience={patience} use_stats={use_stats}")

    for m in models_to_run:
        if m not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model in --models: {m}. Available: {list(MODEL_CONFIGS.keys())}")

        try:
            r = run_quick_one(QuickArgs(
                model_name=m,
                audio_dir=audio_dir,
                metadata_path=metadata_path,
                label_mapping_path=label_mapping_path,
                output_root=output_root,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                use_stats=use_stats,
            ))
            all_results.append(r)
        except Exception as e:
            print(f"\n[ERROR] Model failed: {m}")
            print(str(e))
            print("[TRACEBACK]")
            traceback.print_exc()
            all_results.append({
                "model_name": m,
                "mode": "quick",
                "seed": int(seed),
                "failed": True,
                "error": str(e),
            })

    # write global summary
    summary_path = os.path.join(output_root, "quick_all_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("[INFO] Quick compare summary (sorted by test_acc, desc)")
    ok = [r for r in all_results if not r.get("failed", False) and "test_acc" in r]
    ok_sorted = sorted(ok, key=lambda x: x["test_acc"], reverse=True)
    for r in ok_sorted:
        print(
            f"{r['model_name']:<24} | "
            f"Acc={r['test_acc']:.4f} | "
            f"F1_w={r['test_f1_weighted']:.4f} | "
            f"F1_m={r['test_f1_macro']:.4f} | "
            f"device={r['device']}"
        )
    print(f"[INFO] Saved: {summary_path}")
    print("=" * 100 + "\n")

    return all_results
