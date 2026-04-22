"""
scripts/train_kfold.py

5-fold cross validation entry + training logic for bird sound classification.

- Models:
  - baseline_mfcc
  - model_a
  - model_b
  - model_b_cr
  - model_c_no_cr 
  - model_c_cr
  - model_b_chroma_no_cr
  - model_b_chroma_cr
  - model_b_pcen_no_cr
  - model_b_pcen_cr
  - model_b_spectral_no_cr
  - model_b_spectral_cr

- Datasets (from utils/dataset.py):
  - MFCCDataset
  - MFCCTemporalDataset 
  - MFCCEnergyDataset
  - ThreeFeatureDataset
  - MultiFeatureDataset

Notes:
- Uses StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
- For each fold: train/val split from train_idx (90%/10%) with permutation under the same seed
- Saves per-fold artifacts and a final kfold_summary.json
"""


import os
import sys
import json
import random
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# add project root to PYTHONPATH so we can import models/utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline import ImprovedBirdNet
from models.attention_net import ImprovedBirdNetWithAttention
from utils.dataset import (
    MFCCDataset,
    MFCCTemporalDataset,
    MFCCEnergyDataset,
    ThreeFeatureDataset,
    MultiFeatureDataset,
)


# ============================================================
# 0) Reproducibility
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 1) Helpers
# ============================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def calculate_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    weights[c] ∝ 1 / count(c), normalized to sum=1
    """
    label_counts = Counter(labels)
    total = len(labels)

    w = np.zeros(num_classes, dtype=np.float32)
    for l, c in label_counts.items():
        l = int(l)
        w[l] = total / (num_classes * float(c))

    w = w / (w.sum() + 1e-12)
    return torch.tensor(w, dtype=torch.float32)


def mean_std(x: List[float]) -> Tuple[float, float]:
    return float(np.mean(x)), float(np.std(x))


# ============================================================
# 2) Train / Eval
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    all_y, all_pred = [], []

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        running_loss += float(loss.item()) * x.size(0)
        pred = torch.argmax(logits, dim=1)

        all_y.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(pred.detach().cpu().numpy().tolist())

        pbar.set_postfix(loss=float(loss.item()))

    epoch_loss = running_loss / max(1, len(loader.dataset))
    acc = accuracy_score(all_y, all_pred) if all_y else 0.0
    f1_w = f1_score(all_y, all_pred, average="weighted", zero_division=0) if all_y else 0.0
    return float(epoch_loss), float(acc), float(f1_w)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_names: Optional[List[str]] = None,
    split_name: str = "Val",
) -> Dict:
    model.eval()
    running_loss = 0.0
    all_y, all_pred = [], []

    pbar = tqdm(loader, desc=f"Eval {split_name}", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits, _ = model(x)
        loss = criterion(logits, y)

        running_loss += float(loss.item()) * x.size(0)
        pred = torch.argmax(logits, dim=1)

        all_y.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(pred.detach().cpu().numpy().tolist())

    loss = running_loss / max(1, len(loader.dataset))
    acc = accuracy_score(all_y, all_pred) if all_y else 0.0
    f1_w = f1_score(all_y, all_pred, average="weighted", zero_division=0) if all_y else 0.0
    f1_m = f1_score(all_y, all_pred, average="macro", zero_division=0) if all_y else 0.0

    report_text, report_dict = "", {}
    if label_names is not None and len(label_names) > 0 and all_y:
        report_text = classification_report(
            all_y,
            all_pred,
            labels=list(range(len(label_names))),
            target_names=label_names,
            digits=4,
            zero_division=0,
        )
        report_dict = classification_report(
            all_y,
            all_pred,
            labels=list(range(len(label_names))),
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        )

    return {
        "loss": float(loss),
        "acc": float(acc),
        "f1_weighted": float(f1_w),
        "f1_macro": float(f1_m),
        "report_text": report_text,
        "report_dict": report_dict,
        "y_true": all_y,
        "y_pred": all_pred,
    }


# ============================================================
# 3) Model + Dataset switch
# ============================================================
def build_model_and_dataset(
    model_name: str,
    metadata: List[dict],
    audio_dir: str,
    indices: np.ndarray,
    stats: Optional[dict],
    is_train: bool,
):
    """
    Note: stats is kept for compatibility; current datasets may ignore it.
    """
    labels = [int(item["label"]) for item in metadata]
    num_classes = len(set(labels))

    if model_name == "baseline_mfcc":
        model = ImprovedBirdNet(num_classes=num_classes, in_channels=1)
        dataset = MFCCDataset(metadata, audio_dir, indices=indices, is_train=is_train, stats=stats)

    elif model_name == "model_a":
        model = ImprovedBirdNet(num_classes=num_classes, in_channels=2)
        dataset = MFCCTemporalDataset(metadata, audio_dir, indices=indices, is_train=is_train, stats=stats)

    elif model_name == "model_b":
        # MFCC + Mel
        model = ImprovedBirdNet(num_classes=num_classes, in_channels=2)
        dataset = MFCCEnergyDataset(metadata, audio_dir, indices=indices, is_train=is_train, stats=stats)

    elif model_name == "model_b_cr":
        # MFCC + Mel + CR
        model = ImprovedBirdNetWithAttention(num_classes=num_classes, in_channels=2)
        dataset = MFCCEnergyDataset(metadata, audio_dir, indices=indices, is_train=is_train, stats=stats)

    elif model_name == "model_c_no_cr":
        # MFCC + Time + Mel
        model = ImprovedBirdNet(num_classes=num_classes, in_channels=3)
        dataset = ThreeFeatureDataset(metadata, audio_dir, indices=indices, is_train=is_train, stats=stats)

    elif model_name == "model_c_cr":
        # MFCC + Time + Mel + CR
        model = ImprovedBirdNetWithAttention(num_classes=num_classes, in_channels=3)
        dataset = ThreeFeatureDataset(metadata, audio_dir, indices=indices, is_train=is_train, stats=stats)

    elif model_name == "model_b_chroma_no_cr":
        # MFCC + Mel + Chroma
        model = ImprovedBirdNet(num_classes=num_classes, in_channels=3)
        dataset = MultiFeatureDataset(
            metadata=metadata,
            audio_dir=audio_dir,
            feature_names=["mfcc", "energy", "chroma"],
            indices=indices,
            is_train=is_train,
            stats=stats,
        )

    elif model_name == "model_b_chroma_cr":
        # MFCC + Mel + Chroma + CR
        model = ImprovedBirdNetWithAttention(num_classes=num_classes, in_channels=3)
        dataset = MultiFeatureDataset(
            metadata=metadata,
            audio_dir=audio_dir,
            feature_names=["mfcc", "energy", "chroma"],
            indices=indices,
            is_train=is_train,
            stats=stats,
        )

    elif model_name == "model_b_pcen_no_cr":
        # MFCC + Mel + PCEN
        model = ImprovedBirdNet(num_classes=num_classes, in_channels=3)
        dataset = MultiFeatureDataset(
            metadata=metadata,
            audio_dir=audio_dir,
            feature_names=["mfcc", "energy", "pcen"],
            indices=indices,
            is_train=is_train,
            stats=stats,
        )

    elif model_name == "model_b_pcen_cr":
        # MFCC + Mel + PCEN + CR
        model = ImprovedBirdNetWithAttention(num_classes=num_classes, in_channels=3)
        dataset = MultiFeatureDataset(
            metadata=metadata,
            audio_dir=audio_dir,
            feature_names=["mfcc", "energy", "pcen"],
            indices=indices,
            is_train=is_train,
            stats=stats,
        )

    elif model_name == "model_b_spectral_no_cr":
        # MFCC + Mel + Spectral Contrast
        model = ImprovedBirdNet(num_classes=num_classes, in_channels=3)
        dataset = MultiFeatureDataset(
            metadata=metadata,
            audio_dir=audio_dir,
            feature_names=["mfcc", "energy", "spectral"],
            indices=indices,
            is_train=is_train,
            stats=stats,
        )

    elif model_name == "model_b_spectral_cr":
        # MFCC + Mel + Spectral Contrast + CR
        model = ImprovedBirdNetWithAttention(num_classes=num_classes, in_channels=3)
        dataset = MultiFeatureDataset(
            metadata=metadata,
            audio_dir=audio_dir,
            feature_names=["mfcc", "energy", "spectral"],
            indices=indices,
            is_train=is_train,
            stats=stats,
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model, dataset


# ============================================================
# 4) K-Fold main
# ============================================================
def run_kfold(
    model_name: str,
    audio_dir: str,
    metadata_path: str,
    label_mapping_path: str,
    output_root: str,
    n_splits: int = 5,
    seed: int = 42,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    num_workers: int = 0,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
):
    set_seed(seed)

    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"--audio_dir not found: {audio_dir}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"--metadata not found: {metadata_path}")
    if not os.path.isfile(label_mapping_path):
        raise FileNotFoundError(f"--label_mapping not found: {label_mapping_path}")

    ensure_dir(output_root)
    out_dir = os.path.join(output_root, f"{model_name}_seed{seed}")
    ensure_dir(out_dir)

    config = dict(
        model_name=model_name,
        audio_dir=audio_dir,
        metadata=metadata_path,
        label_mapping=label_mapping_path,
        output_root=output_root,
        n_splits=n_splits,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        num_workers=num_workers,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
    )
    dump_json(config, os.path.join(out_dir, "config.json"))

    label_mapping = load_json(label_mapping_path)
    label_to_name = label_mapping.get("label_to_name", {})
    if isinstance(label_to_name, dict) and len(label_to_name) > 0:
        num_classes = len(label_to_name)
        label_names = [label_to_name[str(i)] for i in range(num_classes)]
    else:
        label_names = None

    metadata = load_json(metadata_path)
    if not isinstance(metadata, list) or len(metadata) == 0:
        raise ValueError("metadata.json is empty or invalid format (expected a non-empty list).")

    all_labels = [int(item["label"]) for item in metadata]
    all_indices = np.arange(len(metadata))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device = {device}")
    print(f"[INFO] Model  = {model_name}")
    print(f"[INFO] N      = {len(metadata)} | #classes = {len(set(all_labels))}")
    print(f"[INFO] Label distribution = {Counter(all_labels)}")

    stats = None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results: List[Dict] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(all_indices, all_labels), start=1):
        print(f"\n========== {model_name} | Fold {fold}/{n_splits} ==========")

        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        ensure_dir(fold_dir)

        train_idx = np.array(train_idx)
        perm = np.random.RandomState(seed + fold).permutation(len(train_idx))
        train_idx = train_idx[perm]

        n_train = int(len(train_idx) * 0.9)
        real_train_idx = train_idx[:n_train]
        val_idx = train_idx[n_train:]

        train_labels_for_weight = [all_labels[i] for i in real_train_idx]
        num_classes = len(set(all_labels))
        class_weights = calculate_class_weights(train_labels_for_weight, num_classes=num_classes).to(device)

        model, train_dataset = build_model_and_dataset(
            model_name=model_name,
            metadata=metadata,
            audio_dir=audio_dir,
            indices=real_train_idx,
            stats=stats,
            is_train=True,
        )
        _, val_dataset = build_model_and_dataset(
            model_name=model_name,
            metadata=metadata,
            audio_dir=audio_dir,
            indices=val_idx,
            stats=stats,
            is_train=False,
        )
        _, test_dataset = build_model_and_dataset(
            model_name=model_name,
            metadata=metadata,
            audio_dir=audio_dir,
            indices=np.array(test_idx),
            stats=stats,
            is_train=False,
        )

        pin = bool(torch.cuda.is_available())
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin
        )

        model = model.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        best_val_f1 = -1.0
        early_stop = 0

        history = {
            "train_loss": [], "train_acc": [], "train_f1_weighted": [],
            "val_loss": [], "val_acc": [], "val_f1_weighted": [], "val_f1_macro": []
        }

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc, tr_f1_w = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, grad_clip=grad_clip
            )
            val_res = eval_epoch(
                model, val_loader, criterion, device,
                label_names=label_names, split_name=f"Fold{fold}-Val"
            )

            scheduler.step(val_res["f1_weighted"])

            history["train_loss"].append(float(tr_loss))
            history["train_acc"].append(float(tr_acc))
            history["train_f1_weighted"].append(float(tr_f1_w))
            history["val_loss"].append(float(val_res["loss"]))
            history["val_acc"].append(float(val_res["acc"]))
            history["val_f1_weighted"].append(float(val_res["f1_weighted"]))
            history["val_f1_macro"].append(float(val_res["f1_macro"]))

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train loss={tr_loss:.4f}, acc={tr_acc:.4f}, f1_w={tr_f1_w:.4f} | "
                f"Val loss={val_res['loss']:.4f}, acc={val_res['acc']:.4f}, "
                f"f1_w={val_res['f1_weighted']:.4f}, f1_m={val_res['f1_macro']:.4f}"
            )

            if val_res["f1_weighted"] > best_val_f1:
                best_val_f1 = float(val_res["f1_weighted"])
                early_stop = 0

                torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pth"))
                dump_json(history, os.path.join(fold_dir, "history.json"))

                if val_res["report_text"]:
                    with open(os.path.join(fold_dir, "val_report.txt"), "w", encoding="utf-8") as f:
                        f.write(val_res["report_text"])
                if val_res["report_dict"]:
                    dump_json(val_res["report_dict"], os.path.join(fold_dir, "val_report.json"))

                print(f"[INFO] Saved best model. best_val_f1_w={best_val_f1:.4f}")
            else:
                early_stop += 1
                print(f"[INFO] EarlyStop: {early_stop}/{patience}")
                if early_stop >= patience:
                    print("[INFO] Early stopping triggered.")
                    break

        best_model_path = os.path.join(fold_dir, "best_model.pth")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        test_res = eval_epoch(
            model, test_loader, criterion, device,
            label_names=label_names, split_name=f"Fold{fold}-Test"
        )

        if test_res["report_text"]:
            with open(os.path.join(fold_dir, "test_report.txt"), "w", encoding="utf-8") as f:
                f.write(test_res["report_text"])
        if test_res["report_dict"]:
            dump_json(test_res["report_dict"], os.path.join(fold_dir, "test_report.json"))

        fold_results.append({
            "fold": int(fold),
            "best_val_f1_weighted": float(best_val_f1),
            "test_loss": float(test_res["loss"]),
            "test_acc": float(test_res["acc"]),
            "test_f1_weighted": float(test_res["f1_weighted"]),
            "test_f1_macro": float(test_res["f1_macro"]),
        })

        dump_json(fold_results[-1], os.path.join(fold_dir, "fold_result.json"))

    accs = [r["test_acc"] for r in fold_results]
    f1ws = [r["test_f1_weighted"] for r in fold_results]
    f1ms = [r["test_f1_macro"] for r in fold_results]

    acc_mean, acc_std = mean_std(accs)
    f1w_mean, f1w_std = mean_std(f1ws)
    f1m_mean, f1m_std = mean_std(f1ms)

    summary = {
        "model_name": model_name,
        "seed": seed,
        "n_splits": n_splits,
        "fold_results": fold_results,
        "accs": accs,
        "f1_weighted": f1ws,
        "f1_macro": f1ms,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "f1w_mean": f1w_mean,
        "f1w_std": f1w_std,
        "f1m_mean": f1m_mean,
        "f1m_std": f1m_std,
    }

    dump_json(summary, os.path.join(out_dir, "kfold_summary.json"))

    print(f"\n=== {model_name} | 5-fold summary ===")
    print(f"Acc : {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"F1_w: {f1w_mean:.4f} ± {f1w_std:.4f}")
    print(f"F1_m: {f1m_mean:.4f} ± {f1m_std:.4f}")

    return summary


# ============================================================
# 5) CLI
# ============================================================
def parse_models_arg(s: str) -> List[str]:
    """
    --models can be:
      - "model_b_cr"
      - "model_b,model_b_cr,model_c_cr"
      - "all"
    """
    s = (s or "").strip()
    if not s:
        return []

    if s.lower() == "all":
        return [
            "baseline_mfcc",
            "model_a",
            "model_b",
            "model_b_cr",
            "model_c_no_cr",
            "model_c_cr",
            "model_b_chroma_no_cr",
            "model_b_chroma_cr",
            "model_b_pcen_no_cr",
            "model_b_pcen_cr",
            "model_b_spectral_no_cr",
            "model_b_spectral_cr",
        ]

    return [m.strip() for m in s.split(",") if m.strip()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="5-fold training/evaluation (kfold)")

    parser.add_argument("--audio_dir", type=str, required=True, help="Directory of processed wav files")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata.json")
    parser.add_argument("--label_mapping", type=str, required=True, help="Path to label_mapping.json")
    parser.add_argument("--out_dir", type=str, required=True, help="Output root directory")

    parser.add_argument(
        "--models",
        type=str,
        default="model_b_cr",
        help='Comma-separated model names, e.g. "model_b,model_b_cr" or "all"',
    )

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    args = parser.parse_args()

    models_to_run = parse_models_arg(args.models)
    if not models_to_run:
        raise ValueError("Empty --models. Provide at least one model name or use --models all")

    for mname in models_to_run:
        run_kfold(
            model_name=mname,
            audio_dir=args.audio_dir,
            metadata_path=args.metadata,
            label_mapping_path=args.label_mapping,
            output_root=args.out_dir,
            n_splits=args.n_splits,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            num_workers=args.num_workers,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
        )
