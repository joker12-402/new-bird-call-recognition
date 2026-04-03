import os
import sys
import json
import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm

# 把项目根目录加入环境变量，这样才能 import models 和 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline import ImprovedBirdNet
from models.attention_net import ImprovedBirdNetWithAttention
from utils.dataset import MFCCDataset, MFCCTemporalDataset, MFCCEnergyDataset, ThreeFeatureDataset

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_class_weights(labels, num_classes):
    label_counts = Counter(labels)
    total_samples = len(labels)
    weights = np.zeros(num_classes, dtype=np.float32)
    for l, c in label_counts.items():
        weights[int(l)] = total_samples / (num_classes * c)
    weights = weights / (weights.sum() + 1e-12)
    return torch.tensor(weights, dtype=torch.float32)

# ... [保留原代码中的 train_one_epoch, eval_epoch, build_model_and_dataset, run_kfold 函数] ...

if __name__ == "__main__":
    AUDIO_DIR = "D:/paper/data/processed_audio"
    METADATA = "D:/paper/data/features/metadata.json"
    LABEL_MAPPING = "D:/paper/data/features/label_mapping.json"
    OUTPUT_ROOT = "D:/paper/big_paper_results"

    models_to_run = [
        "model_c_no_cr",
        # 如果你想跑 model_b_cr，可以在这里加上
    ]

    for mname in models_to_run:
        run_kfold(
            model_name=mname,
            audio_dir=AUDIO_DIR,
            metadata_path=METADATA,
            label_mapping_path=LABEL_MAPPING,
            output_root=OUTPUT_ROOT,
            n_splits=5,
            seed=42,
            epochs=80,
            batch_size=32,
            lr=1e-3,
            patience=10
        )
