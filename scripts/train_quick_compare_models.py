# -*- coding: utf-8 -*-
"""
Quick compare entry (fixed split 7/1/2)

This file is intentionally kept as an *entry script only*.
Core quick logic should live in utils/quick_runner.py.

Example:
python scripts/train_quick_compare_models.py --audio_dir D:/paper/data/processed_audio --metadata D:/paper/data/features/metadata.json --label_mapping D:/paper/data/features/label_mapping.json --out_dir D:/paper/quick_compare_results
"""

import argparse
import os
import sys


# add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.quick_runner import run_quick_compare  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", type=str, required=True, help="Directory containing wav files")
    p.add_argument("--metadata", type=str, required=True, help="metadata.json (list of {file_path,label})")
    p.add_argument("--label_mapping", type=str, required=True, help="label_mapping.json (label_to_name)")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for quick results")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)

    p.add_argument("--use_stats", action="store_true", help="Enable feature standardization using train-set stats")

    # default models = the 9 you used
    p.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=[
            "baseline_mfcc",
            "model_b_no_cr",
            "model_b_cr",
            "model_b_chroma_no_cr",
            "model_b_chroma_cr",
            "model_b_pcen_no_cr",
            "model_b_pcen_cr",
            "model_b_spectral_no_cr",
            "model_b_spectral_cr",
        ],
        help="Model names to run"
    )
    return p.parse_args()


def main():
    args = parse_args()

    run_quick_compare(
        models_to_run=args.models,
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        label_mapping_path=args.label_mapping,
        output_root=args.out_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        use_stats=args.use_stats,
    )


if __name__ == "__main__":
    main()
