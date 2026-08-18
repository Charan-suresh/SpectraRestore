#!/usr/bin/env python3
"""
Generate a synthetic demonstration dataset for testing and dry-running
the SpectraRestore pipeline before the full KLA dataset is uploaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def generate_synthetic_sem_pair(
    seed: int,
    gt_size: int = 256,
    scale: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate clean GT with periodic structures and degraded downsampled input."""
    rng = np.random.RandomState(seed)
    
    # 1. Clean SEM ground truth (periodic lines, vias, and grain texture)
    y, x = np.meshgrid(np.linspace(0, 1, gt_size), np.linspace(0, 1, gt_size), indexing="ij")
    
    freq_x = rng.choice([16, 24, 32, 48])
    freq_y = rng.choice([16, 24, 32, 48])
    pattern_type = rng.choice(["lines", "vias", "grid"])
    
    if pattern_type == "lines":
        structure = 0.5 + 0.4 * np.sin(2 * np.pi * freq_x * x)
    elif pattern_type == "vias":
        structure = 0.5 + 0.4 * np.sin(2 * np.pi * freq_x * x) * np.sin(2 * np.pi * freq_y * y)
    else:
        structure = 0.5 + 0.25 * np.sin(2 * np.pi * freq_x * x) + 0.25 * np.cos(2 * np.pi * freq_y * y)
        
    # Add substrate grain noise
    grain = 0.03 * rng.randn(gt_size, gt_size)
    gt = np.clip(structure + grain, 0.0, 1.0).astype(np.float32)
    
    # 2. Degraded input: 2x downsample + speckle noise + additive Gaussian noise
    in_size = gt_size // scale
    # Area average downsampling (2x)
    deg_clean = gt.reshape(in_size, scale, in_size, scale).mean(axis=(1, 3))
    
    # Multiplicative speckle
    speckle = 1.0 + 0.12 * rng.randn(in_size, in_size)
    # Additive Gaussian
    gaussian = 0.04 * rng.randn(in_size, in_size)
    
    deg = deg_clean * speckle + gaussian
    # Note: degraded values may exceed [0, 1] per KLA specification
    return deg.astype(np.float32), gt.astype(np.float32)


def create_demo_dataset(
    data_root: str | Path = "data",
    num_train: int = 16,
    num_val: int = 4,
    gt_size: int = 256,
    scale: int = 2,
) -> None:
    root = Path(data_root)
    train_deg = root / "train" / "degraded"
    train_gt = root / "train" / "gt"
    val_deg = root / "val" / "degraded"
    val_gt = root / "val" / "gt"

    for d in [train_deg, train_gt, val_deg, val_gt]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[demo data] Generating {num_train} train pairs and {num_val} val pairs in '{root}'...")

    for i in range(num_train):
        deg, gt = generate_synthetic_sem_pair(seed=100 + i, gt_size=gt_size, scale=scale)
        stem = f"sem_sample_{i+1:04d}"
        # Save GT as uint8 PNG in [0, 1]
        Image.fromarray((gt * 255.0 + 0.5).astype(np.uint8)).save(train_gt / f"{stem}.png")
        # Save degraded as NPY to preserve out-of-range speckle values
        np.save(train_deg / f"{stem}.npy", deg)

    for i in range(num_val):
        deg, gt = generate_synthetic_sem_pair(seed=500 + i, gt_size=gt_size, scale=scale)
        stem = f"sem_val_sample_{i+1:04d}"
        Image.fromarray((gt * 255.0 + 0.5).astype(np.uint8)).save(val_gt / f"{stem}.png")
        np.save(val_deg / f"{stem}.npy", deg)

    print(f"[demo data] Successfully created demo dataset in {root.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create synthetic demonstration dataset")
    p.add_argument("--data_root", type=str, default="data", help="Output data directory")
    p.add_argument("--num_train", type=int, default=16)
    p.add_argument("--num_val", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    create_demo_dataset(args.data_root, args.num_train, args.num_val)


if __name__ == "__main__":
    main()
