#!/usr/bin/env python3
"""
SpectraRestore Dataset Pre-Flight Check

Strictly validates dataset structure, file counts, extensions, pairing,
and 2x scale geometry before starting any training or evaluation.
Fails immediately with clear diagnostic logs if any discrepancy is found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy", ".npz", ".pt"}


def is_valid_image_file(path: Path) -> bool:
    """Return True if path is a recognized non-hidden image file."""
    if not path.is_file():
        return False
    name = path.name
    if name.startswith(".") or name.startswith("._") or name == "Thumbs.db":
        return False
    return path.suffix.lower() in IMG_EXTS


def get_image_files(directory: Path) -> Dict[str, Path]:
    """Scan directory for valid image files mapped by stem (ignoring hidden files)."""
    if not directory.is_dir():
        return {}
    files: Dict[str, Path] = {}
    for p in directory.iterdir():
        if is_valid_image_file(p):
            files[p.stem] = p
    return files


def validate_split(
    split_dir: Path, split_name: str, scale: int = 2, check_geometry: bool = True
) -> Tuple[bool, int, List[str]]:
    """Validate a specific dataset split (train or val)."""
    errors: List[str] = []
    if not split_dir.is_dir():
        errors.append(f"Directory missing: {split_dir}")
        return False, 0, errors

    deg_dir = split_dir / "degraded"
    gt_dir = split_dir / "gt"

    # Also check alternate conventions if degraded/gt aren't directly present
    if not deg_dir.is_dir() or not gt_dir.is_dir():
        for d_name, g_name in [("lq", "hq"), ("input", "target"), ("noisy", "clean"), ("low", "high")]:
            if (split_dir / d_name).is_dir() and (split_dir / g_name).is_dir():
                deg_dir = split_dir / d_name
                gt_dir = split_dir / g_name
                break

    if not deg_dir.is_dir():
        errors.append(f"Degraded directory missing under {split_dir} (expected '{split_dir}/degraded')")
    if not gt_dir.is_dir():
        errors.append(f"Ground-truth directory missing under {split_dir} (expected '{split_dir}/gt')")

    if errors:
        return False, 0, errors

    deg_files = get_image_files(deg_dir)
    gt_files = get_image_files(gt_dir)

    deg_stems = set(deg_files.keys())
    gt_stems = set(gt_files.keys())

    if len(deg_stems) == 0:
        errors.append(f"No valid image files found in {deg_dir}")
    if len(gt_stems) == 0:
        errors.append(f"No valid image files found in {gt_dir}")

    missing_gt = sorted(deg_stems - gt_stems)
    missing_deg = sorted(gt_stems - deg_stems)

    if missing_gt:
        sample_missing = ", ".join(missing_gt[:5]) + ("..." if len(missing_gt) > 5 else "")
        errors.append(f"{len(missing_gt)} degraded image(s) lack corresponding GT in {split_name} (e.g. {sample_missing})")
    if missing_deg:
        sample_missing = ", ".join(missing_deg[:5]) + ("..." if len(missing_deg) > 5 else "")
        errors.append(f"{len(missing_deg)} GT image(s) lack corresponding degraded in {split_name} (e.g. {sample_missing})")

    paired_count = len(deg_stems & gt_stems)

    # Optional geometry check on first few samples
    if check_geometry and paired_count > 0 and not errors:
        try:
            from PIL import Image
            for stem in sorted(deg_stems & gt_stems)[:10]:
                d_p = deg_files[stem]
                g_p = gt_files[stem]
                
                if d_p.suffix.lower() == ".npy":
                    d_shape = np.load(d_p, mmap_mode="r").shape[:2]
                else:
                    with Image.open(d_p) as im:
                        d_shape = (im.height, im.width)
                        
                if g_p.suffix.lower() == ".npy":
                    g_shape = np.load(g_p, mmap_mode="r").shape[:2]
                else:
                    with Image.open(g_p) as im:
                        g_shape = (im.height, im.width)
                        
                expected_shape = (d_shape[0] * scale, d_shape[1] * scale)
                if g_shape != expected_shape:
                    errors.append(
                        f"Geometry mismatch for {stem}: degraded is {d_shape}, "
                        f"GT is {g_shape}, expected {expected_shape} (scale={scale})"
                    )
                    break
        except Exception as e:
            errors.append(f"Failed during geometry validation: {e}")

    return len(errors) == 0, paired_count, errors


def validate_dataset(
    data_root: str | Path,
    scale: int = 2,
    check_geometry: bool = True,
    require_val: bool = True,
) -> Dict[str, int]:
    """
    Validate the entire dataset at data_root.
    Raises RuntimeError if validation fails.
    Returns dictionary with counts.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise RuntimeError(
            f"Dataset preflight FAILED: root directory '{root}' does not exist.\n"
            f"Please place your dataset under '{root}' with train/ and val/ folders."
        )

    all_errors: List[str] = []
    counts: Dict[str, int] = {}

    train_ok, train_cnt, train_errs = validate_split(
        root / "train", "train", scale=scale, check_geometry=check_geometry
    )
    counts["train"] = train_cnt
    if not train_ok:
        all_errors.extend([f"[train] {e}" for e in train_errs])

    val_dir = root / "val"
    if val_dir.is_dir() or require_val:
        val_ok, val_cnt, val_errs = validate_split(
            val_dir, "val", scale=scale, check_geometry=check_geometry
        )
        counts["val"] = val_cnt
        if not val_ok:
            all_errors.extend([f"[val] {e}" for e in val_errs])
    else:
        counts["val"] = 0

    print("=" * 60)
    print("SPECTRARESTORE DATASET PRE-FLIGHT CHECK")
    print("=" * 60)
    print(f"Dataset root: {root.resolve()}")
    print(f"train/degraded : {counts.get('train', 0)}")
    print(f"train/gt       : {counts.get('train', 0)}")
    print(f"val/degraded   : {counts.get('val', 0)}")
    print(f"val/gt         : {counts.get('val', 0)}")
    print("-" * 60)

    if all_errors:
        print("Dataset preflight FAILED.\n")
        for err in all_errors:
            print(f"  ❌ {err}")
        print("\nTraining has been blocked to prevent invalid runs.")
        print("=" * 60)
        raise RuntimeError(f"Dataset preflight failed with {len(all_errors)} error(s).")

    print("Pairing: PASS")
    print("Geometry (2x scale): PASS")
    print("Dataset: PASS")
    print("=" * 60)
    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SpectraRestore Dataset Pre-Flight Validator")
    p.add_argument("--data_root", type=str, default="data", help="Path to dataset root")
    p.add_argument("--scale", type=int, default=2, help="Super-resolution scaling factor (default: 2)")
    p.add_argument("--no_geometry", action="store_true", help="Skip shape geometry validation")
    p.add_argument("--allow_no_val", action="store_true", help="Allow missing val/ directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        validate_dataset(
            data_root=args.data_root,
            scale=args.scale,
            check_geometry=not args.no_geometry,
            require_val=not args.allow_no_val,
        )
    except RuntimeError as e:
        sys.exit(1)


if __name__ == "__main__":
    main()
