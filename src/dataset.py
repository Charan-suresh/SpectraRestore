"""
Paired degraded ↔ ground-truth dataset for SpectraRestore.

Expected layout (flexible — auto-detects common conventions):

  data/
    train/
      degraded/   *.png|*.tif|*.npy|*.pt
      gt/         matching filenames (or paired by stem)
    val/
      degraded/
      gt/

Also accepts flat paired naming:
  sample_001_lq.png / sample_001_hq.png
  sample_001_input.png / sample_001_target.png
  sample_001_degraded.png / sample_001_gt.png
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.image_io import normalize_image_array

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy", ".npz", ".pt"}


def _load_gray(path: Path) -> np.ndarray:
    """Load a grayscale image as float32, preserving native range (no forced [0,1] clip)."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = normalize_image_array(np.load(path))
    elif suffix in {".npz", ".pt"}:
        t = torch.load(path, map_location="cpu", weights_only=True)
        raw = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        arr = normalize_image_array(raw)
    else:
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("Pillow is required to load image files") from e
        with Image.open(path) as img:
            arr = normalize_image_array(np.asarray(img))

    if arr.ndim == 3:
        # take first channel / luminance
        if arr.shape[-1] in (3, 4):
            arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        else:
            arr = arr[..., 0]
    return arr.astype(np.float32)


def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def _discover_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Discover (degraded, gt) pairs under root."""
    pairs: list[tuple[Path, Path]] = []

    # Convention A: root/{degraded,gt} or root/{lq,hq} or root/{input,target}
    for deg_name, gt_name in (
        ("degraded", "gt"),
        ("lq", "hq"),
        ("input", "target"),
        ("noisy", "clean"),
        ("low", "high"),
    ):
        deg_dir, gt_dir = root / deg_name, root / gt_name
        if deg_dir.is_dir() and gt_dir.is_dir():
            deg_files = {p.stem: p for p in _list_images(deg_dir)}
            gt_files = {p.stem: p for p in _list_images(gt_dir)}
            for stem in sorted(set(deg_files) & set(gt_files)):
                pairs.append((deg_files[stem], gt_files[stem]))
            if pairs:
                return pairs

    # Convention B: flat folder with suffix pairs
    files = _list_images(root)
    by_stem: dict[str, Path] = {p.stem: p for p in files}
    suffix_pairs = (
        ("_lq", "_hq"),
        ("_input", "_target"),
        ("_degraded", "_gt"),
        ("_noisy", "_clean"),
        ("_low", "_high"),
        ("_LR", "_HR"),
        ("_lr", "_hr"),
    )
    used: set[str] = set()
    for deg_suf, gt_suf in suffix_pairs:
        for stem, path in by_stem.items():
            if stem in used:
                continue
            if stem.endswith(deg_suf):
                base = stem[: -len(deg_suf)]
                gt_stem = base + gt_suf
                if gt_stem in by_stem:
                    pairs.append((path, by_stem[gt_stem]))
                    used.add(stem)
                    used.add(gt_stem)
    if pairs:
        return pairs

    # Convention C: two sibling folders named train/val already handled by caller;
    # also try root itself if it has two subdirs that look image-like
    subdirs = [d for d in root.iterdir() if d.is_dir()] if root.is_dir() else []
    if len(subdirs) == 2:
        a, b = sorted(subdirs, key=lambda p: p.name)
        # heuristic: 'gt/hq/clean/target/high' is GT
        gt_keywords = ("gt", "hq", "clean", "target", "high", "hr")
        if any(k in b.name.lower() for k in gt_keywords):
            deg_dir, gt_dir = a, b
        elif any(k in a.name.lower() for k in gt_keywords):
            deg_dir, gt_dir = b, a
        else:
            deg_dir, gt_dir = a, b
        deg_files = {p.stem: p for p in _list_images(deg_dir)}
        gt_files = {p.stem: p for p in _list_images(gt_dir)}
        for stem in sorted(set(deg_files) & set(gt_files)):
            pairs.append((deg_files[stem], gt_files[stem]))

    return pairs


def degrade_augment(deg: np.ndarray, p: float = 0.3) -> np.ndarray:
    """
    OOD weapon: with probability p, inject EXTRA randomised degradation into the
    already-degraded input (speckle / Gaussian / mild blur). Target unchanged.
    Blur is included because KLA's webinar deck lists blur alongside noise/resolution.
    """
    if random.random() >= p:
        return deg
    out = deg.copy()
    choice = random.random()
    if choice < 0.4:
        # extra multiplicative speckle
        strength = random.uniform(0.02, 0.15)
        out = out * (1.0 + strength * np.random.randn(*out.shape).astype(np.float32))
    elif choice < 0.8:
        # extra additive Gaussian
        sigma = random.uniform(0.01, 0.08)
        out = out + sigma * np.random.randn(*out.shape).astype(np.float32)
    else:
        # mild blur (KLA PPT factor); keep light so detail isn't destroyed
        try:
            from scipy.ndimage import gaussian_filter

            out = gaussian_filter(out, sigma=random.uniform(0.3, 0.9)).astype(np.float32)
        except ImportError:
            # fallback: simple 3x3 box blur without scipy
            k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
            padded = np.pad(out, 1, mode="reflect")
            out = (
                k[0, 0] * padded[:-2, :-2]
                + k[0, 1] * padded[:-2, 1:-1]
                + k[0, 2] * padded[:-2, 2:]
                + k[1, 0] * padded[1:-1, :-2]
                + k[1, 1] * padded[1:-1, 1:-1]
                + k[1, 2] * padded[1:-1, 2:]
                + k[2, 0] * padded[2:, :-2]
                + k[2, 1] * padded[2:, 1:-1]
                + k[2, 2] * padded[2:, 2:]
            ).astype(np.float32)
    return out.astype(np.float32)


def paired_augment(
    deg: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned geometric aug: flips + 90° rotations only."""
    if random.random() < 0.5:
        deg = np.flip(deg, axis=1).copy()
        gt = np.flip(gt, axis=1).copy()
    if random.random() < 0.5:
        deg = np.flip(deg, axis=0).copy()
        gt = np.flip(gt, axis=0).copy()
    k = random.randint(0, 3)
    if k:
        deg = np.rot90(deg, k).copy()
        gt = np.rot90(gt, k).copy()
    return deg, gt


def random_paired_crop(
    deg: np.ndarray, gt: np.ndarray, gt_crop: int = 256, scale: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned crop: GT crop size → input crop = gt_crop // scale."""
    in_crop = gt_crop // scale
    gh, gw = gt.shape[-2], gt.shape[-1]
    dh, dw = deg.shape[-2], deg.shape[-1]

    # ensure GT is exactly scale× the degraded size (or crop within both)
    if gh < gt_crop or gw < gt_crop or dh < in_crop or dw < in_crop:
        # pad if needed
        pad_gt_h = max(0, gt_crop - gh)
        pad_gt_w = max(0, gt_crop - gw)
        pad_d_h = max(0, in_crop - dh)
        pad_d_w = max(0, in_crop - dw)
        if pad_gt_h or pad_gt_w:
            gt = np.pad(gt, ((0, pad_gt_h), (0, pad_gt_w)), mode="reflect")
        if pad_d_h or pad_d_w:
            deg = np.pad(deg, ((0, pad_d_h), (0, pad_d_w)), mode="reflect")
        gh, gw = gt.shape[-2], gt.shape[-1]
        dh, dw = deg.shape[-2], deg.shape[-1]

    # pick crop in GT space, map to LR by integer division
    top = random.randint(0, gh - gt_crop)
    left = random.randint(0, gw - gt_crop)
    # snap to scale grid so LR crop aligns
    top = (top // scale) * scale
    left = (left // scale) * scale
    if top + gt_crop > gh:
        top = ((gh - gt_crop) // scale) * scale
    if left + gt_crop > gw:
        left = ((gw - gt_crop) // scale) * scale

    gt_c = gt[top : top + gt_crop, left : left + gt_crop]
    deg_c = deg[top // scale : top // scale + in_crop, left // scale : left // scale + in_crop]
    return deg_c, gt_c


class PairedRestoreDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        gt_crop: int = 256,
        scale: int = 2,
        degrade_aug_p: float = 0.3,
        geometric_aug: bool = True,
        resize_misaligned_gt: bool = False,
        transform: Callable | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.gt_crop = gt_crop
        self.scale = scale
        self.degrade_aug_p = degrade_aug_p if split == "train" else 0.0
        self.geometric_aug = geometric_aug and split == "train"
        self.resize_misaligned_gt = resize_misaligned_gt
        self.transform = transform

        split_root = self.root / split if (self.root / split).is_dir() else self.root
        self.pairs = _discover_pairs(split_root)
        if not self.pairs:
            raise FileNotFoundError(
                f"No paired images found under {split_root}. "
                "Expected degraded/gt subfolders or *_lq/*_hq naming."
            )
        print(f"[dataset] {split}: {len(self.pairs)} pairs from {split_root}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        deg_path, gt_path = self.pairs[idx]
        deg = _load_gray(deg_path)
        gt = _load_gray(gt_path)

        # Pair alignment is a data invariant. Resizing is only an explicit opt-in
        # compatibility path for preprocessed datasets.
        expected_h, expected_w = deg.shape[0] * self.scale, deg.shape[1] * self.scale
        if gt.shape[0] != expected_h or gt.shape[1] != expected_w:
            if not self.resize_misaligned_gt:
                raise ValueError(
                    f"Misaligned pair: {deg_path} is {deg.shape}, but {gt_path} is {gt.shape}; "
                    f"expected target shape {(expected_h, expected_w)} for scale={self.scale}. "
                    "Fix the dataset or enable resize_misaligned_gt explicitly."
                )
            from PIL import Image

            gt_img = Image.fromarray(gt)
            gt = np.asarray(
                gt_img.resize((expected_w, expected_h), Image.BICUBIC), dtype=np.float32
            )

        if self.split == "train":
            if self.geometric_aug:
                deg, gt = paired_augment(deg, gt)
            deg, gt = random_paired_crop(deg, gt, self.gt_crop, self.scale)
            if self.degrade_aug_p > 0:
                deg = degrade_augment(deg, self.degrade_aug_p)

        deg_t = torch.from_numpy(np.ascontiguousarray(deg)).unsqueeze(0)  # 1×H×W
        gt_t = torch.from_numpy(np.ascontiguousarray(gt)).unsqueeze(0)

        if self.transform:
            deg_t, gt_t = self.transform(deg_t, gt_t)

        return {"degraded": deg_t, "gt": gt_t, "name": deg_path.stem}


def make_dataloader(
    root: str | Path,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    gt_crop: int = 256,
    degrade_aug_p: float = 0.3,
    resize_misaligned_gt: bool = False,
    shuffle: bool | None = None,
) -> torch.utils.data.DataLoader:
    from torch.utils.data import DataLoader

    ds = PairedRestoreDataset(
        root=root,
        split=split,
        gt_crop=gt_crop,
        degrade_aug_p=degrade_aug_p,
        resize_misaligned_gt=resize_misaligned_gt,
    )
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=split == "train",
    )
