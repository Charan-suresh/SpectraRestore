#!/usr/bin/env python3
"""
SpectraRestore Curated Visual Evidence Generator

Generates high-resolution, deterministic, multi-zoom visual evidence
comparing:
  [Degraded Input] → [Our Restoration] → [Ground Truth] → [|Error| Heatmap]

Features:
  - Deterministic case selection (fixed seed / curated list)
  - Prioritizes fine periodic / repetitive semiconductor structures
  - Generates 3 view levels per case: Full-frame, Zoom Crop 1 (2x), Zoom Crop 2 (4x)
  - Synchronized bounding box & crop coordinates across all panels
  - Saves publication-ready figure to outputs/visual_evidence.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import load_gray


def find_high_frequency_crop(
    image: np.ndarray, crop_size: int = 64
) -> Tuple[int, int]:
    """Find the top-left (y, x) of the crop with maximum high-frequency gradient energy."""
    h, w = image.shape[-2:]
    if h <= crop_size or w <= crop_size:
        return 0, 0
    # Sobel-like gradient energy
    gy = np.abs(image[1:, :] - image[:-1, :])
    gx = np.abs(image[:, 1:] - image[:, :-1])
    energy = gy[:, :-1] + gx[:-1, :]
    
    eh, ew = energy.shape
    best_energy = -1.0
    best_y, best_x = (h - crop_size) // 2, (w - crop_size) // 2  # default center
    
    step = max(8, crop_size // 4)
    for y in range(0, eh - crop_size + 1, step):
        for x in range(0, ew - crop_size + 1, step):
            e_sum = float(np.sum(energy[y : y + crop_size, x : x + crop_size]))
            if e_sum > best_energy:
                best_energy = e_sum
                best_y, best_x = y, x
                
    return best_y, best_x


def generate_visual_evidence(
    restored_dir: str | Path,
    degraded_dir: str | Path,
    gt_dir: Optional[str | Path] = None,
    output_path: str | Path = "outputs/visual_evidence.png",
    curated_cases: Optional[List[str]] = None,
    num_cases: int = 2,
    seed: int = 42,
) -> Path:
    """Generate multi-zoom visual comparison figure."""
    rest_p = Path(restored_dir)
    deg_p = Path(degraded_dir)
    gt_p = Path(gt_dir) if gt_dir else None
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    restored_files = sorted([p for p in rest_p.rglob("*") if p.is_file() and not p.name.startswith(".")])
    if not restored_files:
        raise FileNotFoundError(f"No restored outputs found in {rest_p}. Run inference first.")

    by_stem = {p.stem: p for p in restored_files}

    selected_stems: List[str] = []
    if curated_cases:
        for c in curated_cases:
            stem = Path(c).stem
            if stem in by_stem:
                selected_stems.append(stem)

    if not selected_stems:
        # Deterministic selection: prioritize files with periodic/array/dense names or highest spectral energy
        rng = np.random.RandomState(seed)
        all_stems = sorted(by_stem.keys())
        
        # Check for periodic keywords in filenames
        periodic_keywords = ["array", "periodic", "grid", "checker", "dense", "line", "via", "sem"]
        prio = [s for s in all_stems if any(k in s.lower() for k in periodic_keywords)]
        remaining = [s for s in all_stems if s not in prio]
        
        pool = prio + remaining
        selected_stems = pool[:num_cases]

    print(f"[visual evidence] Selected cases: {', '.join(selected_stems)}")

    # 3 zoom rows per case: Full-frame, Zoom-1 (2x), Zoom-2 (4x)
    total_rows = len(selected_stems) * 3
    has_gt = gt_p and gt_p.is_dir()
    total_cols = 4 if has_gt else 2

    fig = plt.figure(figsize=(4.2 * total_cols, 3.8 * total_rows), dpi=180)
    col_titles = ["Degraded Input", "Our Restoration", "Ground Truth", "|Error| Heatmap"] if has_gt else ["Degraded Input", "Our Restoration"]

    row_idx = 0
    for case_idx, stem in enumerate(selected_stems):
        r_file = by_stem[stem]
        d_candidates = list(deg_p.glob(f"{stem}.*"))
        g_candidates = list(gt_p.glob(f"{stem}.*")) if has_gt else []

        if not d_candidates:
            continue

        deg_img = load_gray(d_candidates[0])
        rest_img = load_gray(r_file)
        gt_img = load_gray(g_candidates[0]) if g_candidates else None

        # Ensure spatial matching for display
        gh, gw = rest_img.shape[-2:]
        if deg_img.shape != (gh, gw):
            # Bilinear upsample degraded input to GT resolution for visual alignment
            t_deg = torch.from_numpy(deg_img).unsqueeze(0).unsqueeze(0)
            deg_img = F.interpolate(t_deg, size=(gh, gw), mode="bilinear", align_corners=False).squeeze().numpy()

        deg_c = np.clip(deg_img, 0, 1)
        rest_c = np.clip(rest_img, 0, 1)
        gt_c = np.clip(gt_img, 0, 1) if gt_img is not None else None
        err_map = np.abs(rest_c - gt_c) if gt_c is not None else None

        # Zoom 1 (2x zoom, crop size H/2)
        crop1_sz = min(gh, gw) // 2
        y1, x1 = find_high_frequency_crop(gt_c if gt_c is not None else rest_c, crop1_sz)

        # Zoom 2 (4x zoom, crop size H/4)
        crop2_sz = min(gh, gw) // 4
        y2, x2 = find_high_frequency_crop(
            (gt_c if gt_c is not None else rest_c)[y1 : y1 + crop1_sz, x1 : x1 + crop1_sz], crop2_sz
        )
        y2, x2 = y1 + y2, x1 + x2

        views = [
            ("Full Frame", (0, 0, gh, gw)),
            (f"Zoom 1 (2× Crop: [{x1}:{x1+crop1_sz}, {y1}:{y1+crop1_sz}])", (y1, x1, crop1_sz, crop1_sz)),
            (f"Zoom 2 (4× High-Detail: [{x2}:{x2+crop2_sz}, {y2}:{y2+crop2_sz}])", (y2, x2, crop2_sz, crop2_sz)),
        ]

        for view_title, (vy, vx, vh, vw) in views:
            d_v = deg_c[vy : vy + vh, vx : vx + vw]
            r_v = rest_c[vy : vy + vh, vx : vx + vw]
            g_v = gt_c[vy : vy + vh, vx : vx + vw] if gt_c is not None else None
            e_v = err_map[vy : vy + vh, vx : vx + vw] if err_map is not None else None

            # 1. Degraded
            ax0 = fig.add_subplot(total_rows, total_cols, row_idx * total_cols + 1)
            ax0.imshow(d_v, cmap="gray", vmin=0, vmax=1)
            if row_idx == 0:
                ax0.set_title(col_titles[0], fontsize=13, fontweight="bold", pad=8)
            ax0.set_ylabel(f"Case {case_idx+1}: {stem}\n{view_title}", fontsize=10, fontweight="semibold")
            ax0.set_xticks([])
            ax0.set_yticks([])

            # Draw zoom bounding boxes on full frame
            if vh == gh and vw == gw:
                rect1 = patches.Rectangle((x1, y1), crop1_sz, crop1_sz, linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--")
                rect2 = patches.Rectangle((x2, y2), crop2_sz, crop2_sz, linewidth=1.5, edgecolor="yellow", facecolor="none", linestyle=":")
                ax0.add_patch(rect1)
                ax0.add_patch(rect2)

            # 2. Restored
            ax1 = fig.add_subplot(total_rows, total_cols, row_idx * total_cols + 2)
            ax1.imshow(r_v, cmap="gray", vmin=0, vmax=1)
            if row_idx == 0:
                ax1.set_title(col_titles[1], fontsize=13, fontweight="bold", pad=8)
            ax1.axis("off")
            if vh == gh and vw == gw:
                rect1_r = patches.Rectangle((x1, y1), crop1_sz, crop1_sz, linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--")
                rect2_r = patches.Rectangle((x2, y2), crop2_sz, crop2_sz, linewidth=1.5, edgecolor="yellow", facecolor="none", linestyle=":")
                ax1.add_patch(rect1_r)
                ax1.add_patch(rect2_r)

            # 3. Ground Truth (if present)
            if has_gt:
                ax2 = fig.add_subplot(total_rows, total_cols, row_idx * total_cols + 3)
                ax2.imshow(g_v, cmap="gray", vmin=0, vmax=1)
                if row_idx == 0:
                    ax2.set_title(col_titles[2], fontsize=13, fontweight="bold", pad=8)
                ax2.axis("off")
                if vh == gh and vw == gw:
                    rect1_g = patches.Rectangle((x1, y1), crop1_sz, crop1_sz, linewidth=1.5, edgecolor="cyan", facecolor="none", linestyle="--")
                    rect2_g = patches.Rectangle((x2, y2), crop2_sz, crop2_sz, linewidth=1.5, edgecolor="yellow", facecolor="none", linestyle=":")
                    ax2.add_patch(rect1_g)
                    ax2.add_patch(rect2_g)

                # 4. Error Heatmap
                ax3 = fig.add_subplot(total_rows, total_cols, row_idx * total_cols + 4)
                im_e = ax3.imshow(e_v, cmap="turbo", vmin=0.0, vmax=0.8)
                if row_idx == 0:
                    ax3.set_title(col_titles[3], fontsize=13, fontweight="bold", pad=8)
                ax3.axis("off")
                cbar = fig.colorbar(im_e, ax=ax3, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)

            row_idx += 1

    plt.tight_layout()
    plt.savefig(out_p, bbox_inches="tight", dpi=180)
    plt.close()

    print(f"[visual evidence] Saved: {out_p.resolve()}")
    return out_p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SpectraRestore Visual Evidence Generator")
    p.add_argument("--restored_dir", type=str, default="outputs/val_restored", help="Path to restored output images")
    p.add_argument("--degraded_dir", type=str, default="data/val/degraded", help="Path to degraded input images")
    p.add_argument("--gt_dir", type=str, default="data/val/gt", help="Path to ground truth images")
    p.add_argument("--output", type=str, default="outputs/visual_evidence.png", help="Path to save output figure")
    p.add_argument("--cases", nargs="*", default=None, help="Optional curated case stems")
    p.add_argument("--num_cases", type=int, default=2, help="Number of cases to visualize (default: 2)")
    p.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        generate_visual_evidence(
            restored_dir=args.restored_dir,
            degraded_dir=args.degraded_dir,
            gt_dir=args.gt_dir,
            output_path=args.output,
            curated_cases=args.cases,
            num_cases=args.num_cases,
            seed=args.seed,
        )
    except Exception as e:
        print(f"[visual evidence] Error generating visual evidence: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
