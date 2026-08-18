#!/usr/bin/env python3
"""
Benchmark validation metrics (SSIM, pSNR, LPIPS, Latency) comparing:
  - Degraded input baseline (bilinear 2x upsampled)
  - SpectraRestore restored output

Outputs the exact markdown table required for Slide 6 and reports.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import load_checkpoint, resolve_weights
from src.dataset import make_dataloader
from src.metrics import MetricMeter, psnr, ssim
from src.model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SpectraRestore Validation Benchmarking")
    p.add_argument("--data_root", type=str, default="data", help="Path to data/ with val/ directory")
    p.add_argument("--weights", type=str, default="", help="Path to checkpoint (.pt)")
    p.add_argument("--preset", type=str, default="", help="Override model preset (default/fast/tiny)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_lpips", action="store_true", help="Disable LPIPS calculation")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    try:
        val_loader = make_dataloader(
            args.data_root,
            split="val",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            gt_crop=0,  # full image resolution
            degrade_aug_p=0.0,
            shuffle=False,
        )
    except Exception as e:
        print(f"[benchmark] Could not load validation set from {args.data_root}: {e}")
        return

    # Load model
    weights_path = resolve_weights(args.weights or None)
    state, preset = load_checkpoint(weights_path, device)
    if args.preset:
        preset = args.preset

    model = build_model(preset).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    meter_deg = MetricMeter(use_lpips=not args.no_lpips)
    meter_restored = MetricMeter(use_lpips=not args.no_lpips)
    if meter_deg._lpips is not None:
        meter_deg._lpips.to(device)
    if meter_restored._lpips is not None:
        meter_restored._lpips.to(device)

    total_infer_time = 0.0
    num_samples = 0

    print(f"[benchmark] Running evaluation on {len(val_loader)} batches...")
    for batch in val_loader:
        deg = batch["degraded"].to(device)
        gt = batch["gt"].to(device)

        # 1. Baseline: bilinear upsampled degraded input
        deg_up = F.interpolate(deg, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        meter_deg.update(deg_up.float().clamp(0, 1), gt.float().clamp(0, 1))

        # 2. Restored: model forward pass
        # Pad to multiple of 16
        _, _, h, w = deg.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        deg_pad = F.pad(deg, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else deg

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        use_amp = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(deg_pad)
        pred = pred[..., : h * model.scale, : w * model.scale]

        if device.type == "cuda":
            torch.cuda.synchronize()
        total_infer_time += time.perf_counter() - t0

        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)

        meter_restored.update(pred.float().clamp(0, 1), gt.float().clamp(0, 1))
        num_samples += deg.shape[0]

    deg_res = meter_deg.compute()
    rest_res = meter_restored.compute()
    avg_latency_ms = (total_infer_time / max(num_samples, 1)) * 1000.0

    print("\n" + "=" * 60)
    print("SLIDE 6 VALIDATION RESULTS TABLE")
    print("=" * 60)
    print(f"| Metric | Degraded input (baseline) | Ours (restored) | Delta |")
    print(f"|---|---|---|---|")
    print(f"| SSIM ↑ | {deg_res['ssim']:.4f} | {rest_res['ssim']:.4f} | {rest_res['ssim'] - deg_res['ssim']:+.4f} |")
    print(f"| pSNR (dB) ↑ | {deg_res['psnr']:.2f} | {rest_res['psnr']:.2f} | {rest_res['psnr'] - deg_res['psnr']:+.2f} dB |")
    lp_deg_str = f"{deg_res['lpips']:.4f}" if not math.isnan(deg_res['lpips']) else "N/A"
    lp_rest_str = f"{rest_res['lpips']:.4f}" if not math.isnan(rest_res['lpips']) else "N/A"
    lp_delta_str = f"{rest_res['lpips'] - deg_res['lpips']:+.4f}" if not math.isnan(deg_res['lpips']) else "N/A"
    print(f"| LPIPS ↓ | {lp_deg_str} | {lp_rest_str} | {lp_delta_str} |")
    print(f"| Inference (ms/img, FP16) | — | {avg_latency_ms:.2f} ms | — |")
    print("=" * 60)


if __name__ == "__main__":
    import math
    main()
