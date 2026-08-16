"""
SpectraRestore training loop.

Example:
  python -m src.train --data_root data --preset default --iters 200000 --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# allow `python -m src.train` and `python src/train.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import make_dataloader
from src.losses import CompositeRestoreLoss
from src.metrics import MetricMeter
from src.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}


@torch.no_grad()
def validate(model: nn.Module, loader, device: torch.device, use_lpips: bool = True) -> dict:
    model.eval()
    meter = MetricMeter(use_lpips=use_lpips)
    if meter._lpips is not None:
        meter._lpips.to(device)
    for batch in loader:
        deg = batch["degraded"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda", dtype=torch.bfloat16):
            pred = model(deg)
        # match spatial size if needed
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = torch.nn.functional.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        meter.update(pred.float().clamp(0, 1), gt.float().clamp(0, 1))
    model.train()
    return meter.compute() | {"score": meter.combined_score()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SpectraRestore (NAFNet-SR2×)")
    p.add_argument("--data_root", type=str, required=True, help="Dataset root with train/ and val/")
    p.add_argument("--preset", type=str, default="default", choices=["default", "large", "fast", "tiny"])
    p.add_argument("--out_dir", type=str, default="weights")
    p.add_argument("--iters", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--gt_crop", type=int, default=256)
    p.add_argument("--degrade_aug_p", type=float, default=0.3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_every", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader = make_dataloader(
        args.data_root,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gt_crop=args.gt_crop,
        degrade_aug_p=args.degrade_aug_p,
    )
    try:
        val_loader = make_dataloader(
            args.data_root,
            split="val",
            batch_size=1,
            num_workers=max(1, args.num_workers // 2),
            gt_crop=args.gt_crop,
            degrade_aug_p=0.0,
            shuffle=False,
        )
    except FileNotFoundError:
        print("[train] No val/ split found — using 5% of train as val is recommended. Continuing without val.")
        val_loader = None

    model = build_model(args.preset).to(device)
    print(f"[train] model={args.preset}  params={model.num_params()/1e6:.2f}M  device={device}")

    criterion = CompositeRestoreLoss(
        total_iters=args.iters,
        use_lpips=not args.no_lpips,
    ).to(device)
    if criterion.lpips is not None and criterion.lpips.available:
        criterion.lpips.lpips.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9), weight_decay=1e-4)
    scaler = GradScaler(enabled=device.type == "cuda")
    ema = EMA(model, decay=args.ema_decay)

    start_iter = 0
    best_score = -1e9
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        start_iter = ckpt.get("iter", 0)
        best_score = ckpt.get("best_score", best_score)
        print(f"[train] resumed from {args.resume} @ iter {start_iter}")

    def lr_at(step: int) -> float:
        # cosine decay
        if step >= args.iters:
            return args.min_lr
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * step / args.iters))

    model.train()
    data_iter = iter(train_loader)
    t0 = time.time()

    for step in range(start_iter + 1, args.iters + 1):
        lr = lr_at(step)
        for g in optim.param_groups:
            g["lr"] = lr

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        deg = batch["degraded"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda", dtype=torch.bfloat16):
            pred = model(deg)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred, size=gt.shape[-2:], mode="bilinear", align_corners=False
                )
            loss, logs = criterion(pred.float(), gt.float(), step=step)

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()
        ema.update(model)

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            ips = args.log_every * args.batch_size / max(elapsed, 1e-6)
            print(
                f"[{step}/{args.iters}] loss={logs['loss/total']:.4f} "
                f"char={logs['loss/char']:.4f} ssim={logs['loss/ssim']:.4f} "
                f"fft={logs['loss/fft']:.4f} lpips={logs['loss/lpips']:.4f} "
                f"lr={lr:.2e}  {ips:.1f} img/s"
            )
            t0 = time.time()

        if val_loader is not None and step % args.val_every == 0:
            # validate EMA weights
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            metrics = validate(model, val_loader, device, use_lpips=not args.no_lpips)
            model.load_state_dict(backup)
            print(
                f"[val @ {step}] PSNR={metrics['psnr']:.3f}  SSIM={metrics['ssim']:.4f}  "
                f"LPIPS={metrics['lpips']:.4f}  score={metrics['score']:.4f}"
            )
            if metrics["score"] > best_score:
                best_score = metrics["score"]
                torch.save(
                    {
                        "iter": step,
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "preset": args.preset,
                        "best_score": best_score,
                        "metrics": metrics,
                    },
                    out_dir / "best.pt",
                )
                print(f"[val] new best → {out_dir / 'best.pt'}")

        if step % args.save_every == 0 or step == args.iters:
            torch.save(
                {
                    "iter": step,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optim": optim.state_dict(),
                    "preset": args.preset,
                    "best_score": best_score,
                },
                out_dir / f"ckpt_{step:06d}.pt",
            )
            # also always keep latest EMA as the shippable weights
            torch.save(
                {"model": ema.state_dict(), "preset": args.preset, "iter": step},
                out_dir / "last_ema.pt",
            )

    print("[train] done.")


if __name__ == "__main__":
    main()
