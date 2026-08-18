#!/usr/bin/env python3
"""End-to-end smoke test without the real dataset."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.losses import CompositeRestoreLoss
from src.model import build_model
from src.dataset import PairedRestoreDataset, _load_gray
from evaluate import load_gray, save_gray


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    model = build_model("tiny").to(device)
    print(f"[smoke] params={model.num_params()/1e6:.3f}M")

    # synthetic pair: GT clean grid, degraded = downsample + noise (may exceed [0,1])
    gt = torch.zeros(2, 1, 256, 256, device=device)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, 256, device=device),
        torch.linspace(0, 1, 256, device=device),
        indexing="ij",
    )
    gt[:, 0] = 0.5 + 0.4 * torch.sin(40 * np.pi * xx) * torch.sin(40 * np.pi * yy)
    deg = torch.nn.functional.avg_pool2d(gt, 2)
    deg = deg * (1 + 0.1 * torch.randn_like(deg)) + 0.05 * torch.randn_like(deg)

    pred = model(deg)
    assert pred.shape == gt.shape, f"shape mismatch {pred.shape} vs {gt.shape}"
    print(f"[smoke] forward OK  in={tuple(deg.shape)} out={tuple(pred.shape)}")

    crit = CompositeRestoreLoss(total_iters=100, use_lpips=False).to(device)
    loss, logs = crit(pred, gt, step=0)
    loss.backward()
    print(f"[smoke] loss OK  total={logs['loss/total']:.4f}")

    # save a tiny checkpoint and run evaluate.py round-trip
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Regression: scale low-valued 16-bit TIFFs by dtype range, not observed max.
        dark16 = tmp / "dark16.tiff"
        Image.fromarray(np.array([[100]], dtype=np.uint16)).save(dark16)
        expected = 100 / 65535
        assert np.isclose(_load_gray(dark16)[0, 0], expected), "dataset 16-bit scaling failed"
        assert np.isclose(load_gray(dark16)[0, 0], expected), "inference 16-bit scaling failed"
        uint16_npy = tmp / "dark16.npy"
        np.save(uint16_npy, np.array([[100]], dtype=np.uint16))
        assert np.isclose(_load_gray(uint16_npy)[0, 0], expected), "dataset NPY scaling failed"
        assert np.isclose(load_gray(uint16_npy)[0, 0], expected), "inference NPY scaling failed"
        save_gray(tmp / "roundtrip.tiff", np.array([[expected]], dtype=np.float32), source_dtype=np.dtype("uint16"))
        assert np.asarray(Image.open(tmp / "roundtrip.tiff")).dtype == np.uint16, "16-bit output was not preserved"

        # Regression: pair geometry must be valid unless resizing is explicitly requested.
        pair_root = tmp / "pairs" / "train"
        (pair_root / "degraded").mkdir(parents=True)
        (pair_root / "gt").mkdir()
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(pair_root / "degraded" / "bad.png")
        Image.fromarray(np.zeros((15, 16), dtype=np.uint8)).save(pair_root / "gt" / "bad.png")
        try:
            PairedRestoreDataset(tmp / "pairs", gt_crop=16)[0]
            raise AssertionError("misaligned pair was accepted")
        except ValueError as error:
            assert "Misaligned pair" in str(error)

        ckpt = tmp / "tiny.pt"
        torch.save({"model": model.state_dict(), "preset": "tiny"}, ckpt)

        in_dir = tmp / "in"
        out_dir = tmp / "out"
        in_dir.mkdir()
        # write a degraded png
        arr = deg[0, 0].detach().cpu().numpy()
        # allow values outside [0,1] in npy to test that path
        np.save(in_dir / "sample.npy", arr)

        # also a clipped png for PIL path
        Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(in_dir / "sample.png")

        from evaluate import load_checkpoint, restore_image, resolve_weights
        # direct call rather than subprocess
        state, preset = load_checkpoint(ckpt, device)
        m2 = build_model(preset).to(device)
        m2.load_state_dict(state)
        m2.eval()
        restored = restore_image(m2, arr, device)
        assert restored.shape == (256, 256), restored.shape
        print(f"[smoke] evaluate path OK  restored={restored.shape}  range=[{restored.min():.3f},{restored.max():.3f}]")

    print("[smoke] ALL PASSED")


if __name__ == "__main__":
    main()
