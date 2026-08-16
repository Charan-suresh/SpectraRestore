#!/usr/bin/env python3
"""
SpectraRestore — KLA-compliant evaluation / inference script.

Usage (exactly as required by the hackathon brief):
  python evaluate.py --input_dir <path_to_test_images> --output_dir <path_to_write_restored>

The script:
  1. Auto-loads trained weights from ./weights/best.pt (or WEIGHTS_PATH / --weights)
  2. Restores every image in --input_dir
  3. Writes restored outputs to --output_dir
  4. Runs with zero manual edits — tested as a standalone .py (not a notebook)

Optional:
  python evaluate.py --input_dir test/ --output_dir outputs/ --weights weights/best.pt --preset default
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import build_model

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}


def load_gray(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        from PIL import Image

        arr = np.asarray(Image.open(path), dtype=np.float32)
        if arr.max() > 1.5 and arr.max() <= 255:
            arr = arr / 255.0
        elif arr.max() > 255:
            arr = arr / 65535.0
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        else:
            arr = arr[..., 0]
    return arr.astype(np.float32)


def save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr, 0.0, 1.0)
    if path.suffix.lower() == ".npy":
        np.save(path, arr.astype(np.float32))
        return
    from PIL import Image

    img = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))
    img.save(path)


def resolve_weights(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"Weights not found: {p}")
        return p
    candidates = [
        ROOT / "weights" / "best.pt",
        ROOT / "weights" / "last_ema.pt",
        ROOT / "weights" / "model.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "No weights found. Place a checkpoint at weights/best.pt "
        "or pass --weights <path>."
    )


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    preset = ckpt.get("preset", "default")
    # prefer EMA weights when present
    if "ema" in ckpt:
        state = ckpt["ema"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    return state, preset


@torch.no_grad()
def restore_image(model: torch.nn.Module, arr: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # 1×1×H×W
    # pad to multiple of 16 for U-Net downsampling (4 levels → 16)
    _, _, h, w = x.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    use_amp = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        y = model(x)
    y = y[..., : h * model.scale, : w * model.scale]
    return y.float().squeeze().cpu().numpy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SpectraRestore evaluation / inference")
    p.add_argument("--input_dir", type=str, required=True, help="Directory of degraded test images")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to write restored images")
    p.add_argument("--weights", type=str, default="", help="Optional path to .pt checkpoint")
    p.add_argument("--preset", type=str, default="", help="Override model preset (default/fast/tiny)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ext", type=str, default="", help="Force output extension (e.g. .png). Default: keep input ext.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    images = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        # also allow nested single level
        images = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file())
    if not images:
        raise FileNotFoundError(f"No images found in {in_dir}")

    weights_path = resolve_weights(args.weights or None)
    state, preset = load_checkpoint(weights_path, device)
    if args.preset:
        preset = args.preset

    model = build_model(preset).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    print(f"[evaluate] weights={weights_path}  preset={preset}  params={model.num_params()/1e6:.2f}M")
    print(f"[evaluate] {len(images)} images  {in_dir} → {out_dir}  device={device}")

    total_s = 0.0
    for i, path in enumerate(images, 1):
        arr = load_gray(path)
        t0 = time.perf_counter()
        restored = restore_image(model, arr, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_s += time.perf_counter() - t0

        # Keep the same filename as the input so KLA can match outputs for scoring.
        out_ext = args.ext if args.ext else path.suffix
        if not out_ext.startswith("."):
            out_ext = "." + out_ext
        out_name = f"{path.stem}{out_ext}"
        out_path = out_dir / out_name
        try:
            rel = path.relative_to(in_dir)
            if len(rel.parts) > 1:
                out_path = out_dir / rel.parent / out_name
        except ValueError:
            pass
        save_gray(out_path, restored)
        if i == 1 or i % 20 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}] {path.name} → {out_path.name}  shape={restored.shape}")

    avg_ms = (total_s / len(images)) * 1000.0
    print(f"[evaluate] done. avg inference {avg_ms:.2f} ms/image ({len(images)} images)")


if __name__ == "__main__":
    main()
