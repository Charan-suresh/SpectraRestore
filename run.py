#!/usr/bin/env python3
"""
SpectraRestore — Official KLA Submission Entry Script.

Competition: KLA Problem Statement – AI-Based Restoration of Degraded Images
Team: SpectraRestore

Usage:
  python run.py <input-dir> <output-dir>

Or with optional flags:
  python run.py --input_dir <input-dir> --output_dir <output-dir> [--weights <path>] [--device <cuda|cpu>]

Compliance Highlights:
  ✅ Reads all .npy files from the input directory.
  ✅ Creates the output directory if it does not already exist.
  ✅ Generates one restored .npy file for every input file.
  ✅ Each output has the exact same filename as its corresponding input.
  ✅ Outputs are grayscale 2D float32 arrays with shape (H_target, W_target) = (2*H, 2*W).
  ✅ Output values are strictly within [0.0, 1.0] and contain NO NaN or Inf values.
  ✅ Restored images have the correct 2x target resolution.
  ✅ Runs offline on NVIDIA GPU (with clean CPU fallback) with no manual setup.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure repository root is in python path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import build_model, SpectraRestore
from src.image_io import normalize_image_array


def load_npy_image(path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load a degraded image from an .npy file.
    
    Returns:
        arr: 2D float32 array with shape (H, W).
        orig_shape: (H, W) of the input.
    """
    raw = np.load(path)
    # Normalize if stored as uint8/uint16; keep float range as-is
    arr = normalize_image_array(raw)
    
    # Handle possible extra channel dimensions: (H, W, 1), (1, H, W), (H, W, 3/4)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            # Standard luminosity weights for RGB(A)
            arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0, ...]
        else:
            arr = arr[..., 0]
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported array dimensions: {arr.shape} in {path}")
    
    # Clean any corrupted NaN/Inf in raw input before feeding to model
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = arr.astype(np.float32)
    return arr, (arr.shape[0], arr.shape[1])


def resolve_weights(explicit_path: str | None = None) -> Path:
    """
    Locate the trained model checkpoint automatically.
    Prioritizes models/ directory, followed by weights/ and custom paths.
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified weights file not found: {p}")
    
    search_candidates = [
        ROOT / "models" / "best.pt",
        ROOT / "models" / "model.pt",
        ROOT / "models" / "last_ema.pt",
        ROOT / "weights" / "best.pt",
        ROOT / "weights" / "last_ema.pt",
        ROOT / "weights" / "model.pt",
    ]
    
    for candidate in search_candidates:
        if candidate.is_file():
            return candidate
            
    # Search for any .pt file in models/ or weights/
    for folder in [ROOT / "models", ROOT / "weights"]:
        if folder.is_dir():
            pt_files = sorted(folder.glob("*.pt"))
            if pt_files:
                return pt_files[0]
                
    raise FileNotFoundError(
        "No model checkpoint found in models/ or weights/. "
        "Please ensure model weights (e.g. models/best.pt) are present."
    )


def load_model(weights_path: Path, device: torch.device, override_preset: str = "") -> SpectraRestore:
    """Load model architecture and weights."""
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    preset = override_preset or checkpoint.get("preset", "default")
    
    if "ema" in checkpoint:
        state_dict = checkpoint["ema"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
        
    model = build_model(preset).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        
    return model


@torch.no_grad()
def restore_array(model: SpectraRestore, arr: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Perform 2x super-resolution and denoising on a single (H, W) array.
    
    Returns:
        (2*H, 2*W) float32 numpy array strictly bounded in [0.0, 1.0] with NO NaN/Inf.
    """
    orig_h, orig_w = arr.shape
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
    
    # Pad to multiple of 16 for U-Net architecture levels
    pad_h = (16 - orig_h % 16) % 16
    pad_w = (16 - orig_w % 16) % 16
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        
    use_amp = (device.type == "cuda")
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        out = model(x)
        
    target_h = orig_h * model.scale
    target_w = orig_w * model.scale
    out = out[..., :target_h, :target_w]
    
    result = out.float().squeeze().cpu().numpy()
    
    # Strictly ensure no NaN or Inf and clamp within [0.0, 1.0]
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    result = np.clip(result, 0.0, 1.0).astype(np.float32)
    
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SpectraRestore — KLA AI-Based Restoration of Degraded Images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Support both positional arguments and optional flags
    parser.add_argument(
        "input_pos",
        nargs="?",
        default=None,
        help="Input directory containing degraded .npy files (positional)",
    )
    parser.add_argument(
        "output_pos",
        nargs="?",
        default=None,
        help="Output directory to save restored .npy files (positional)",
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        default=None,
        help="Input directory containing degraded .npy files",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="Output directory to save restored .npy files",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=None,
        help="Path to trained model weights (.pt checkpoint)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="",
        help="Model preset architecture override (default, fast, tiny, large)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu)",
    )
    
    args = parser.parse_args()
    
    # Resolve input and output directory paths from positional or named args
    input_dir = args.input_pos or args.input_dir
    output_dir = args.output_pos or args.output_dir
    
    if not input_dir or not output_dir:
        parser.error(
            "Both <input-dir> and <output-dir> are required.\n"
            "Usage: python run.py <input-dir> <output-dir>"
        )
        
    args.input_dir = input_dir
    args.output_dir = output_dir
    return args


def run_restoration(input_dir_path: Path, output_dir_path: Path, weights_path: Path | None = None, device_name: str = "cuda" if torch.cuda.is_available() else "cpu", preset_override: str = "") -> None:
    """Execute complete restoration pipeline on all .npy files in input_dir."""
    device = torch.device(device_name)
    
    if not input_dir_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir_path}")
        
    # Ensure output directory exists
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .npy files (case-insensitive)
    npy_files = sorted(
        [p for p in input_dir_path.iterdir() if p.suffix.lower() == ".npy" and p.is_file() and not p.name.startswith(".")]
    )
    
    # If not found directly, check subdirectories
    if not npy_files:
        npy_files = sorted(
            [p for p in input_dir_path.rglob("*.npy") if p.is_file() and not p.name.startswith(".")]
        )
        
    if not npy_files:
        raise FileNotFoundError(
            f"No .npy files found in input directory '{input_dir_path}'. "
            f"Please ensure test files have .npy extension."
        )
        
    resolved_weights = resolve_weights(weights_path)
    model = load_model(resolved_weights, device, override_preset=preset_override)
    
    print("=" * 60)
    print("SpectraRestore — Inference Pipeline")
    print("=" * 60)
    print(f"Weights      : {resolved_weights.resolve()}")
    print(f"Device       : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Parameters   : {model.num_params() / 1e6:.2f}M")
    print(f"Scale Factor : {model.scale}x")
    print(f"Input Dir    : {input_dir_path.resolve()} ({len(npy_files)} .npy files)")
    print(f"Output Dir   : {output_dir_path.resolve()}")
    print("-" * 60)
    
    total_time = 0.0
    processed_count = 0
    
    for idx, in_file in enumerate(npy_files, 1):
        arr, in_shape = load_npy_image(in_file)
        
        t0 = time.perf_counter()
        restored = restore_array(model, arr, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        
        # Rigorous validation checks
        expected_shape = (in_shape[0] * model.scale, in_shape[1] * model.scale)
        assert restored.shape == expected_shape, (
            f"Output shape mismatch for {in_file.name}: got {restored.shape}, expected {expected_shape}"
        )
        assert not np.isnan(restored).any(), f"Output contains NaN values: {in_file.name}"
        assert not np.isinf(restored).any(), f"Output contains Inf values: {in_file.name}"
        assert restored.min() >= 0.0 and restored.max() <= 1.0, (
            f"Output values outside [0, 1] range: min={restored.min()}, max={restored.max()} in {in_file.name}"
        )
        
        # Save output with identical filename to input file
        out_file = output_dir_path / in_file.name
        np.save(out_file, restored)
        processed_count += 1
        
        if idx == 1 or idx % 20 == 0 or idx == len(npy_files):
            print(
                f"  [{idx:3d}/{len(npy_files):3d}] {in_file.name} → {out_file.name} "
                f"| in:{in_shape} → out:{restored.shape} | {elapsed * 1000.0:.1f} ms"
            )
            
    avg_latency = (total_time / max(processed_count, 1)) * 1000.0
    print("-" * 60)
    print(f"✅ Successfully restored {processed_count} files.")
    print(f"⚡ Average latency: {avg_latency:.2f} ms / image on {device}")
    print("=" * 60)


def main() -> None:
    args = parse_arguments()
    run_restoration(
        input_dir_path=Path(args.input_dir),
        output_dir_path=Path(args.output_dir),
        weights_path=Path(args.weights) if args.weights else None,
        device_name=args.device,
        preset_override=args.preset,
    )


if __name__ == "__main__":
    main()
