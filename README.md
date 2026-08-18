# SpectraRestore

## AI restoration for semiconductor inspection

**SEMICON India Hackathon 2026 

SpectraRestore restores degraded semiconductor inspection images by solving two tasks together: **denoising** and **2× super-resolution**. Given a noisy, low-resolution grayscale image, the model produces a cleaner image at twice the width and height while preserving fine repeating patterns, edges, and defect-relevant detail.

## The challenge

Inspection imagery can be degraded by sensor noise, low sampling resolution, blur, and changing intensity ranges. These effects can obscure small structures that are important during wafer and defect inspection.

The goal is to reconstruct a clean high-resolution image from a degraded low-resolution input, while keeping the output aligned with the original sample and suitable for automated evaluation.

## Our solution

SpectraRestore uses a restoration-focused U-Net built with **NAFNet blocks** and a **PixelShuffle 2× upsampling head**.

```text
Degraded image (1 × H × W)
          │
          ├─ Input-only standardisation
          ├─ NAFNet encoder → bottleneck → decoder
          ├─ PixelShuffle 2× restoration head
          └─ Bilinear 2× residual skip connection
                    │
                    ▼
Restored image (1 × 2H × 2W)
```

The residual skip retains the input’s large-scale intensity structure. The network learns the correction: it suppresses noise and reconstructs high-frequency detail rather than generating the complete image from scratch.

### Why this model

- **NAFNet backbone:** Efficient restoration blocks capture local texture and long-range structure without heavy attention layers.
- **Joint restoration:** One model learns denoising and upsampling together, avoiding loss between separate stages.
- **PixelShuffle upsampling:** Generates 2× resolution efficiently and avoids common transposed-convolution checkerboard artifacts.
- **Input-only standardisation:** Handles image-to-image intensity variation while keeping predictions in the original intensity domain.
- **Grayscale-first design:** Optimised for the single-channel inspection images used in this challenge.

### Training strategy

The model trains on paired degraded and clean images. Its composite objective combines Charbonnier loss for robust pixels, SSIM for structure, FFT magnitude loss for periodic high-frequency patterns, and LPIPS perceptual loss after warm-up when available. An exponential moving average (EMA) of model weights is evaluated and saved for stable final inference.

| Preset | Parameters | Intended use |
|---|---:|---|
| `tiny` | small | Smoke tests and Colab demo |
| `fast` | ~15M | Faster experiments |
| `default` | ~29M | Recommended challenge baseline |
| `large` | ~65M | Higher-capacity training |

## Run the project

### Google Colab

[`SpectraRestore_Colab.ipynb`](SpectraRestore_Colab.ipynb) is the recommended end-to-end workflow.

1. Select **Runtime → Change runtime type → GPU** in Colab.
2. Upload [`dist/SpectraRestore.zip`](dist/SpectraRestore.zip) to `MyDrive/SpectraRestore/`.
3. Place the KLA dataset at `MyDrive/SpectraRestore/data/`.
4. Open the notebook and choose **Run all**.

The notebook installs dependencies, restores or saves checkpoints on Drive, and creates a small synthetic demo dataset if the KLA data is not yet available.

### Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset format

```text
data/
├── train/
│   ├── degraded/  # noisy, low-resolution inputs
│   └── gt/        # clean, high-resolution targets; matching filenames
└── val/
    ├── degraded/
    └── gt/
```

The loader also accepts `lq/hq`, `input/target`, and flat `*_lq` / `*_hq` pairs. The official KLA dataset is available [here](https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6).

Paired files must have exactly the expected 2× spatial relationship. The loader stops on a mismatch to prevent training on incorrectly aligned labels; use `--resize_misaligned_gt` only for deliberately preprocessed data. PNG and TIFF integer inputs are normalized using their storage dtype (`uint8` / `uint16`), while floating-point NPY arrays retain their supplied range.

## Train

```bash
python -m src.train --config configs/default.yaml
```

CLI options override YAML values, for example: `python -m src.train --config configs/default.yaml --batch_size 4`. `weights/best.pt` contains the best validation EMA checkpoint. `weights/last_ema.pt` is the most recent shippable checkpoint.

## Quick Start & Submission Evaluation

To run the full restoration pipeline using the official submission entrypoint:

```bash
python run.py <input-dir> <output-dir>
```

Example:
```bash
python run.py data/val/degraded outputs/
```

### Technical Submission Checklist

| Requirement | Implementation & Guarantee |
|---|---|
| **Entry Script** | `python run.py <input-dir> <output-dir>` (supports positional args and flags). |
| **Input Format** | Reads all `.npy` files from `<input-dir>`. Supports `(H, W)`, `(H, W, 1)`, and multi-channel arrays. |
| **Output Directory** | Automatically created if it does not already exist. |
| **File Matching** | Generates exactly one `.npy` file per input file with identical filename. |
| **Output Format & Range** | Grayscale 2D float32 arrays with values strictly within `[0.0, 1.0]`. |
| **Data Integrity** | Zero `NaN` and zero `Inf` values guaranteed via post-processing validation. |
| **Target Resolution** | Exact 2× super-resolution: restored shape is `(2*H, 2*W)`. |
| **Model Weights** | Self-contained weights included in `models/` (auto-loaded with zero manual config). |
| **Hardware & Environment** | Runs offline on NVIDIA GPU (with clean CPU fallback) without requiring internet access, API keys, or manual downloads. |
| **Dependencies** | All packages and versions specified in `requirements.txt`. |

### Submission Folder Structure

```text
SpectraRestore/
├── run.py                 # Primary entrypoint for evaluation: python run.py <input-dir> <output-dir>
├── requirements.txt       # Python package dependencies with version constraints
├── README.md              # Technical documentation and execution guide
├── models/                # Trained model checkpoints
│   ├── best.pt            # Best model weights (auto-loaded)
│   └── last_ema.pt        # Exponential moving average weights
├── src/                   # Core architecture and processing modules
│   ├── model.py           # NAFNet-SR2x architecture definition
│   ├── image_io.py        # Image array normalisation and dtype handling
│   ├── dataset.py         # Paired dataset loading and augmentation
│   ├── losses.py          # Composite restoration loss functions
│   ├── metrics.py         # Evaluation metrics (PSNR, SSIM, LPIPS)
│   └── train.py           # Model training pipeline
├── configs/               # YAML configuration presets
└── scripts/               # Utility scripts (smoke test, preflight validator, packaging)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Validate the Environment

Run the end-to-end smoke test to verify model execution, geometry, and `run.py` CLI compliance:

```bash
python scripts/smoke_test.py
```

## Training

```bash
python -m src.train --config configs/default.yaml
```

CLI options override YAML values, for example: `python -m src.train --config configs/default.yaml --batch_size 4`. Trained weights are saved directly to `models/best.pt` and `models/last_ema.pt`.


