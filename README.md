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

## Train

```bash
python -m src.train --data_root data --preset default --batch_size 8 --iters 200000 --out_dir weights
```

`weights/best.pt` contains the best validation EMA checkpoint. `weights/last_ema.pt` is the most recent shippable checkpoint.

## KLA-compatible inference

```bash
python evaluate.py --input_dir <test_images> --output_dir outputs --weights weights/best.pt
```

The standalone evaluator accepts PNG, JPEG, TIFF, BMP, and NPY inputs. It preserves filenames and writes restored outputs to the requested folder, supporting safe image-to-image matching during evaluation.

## Validate the environment

```bash
python -m src.model
python scripts/smoke_test.py
```

