# Solution — AI-Based Restoration of Degraded Images for Semiconductor Inspection (KLA PS01)

> Working title: **"SpectraRestore: Fast Restoration of Semiconductor Inspection Images"**
> Paired supervised joint denoise + 2× super-resolution. Maps 1:1 onto the 9-slide Idea Submission Template.

---

## 1. Problem restated (Slide 2)

Semiconductor inspection tools capture microscopic grayscale images of chip structures to find defects. In production, these images arrive degraded by compounding effects:

1. **Speckle noise** — grainy noise that can push pixel values *outside* the true signal range (NoisyLR may exceed [0,1]; GT is always [0,1]).
2. **Additive Gaussian noise** — softens edges and washes out fine structure.
3. **2× downsampling** — 512×512 → 256×256 (or 256→128), destroying fine detail.
4. **Blur** (emphasized in KLA’s webinar deck alongside noise/resolution) — we treat mild blur as part of degradation augmentation for robustness.

A missed defect is a failed chip. The task: learn a single model that maps a degraded low-resolution image to a clean image at 2× resolution. Report **SSIM, pSNR, LPIPS** on your split; KLA also benchmarks **inference time on an H100**.

Key insight from KLA’s deck: degradations “may appear in any order,” so we learn the inverse mapping from pairs rather than modelling an explicit forward chain.

---

## 2. Core idea (Slide 3)

**A single image-to-image network that does joint denoising + 2× super-resolution in one forward pass, built on a NAFNet backbone with a PixelShuffle upsampling tail, trained with a metric-aligned composite loss (Charbonnier + SSIM + FFT + LPIPS) and degradation augmentation (extra noise + mild blur) for robustness.**

Positioning choices:

- **One joint model, not a pipeline.** Denoise-then-upscale propagates errors and doubles latency.
- **Regression, not generation.** No GANs / diffusion — hallucination risk on inspection imagery; single forward pass for timing.
- **Compute at low resolution.** Heavy compute at input size; upsampling only at the end (~4× cheaper than operating at 2×).

| Degradation | How we address it |
|---|---|
| Speckle (out-of-range) | No input clipping; **input-only** per-image standardization; residual learning |
| Gaussian noise | Denoising backbone + SSIM / FFT loss for edge sharpness |
| 2× resolution loss | PixelShuffle×2 tail + FFT spectrum supervision |
| Blur (webinar factor) | Mild blur in degradation augmentation |

---

## 3. Architecture (Slide 4)

### 3.1 Backbone: NAFNet (ECCV 2022)

Chosen for quality-per-latency on denoise/deblur benchmarks (SIDD/GoPro), where it matched or beat Restormer at lower cost. **This challenge also requires SR** — we adapt NAFNet with a 2× PixelShuffle tail (not a published NAFNet-SR model).

| Candidate | Why not / why yes |
|---|---|
| Diffusion | Multi-step → poor H100 timing; hallucination risk |
| Restormer / SwinIR | Strong quality, slower / heavier attention |
| Plain U-Net / RCAN | Fast but weaker than NAFNet-class blocks |
| **NAFNet + SR tail** | Strong denoise prior, plain convs, FP16-friendly |

### 3.2 Network ("NAFNet-SR2×")

```
degraded input (1×H×W, values may exceed [0,1])
        │
  per-image standardization of INPUT ONLY  (x − mean)/std
        │
  3×3 conv → shallow features (width C=32)
        │
  NAFNet U-Net at input resolution
    encoder [2, 2, 4, 8] · middle [12] · decoder [2, 2, 2, 2]
        │
  PixelShuffle(2) SR tail → 1-channel residual
        │
  + bilinear-upsample(raw input, 2×)     ← absolute-intensity skip
        │
  output in absolute intensity space     ← clamp to [0,1] only at save time
```

Important:

- **Do not** re-apply degraded mean/std to the output (that would bias toward noisy intensity stats; GT is [0,1]).
- Global skip is a *noisy* bilinear upsample; the network learns the correction (denoise + detail). It does not “guarantee” clean low frequencies by itself.
- Fully convolutional → same weights for 128→256 and 256→512.
- Grayscale in/out.

**Presets:** `default` width 32 ≈ **29.2M** (ship) · `fast` ≈ 15M · `large` width 48 ≈ 65M.

### 3.3 Inference

- FP16 + `channels_last`, single forward pass.
- Optional TTA / ONNX / TensorRT are **planned fallbacks**, not required for Round 1.
- Latency: measure on your GPU and report honestly; do not invent H100 ms numbers until measured.

---

## 4. Training strategy (Slide 4)

### 4.1 Data

- Paired degraded (256²/128²) ↔ GT (512²/256²).
- Use a held-out `val/` split (folder-based; if origins are labeled later, prefer origin-aware hold-out).
- Train crops: aligned 128² input ↔ 256² GT. Full images for validation.
- Preserve native range (no premature clip of NoisyLR).

### 4.2 Augmentation

1. **Geometric:** flips + 90° rotations only (axis-aligned layouts).
2. **Degradation augmentation (p≈0.3):** extra speckle, extra Gaussian, or mild blur on the input only — widens degradation family beyond the exact training simulation; targets robustness / OOD noise variants. True OOD may also be new *structures*; don’t overclaim that noise aug alone solves source shift.
3. No MixUp/CutMix.

### 4.3 Loss

```
L = 1.00 · Charbonnier(pred, gt)
  + 0.20 · (1 − SSIM)
  + 0.05 · FFT-L1
  + 0.10 · LPIPS   (after 20% warmup)
```

SSIM / pSNR / LPIPS are what you **report on slides**. H100 “quality scores” are not fully specified — treat LPIPS as a useful training/reporting proxy (esp. on grayscale fab imagery via RGB-repeat), not as confirmed official H100 scoring.

### 4.4 Optimization

- AdamW, LR 3e-4 → cosine to 1e-6, ~200k iters, batch sized to GPU (4–16).
- bf16 AMP, EMA 0.999 (ship EMA).
- Checkpoint by combined val score for *our* selection only (not an official KLA formula).
- Fixed seeds + logged config; freeze a real `pip freeze` before submit.

---

## 5. Innovation (Slide 5) — honest framing

1. Metric-aligned composite loss with LPIPS warmup.  
2. Degradation augmentation including mild blur (aligned with webinar factors).  
3. **Input-only** standardization for out-of-range / shift robustness without polluting GT-space outputs.  
4. FFT loss for periodic structure.  
5. Speed-aware design (LR compute, attention-free backbone, size presets) with measured latency curves.  
6. Regression-only to avoid hallucinated defects.

---

## 6. Results (Slide 6)

Fill after training: SSIM / pSNR / LPIPS (degraded baseline vs restored), visuals, error maps, **measured** latency, ablation with/without degradation aug.

---

## 7. Technology & feasibility (Slide 7)

| Item | Choice |
|---|---|
| Framework | PyTorch 2.x + `lpips` |
| Model size | ~29.2M (`default`) / ~15M (`fast`) / ~65M (`large`) |
| Training | Colab T4/L4/A100-class; batch 4–8 typical on T4 |
| Inference | FP16 single pass; report measured ms/image |
| Data | KLA Drive pairs; png/tif/npy |

---

## 8. GitHub checklist

| Item | Ship |
|---|---|
| `README.md` | clone → install → run |
| `evaluate.py` | `--input_dir` / `--output_dir`; **same output filenames as inputs** |
| `src/train.py` | full train recipe |
| `weights/` | EMA `.pt` |
| `outputs/` | restored test images |
| `requirements.txt` | freeze before submit |

PDF name: `TeamName_KLA_PS01.pdf`, max 8–9 slides.

---

## 9. References

1. Chen et al. — NAFNet, ECCV 2022.  
2. Shi et al. — PixelShuffle ESPCN, CVPR 2016.  
3. Zhang et al. — LPIPS, CVPR 2018.  
4. Zhai et al. — Real-world restoration survey, IEEE Access 2023. *(KLA deck)*  
5. Kumar et al. — Augmentation survey, IEEE Access 2024. *(KLA deck)*  
6. Terven et al. — Loss/metrics survey, 2025. *(KLA deck)*  
7. Monga et al. — Algorithm unrolling, IEEE SPM 2021. *(KLA deck; we cite context, we use direct regression)*  
8. Wang et al. — SSIM, IEEE TIP 2004.

---

## 10. Slide mapping

| Slide | Source |
|---|---|
| 1 Team | your details |
| 2 Problem | §1 |
| 3 Idea | §2 |
| 4 Solution | §3–4 |
| 5 Innovation | §5 |
| 6 Results | §6 after training |
| 7 Feasibility | §7 |
| 8 GitHub / video | §8 |
| 9 References | §9 |

## 11. Execution order

1. Confirm real dataset layout / ranges / pairing.  
2. Keep `evaluate.py` contract frozen (same filenames).  
3. Train `default` (width 32); produce Slide 6 assets.  
4. Ablation: with/without degradation aug.  
5. Measure latency; consider `fast` only if needed.  
6. Fresh-machine test; freeze `pip freeze`; submit by **16 Aug 2026**.
