# SpectraRestore — KLA PS01

**AI-Based Restoration of Degraded Images for Semiconductor Inspection**  
SEMICON India Hackathon 2026 · Track 1 (KLA)

Joint denoise + 2× super-resolution with a **NAFNet** backbone and PixelShuffle tail.  
See [`SOLUTION.md`](SOLUTION.md) for the full design rationale.

---

## Run on Google Colab (recommended)

1. Open [`SpectraRestore_Colab.ipynb`](SpectraRestore_Colab.ipynb) in Colab  
   (*File → Upload notebook*, or push this repo and use “Open in Colab”).
2. **Runtime → Change runtime type → GPU (T4)**.
3. Upload `dist/SpectraRestore.zip` to Drive at  
   `MyDrive/SpectraRestore/SpectraRestore.zip`  
   (or copy the whole project folder there).
4. Put the KLA dataset at `MyDrive/SpectraRestore/data/` with `train/` + `val/` pairs.
5. Run the notebook top → bottom. Weights auto-save back to Drive.

Local zip (regenerate anytime):

```bash
zip -r dist/SpectraRestore.zip src evaluate.py requirements.txt README.md SOLUTION.md configs scripts SpectraRestore_Colab.ipynb .gitignore -x '*/__pycache__/*'
```

## Setup (local)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset layout

```
data/
  train/
    degraded/   # noisy + low-res inputs
    gt/         # clean full-res targets (same filenames)
  val/
    degraded/
    gt/
```

Also accepts `lq/hq`, `input/target`, or flat `*_lq.png` / `*_hq.png` pairs.

Download the official KLA dataset:  
https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6

## Train

```bash
python -m src.train \
  --data_root data \
  --preset default \
  --batch_size 8 \
  --iters 200000 \
  --out_dir weights
```

Presets: `default` (~29M), `large` (~65M), `fast` (~15M), `tiny` (smoke tests).

Checkpoints land in `weights/`:
- `best.pt` — best EMA by combined val score (SSIM + pSNR/50 + 1−LPIPS)
- `last_ema.pt` — latest EMA (what we ship)
- `ckpt_XXXXXX.pt` — periodic full resumes

## Evaluate / inference (KLA script)

```bash
python evaluate.py --input_dir <test_images> --output_dir outputs
```

Optional:

```bash
python evaluate.py \
  --input_dir test_degraded \
  --output_dir outputs \
  --weights weights/best.pt \
  --preset default
```

Writes restored images with the **same filenames** as the inputs (safe for KLA matching).  
This is the **critical deliverable** — KLA runs it AS-IS on an H100.  
It must work on a fresh machine with only `pip install -r requirements.txt` + weights.

## Smoke test (no dataset needed)

```bash
python -m src.model          # forward shapes + param counts
python scripts/smoke_test.py # tiny train step + evaluate round-trip
```

## Repo checklist (hackathon)

| # | Item | Status |
|---|---|---|
| 1 | `README.md` | this file |
| 2 | `evaluate.py` standalone | ✓ |
| 3 | Training script | `src/train.py` |
| 4 | Trained weights | `weights/best.pt` (after training) |
| 5 | Restored test outputs | `outputs/` |
| 6 | `requirements.txt` | ✓ |

## Citation / design

Architecture and training recipe are documented in `SOLUTION.md`.  
Idea deck: `slides/TeamName_KLA_PS01.pptx`.
