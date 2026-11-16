# Cross-Domain Generalization of SAM2 on LoveDA

**Research question:** Does SAM2's broad pretraining give it better urban↔rural
cross-domain generalization than a traditional segmentation model (DeepLabV3)
trained from scratch on the same data?

## Dataset

**LoveDA** — 5,987 high-resolution (1024×1024 px, 0.3 m GSD) aerial images from
three Chinese cities, with 7-class pixel-level land-cover masks:

| Label ID | Class       | Colour (vis) |
|:--------:|-------------|:------------:|
| 0        | no-data ⚠️  | Black        |
| 1        | background  | Grey         |
| 2        | building    | Red          |
| 3        | road        | Yellow       |
| 4        | water       | Blue         |
| 5        | barren      | Brown        |
| 6        | forest      | Green        |
| 7        | agriculture | Lime         |

> **Label 0 (no-data) is always IGNORED** in loss and metric calculations.

Two domains: **Urban** (denser buildings, roads) and **Rural** (more forest, agriculture).

## Prerequisites

```bash
# Python 3.10+, CUDA 12.x recommended
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torchgeo matplotlib tqdm Pillow numpy opencv-python
pip install segment-anything-2   # Meta's official SAM2 package (scripts 02-05)
pip install huggingface_hub      # for SAM2 from_pretrained() auto-download
```

## Script Pipeline

| #  | Script | Purpose | Est. Time |
|----|--------|---------|-----------|
| 01 | `01_setup_and_explore.py` | Download LoveDA, verify data, print stats, visualise samples | ~10 min (first run) |
| 02 | `02_zero_shot_baseline.py` | Zero-shot SAM2 eval on a small LoveDA subset | ~15 min |
| 03 | `03_prompt_generation.py` | Auto-generate point/box prompts from GT masks, visualise | ~2 min |
| 04 | `04_train_sam2_urban.py` | Fine-tune SAM2 (frozen encoder) on Urban train, eval both | ~2-3 hrs |
| 05 | `05_train_sam2_rural.py` | Fine-tune SAM2 (frozen encoder) on Rural train, eval both | ~2-3 hrs |
| 06 | `06_train_baseline_urban.py` | Train DeepLabV3 from scratch on Urban, eval both | ~1-2 hrs |
| 07 | `07_train_baseline_rural.py` | Train DeepLabV3 from scratch on Rural, eval both | ~1-2 hrs |
| 08 | `08_compile_results.py` | Build full results table, compute domain gap, visualise | ~2 min |

**Run order:** Scripts must be run sequentially (01 → 02 → … → 08).

## Quick Smoke Test

Every training script has a `SUBSET_SIZE` variable at the top. Set it to a small
number (e.g. `50`) to verify everything works before committing GPU time to the
full-scale run.

## Hardware

- **Target:** Single NVIDIA RTX 5090 (32 GB VRAM)
- All scripts use a single GPU (`cuda:0`).
- Batch sizes and image resolutions are adjustable variables at the top of each
  script. Reduce them if you hit OOM errors.

## Outputs

```
outputs/
├── 01_exploration/      # stats JSON, sample grids, class distribution charts
├── 02_zero_shot/        # zero-shot prediction images + IoU report
├── 03_prompts/          # prompt visualisation images
├── 04_sam2_urban/       # checkpoints, loss curves, eval results
├── 05_sam2_rural/       # checkpoints, loss curves, eval results
├── 06_baseline_urban/   # checkpoints, loss curves, eval results
├── 07_baseline_rural/   # checkpoints, loss curves, eval results
└── 08_results/          # final tables (CSV + PNG), comparison visuals
```

## Honest Limitations & Design Notes

1. **DeepLabV3 backbone:** `torchvision.models.segmentation.deeplabv3_resnet50`
   uses an **ImageNet-pretrained** ResNet-50 backbone. This means the baseline
   is NOT truly "from scratch" — it has natural-image pretraining (though not
   remote-sensing pretraining). This is stated in code comments and must be
   reported honestly in any write-up.
2. **SAM2 prompt dependency:** SAM2 is a prompted model; its performance depends
   heavily on prompt quality. Our auto-generated prompts (script 03) are derived
   from ground-truth masks, which is not realistic at inference time on unseen
   data. This is a known limitation.
3. **Epoch parity:** Both SAM2 and the baseline are trained for the same number
   of epochs with comparable learning rate schedules to ensure fair comparison.
4. **No multi-GPU:** All training is single-GPU. Scale as needed.
