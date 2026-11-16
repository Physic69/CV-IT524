#!/usr/bin/env python3
"""
01_setup_and_explore.py
=======================
LoveDA Dataset — Setup, Download, Integrity Check & Exploration

What this script does:
  1. Installs / verifies required packages.
  2. Downloads the LoveDA dataset via torchgeo (Train + Val splits,
     Urban and Rural scenes) from Zenodo.
  3. Walks the raw directory tree and prints file counts.
  4. Loads a handful of images + masks and runs integrity checks
     (shape, dtype, value range).
  5. Prints per-domain, per-split image counts.
  6. Computes per-class pixel distribution (urban vs rural, train vs val).
  7. Saves sample visualizations (image + colorised mask side-by-side)
     to an output directory.

Run:
    python 01_setup_and_explore.py

Expected wall-time: ~5-10 min for the initial Zenodo download (~1.5 GB),
then < 2 min for all exploration steps.  Subsequent runs skip the download.

Outputs saved to:  ./outputs/01_exploration/
"""

import os
import sys
import time
import glob
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# User-adjustable variables  (edit these if needed)
# ---------------------------------------------------------------------------
DATA_ROOT      = "./data"              # where the dataset will be stored
OUTPUT_DIR     = "./outputs/01_exploration"
NUM_VIS        = 6                     # how many sample images to visualise per scene
SUBSET_SIZE    = None                  # set to e.g. 50 for a quick smoke-test; None = full

# ---------------------------------------------------------------------------
# 0.  Verify / install dependencies
# ---------------------------------------------------------------------------
def check_deps():
    """Check that required packages are available."""
    required = {
        "torch":      "torch",
        "torchgeo":   "torchgeo",
        "PIL":        "Pillow",
        "matplotlib": "matplotlib",
        "numpy":      "numpy",
        "tqdm":       "tqdm",
    }
    missing = []
    for import_name, pip_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"[!] Missing packages: {missing}")
        print(f"    Install with:  pip install {' '.join(missing)}")
        sys.exit(1)
    print("[✓] All required packages are available.")

check_deps()

# Now safe to import everything
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")                  # non-interactive backend (safe on headless servers)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1.  LoveDA class labels & colour map
# ---------------------------------------------------------------------------
# Mask pixel values in the PNG files:
#   0 = no-data / ignore
#   1 = background          (class index 0 when we train — see note below)
#   2 = building             ...
#   3 = road
#   4 = water
#   5 = barren
#   6 = forest
#   7 = agriculture
#
# IMPORTANT:  Label 0 ("no-data") must be IGNORED during loss computation
# and IoU calculation.  All our training scripts will use ignore_index=0.

CLASS_NAMES = [
    "no-data",       # 0  — IGNORE
    "background",    # 1
    "building",      # 2
    "road",          # 3
    "water",         # 4
    "barren",        # 5
    "forest",        # 6
    "agriculture",   # 7
]
NUM_CLASSES_WITH_NODATA = len(CLASS_NAMES)          # 8
NUM_CLASSES             = NUM_CLASSES_WITH_NODATA - 1  # 7 (trainable)

# Distinct colours for each class (for visualisation only)
CLASS_COLOURS = np.array([
    [0,   0,   0],    # 0  no-data       → black
    [128, 128, 128],  # 1  background    → grey
    [255,   0,   0],  # 2  building      → red
    [255, 255,   0],  # 3  road          → yellow
    [0,   0, 255],    # 4  water         → blue
    [139,  69,  19],  # 5  barren        → brown
    [0,  128,   0],   # 6  forest        → green
    [0,  255,   0],   # 7  agriculture   → lime
], dtype=np.uint8)


def colorize_mask(mask_np: np.ndarray) -> np.ndarray:
    """Convert a single-channel label mask (H,W) → RGB (H,W,3)."""
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id in range(NUM_CLASSES_WITH_NODATA):
        rgb[mask_np == label_id] = CLASS_COLOURS[label_id]
    return rgb


# ---------------------------------------------------------------------------
# 2.  Download via torchgeo
# ---------------------------------------------------------------------------
def download_loveda(root: str):
    """
    Use torchgeo to download Train and Val splits (Urban + Rural).
    Test split has no masks so we skip it for now.
    """
    from torchgeo.datasets import LoveDA

    for split in ["train", "val"]:
        for scene in ["urban", "rural"]:
            print(f"  Downloading / verifying: split={split}, scene={scene} ...")
            try:
                _ = LoveDA(
                    root=root,
                    split=split,
                    scene=[scene],
                    download=True,
                    checksum=True,
                )
                print(f"    [✓] {split}/{scene} ready.")
            except Exception as e:
                print(f"    [!] Error for {split}/{scene}: {e}")
                raise


# ---------------------------------------------------------------------------
# 3.  Raw directory walk — file counts
# ---------------------------------------------------------------------------
def walk_directory(root: str):
    """Print the directory tree with image and mask counts."""
    print("\n" + "=" * 60)
    print("RAW DIRECTORY STRUCTURE")
    print("=" * 60)
    for split_name in ["Train", "Val"]:
        split_dir = os.path.join(root, split_name)
        if not os.path.isdir(split_dir):
            print(f"  [!] Missing directory: {split_dir}")
            continue
        for scene_name in ["Urban", "Rural"]:
            scene_dir = os.path.join(split_dir, scene_name)
            if not os.path.isdir(scene_dir):
                print(f"  [!] Missing directory: {scene_dir}")
                continue
            img_dir  = os.path.join(scene_dir, "images_png")
            mask_dir = os.path.join(scene_dir, "masks_png")
            n_imgs  = len(glob.glob(os.path.join(img_dir,  "*.png"))) if os.path.isdir(img_dir)  else 0
            n_masks = len(glob.glob(os.path.join(mask_dir, "*.png"))) if os.path.isdir(mask_dir) else 0
            match = "✓" if (n_imgs == n_masks and n_imgs > 0) else "✗"
            print(f"  {split_name}/{scene_name}:  {n_imgs} images, {n_masks} masks  [{match}]")


# ---------------------------------------------------------------------------
# 4.  Load images and masks manually (no torchgeo dependency beyond download)
# ---------------------------------------------------------------------------
def get_file_pairs(root: str, split: str, scene: str, subset_size=None):
    """
    Return list of (image_path, mask_path) tuples.
    split: 'Train' or 'Val'  (capitalised to match on-disk names)
    scene: 'Urban' or 'Rural'
    """
    img_dir  = os.path.join(root, split, scene, "images_png")
    mask_dir = os.path.join(root, split, scene, "masks_png")
    img_paths  = sorted(glob.glob(os.path.join(img_dir,  "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    assert len(img_paths) == len(mask_paths), (
        f"Image/mask count mismatch in {split}/{scene}: "
        f"{len(img_paths)} images vs {len(mask_paths)} masks"
    )
    pairs = list(zip(img_paths, mask_paths))
    if subset_size is not None:
        pairs = pairs[:subset_size]
    return pairs


def load_image_mask(img_path, mask_path):
    """Load an image and its mask as numpy arrays."""
    img  = np.array(Image.open(img_path).convert("RGB"))    # (H, W, 3) uint8
    mask = np.array(Image.open(mask_path).convert("L"))       # (H, W)    uint8
    return img, mask


# ---------------------------------------------------------------------------
# 5.  Integrity checks
# ---------------------------------------------------------------------------
def integrity_checks(root: str):
    """Spot-check a few image/mask pairs for shape, dtype, value range."""
    print("\n" + "=" * 60)
    print("INTEGRITY CHECKS  (first 3 samples from each scene)")
    print("=" * 60)
    all_ok = True
    for split in ["Train", "Val"]:
        for scene in ["Urban", "Rural"]:
            pairs = get_file_pairs(root, split, scene)
            for img_path, mask_path in pairs[:3]:
                img, mask = load_image_mask(img_path, mask_path)
                issues = []
                if img.shape != (1024, 1024, 3):
                    issues.append(f"img shape {img.shape} ≠ (1024,1024,3)")
                if mask.shape != (1024, 1024):
                    issues.append(f"mask shape {mask.shape} ≠ (1024,1024)")
                if img.dtype != np.uint8:
                    issues.append(f"img dtype {img.dtype}")
                unique_vals = np.unique(mask)
                if unique_vals.max() > 7:
                    issues.append(f"mask max={unique_vals.max()} > 7")
                status = "✓" if not issues else "✗ " + "; ".join(issues)
                if issues:
                    all_ok = False
                fname = os.path.basename(img_path)
                print(f"  {split}/{scene}/{fname}  {status}")
    if all_ok:
        print("  => All spot-checks passed.")
    else:
        print("  => Some checks FAILED — inspect the issues above.")


# ---------------------------------------------------------------------------
# 6.  Dataset statistics
# ---------------------------------------------------------------------------
def compute_statistics(root: str, subset_size=None):
    """
    Compute and print:
      - Per-domain, per-split image counts
      - Per-class pixel counts (absolute and %)
      - Per-domain class distribution comparison
    Returns a dict of stats for optional downstream use.
    """
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    stats = {}   # stats[split][scene] = {count, class_pixels}

    for split in ["Train", "Val"]:
        stats[split] = {}
        for scene in ["Urban", "Rural"]:
            pairs = get_file_pairs(root, split, scene, subset_size=subset_size)
            n_images = len(pairs)
            class_pixels = np.zeros(NUM_CLASSES_WITH_NODATA, dtype=np.int64)

            print(f"\n  Scanning {split}/{scene}  ({n_images} images) ...")
            for img_path, mask_path in tqdm(pairs, desc=f"    {split}/{scene}",
                                            ncols=80, leave=False):
                _, mask = load_image_mask(img_path, mask_path)
                for c in range(NUM_CLASSES_WITH_NODATA):
                    class_pixels[c] += int((mask == c).sum())

            stats[split][scene] = {
                "count": n_images,
                "class_pixels": class_pixels.tolist(),
            }

    # ---- Print image counts ----
    print("\n  Image counts:")
    total = 0
    for split in ["Train", "Val"]:
        for scene in ["Urban", "Rural"]:
            n = stats[split][scene]["count"]
            total += n
            print(f"    {split:5s} / {scene:5s} : {n:5d}")
    print(f"    {'TOTAL':>13s} : {total:5d}")

    # ---- Print class distribution ----
    print("\n  Per-class pixel distribution (%):")
    header = f"  {'Class':>14s}"
    for split in ["Train", "Val"]:
        for scene in ["Urban", "Rural"]:
            header += f"  {split[:2]}/{scene[:2]:>4s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for c in range(NUM_CLASSES_WITH_NODATA):
        row = f"  {CLASS_NAMES[c]:>14s}"
        for split in ["Train", "Val"]:
            for scene in ["Urban", "Rural"]:
                pix = np.array(stats[split][scene]["class_pixels"])
                total_pix = pix.sum()
                pct = 100.0 * pix[c] / total_pix if total_pix > 0 else 0.0
                row += f"  {pct:7.2f}%"
        print(row)

    # ---- Save stats to JSON for downstream scripts ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stats_path = os.path.join(OUTPUT_DIR, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  [✓] Stats saved to {stats_path}")

    return stats


# ---------------------------------------------------------------------------
# 7.  Visualisation
# ---------------------------------------------------------------------------
def save_sample_visualisations(root: str, num_vis=NUM_VIS):
    """
    Save side-by-side (image | colorised mask) grids for each domain.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print("SAMPLE VISUALISATIONS")
    print("=" * 60)

    for split in ["Train", "Val"]:
        for scene in ["Urban", "Rural"]:
            pairs = get_file_pairs(root, split, scene)
            # Take evenly spaced samples
            indices = np.linspace(0, len(pairs) - 1, min(num_vis, len(pairs)),
                                  dtype=int)
            selected = [pairs[i] for i in indices]

            n = len(selected)
            fig, axes = plt.subplots(n, 3, figsize=(18, 6 * n))
            if n == 1:
                axes = axes[np.newaxis, :]

            fig.suptitle(f"LoveDA — {split} / {scene}  ({n} samples)",
                         fontsize=16, fontweight="bold", y=1.01)

            for row, (img_path, mask_path) in enumerate(selected):
                img, mask = load_image_mask(img_path, mask_path)
                cmask = colorize_mask(mask)

                # Column 0: original image
                axes[row, 0].imshow(img)
                axes[row, 0].set_title(os.path.basename(img_path), fontsize=9)
                axes[row, 0].axis("off")

                # Column 1: colorised mask
                axes[row, 1].imshow(cmask)
                axes[row, 1].set_title("Ground Truth Mask", fontsize=9)
                axes[row, 1].axis("off")

                # Column 2: image with mask overlay (alpha blend)
                overlay = (0.6 * img.astype(np.float32)
                           + 0.4 * cmask.astype(np.float32)).clip(0, 255).astype(np.uint8)
                axes[row, 2].imshow(overlay)
                axes[row, 2].set_title("Overlay", fontsize=9)
                axes[row, 2].axis("off")

            # Add legend
            legend_patches = [
                mpatches.Patch(color=CLASS_COLOURS[c] / 255.0, label=CLASS_NAMES[c])
                for c in range(NUM_CLASSES_WITH_NODATA)
            ]
            fig.legend(handles=legend_patches, loc="lower center",
                       ncol=NUM_CLASSES_WITH_NODATA, fontsize=9,
                       bbox_to_anchor=(0.5, -0.02))

            plt.tight_layout()
            fname = f"samples_{split.lower()}_{scene.lower()}.png"
            fpath = os.path.join(OUTPUT_DIR, fname)
            fig.savefig(fpath, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  [✓] Saved {fpath}")

    print(f"\n  All visualisations saved to {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# 8.  Class-distribution bar chart
# ---------------------------------------------------------------------------
def save_class_distribution_chart(root: str, subset_size=None):
    """Bar chart comparing urban vs rural class distributions (Train split)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load stats if already computed, otherwise compute
    stats_path = os.path.join(OUTPUT_DIR, "dataset_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        stats = compute_statistics(root, subset_size=subset_size)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("LoveDA Class Distribution — Urban vs Rural",
                 fontsize=14, fontweight="bold")

    for ax_idx, split in enumerate(["Train", "Val"]):
        ax = axes[ax_idx]
        urban_pix = np.array(stats[split]["Urban"]["class_pixels"], dtype=np.float64)
        rural_pix = np.array(stats[split]["Rural"]["class_pixels"], dtype=np.float64)

        # Normalise to percentages (skip class 0 = no-data for cleaner chart)
        classes_to_plot = list(range(1, NUM_CLASSES_WITH_NODATA))
        urban_pct = 100.0 * urban_pix[classes_to_plot] / urban_pix[classes_to_plot].sum()
        rural_pct = 100.0 * rural_pix[classes_to_plot] / rural_pix[classes_to_plot].sum()
        labels = [CLASS_NAMES[c] for c in classes_to_plot]

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width / 2, urban_pct, width, label="Urban",
                        color="#3b82f6", edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, rural_pct, width, label="Rural",
                        color="#22c55e", edgecolor="white", linewidth=0.5)

        ax.set_ylabel("Pixel Proportion (%)")
        ax.set_title(f"{split} Split")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "class_distribution_chart.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Class distribution chart saved to {chart_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("  LoveDA — Setup & Data Exploration")
    print("  Script 01 of 08")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Download
    print("\n[Step 1] Downloading LoveDA dataset ...")
    download_loveda(DATA_ROOT)

    # Step 2: Walk directories
    walk_directory(DATA_ROOT)

    # Step 3: Integrity checks
    integrity_checks(DATA_ROOT)

    # Step 4: Statistics
    stats = compute_statistics(DATA_ROOT, subset_size=SUBSET_SIZE)

    # Step 5: Visualisations
    save_sample_visualisations(DATA_ROOT, num_vis=NUM_VIS)

    # Step 6: Class distribution chart
    save_class_distribution_chart(DATA_ROOT, subset_size=SUBSET_SIZE)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  DONE — total time: {elapsed:.1f}s")
    print(f"  Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)
    print("\n  Next step: run  python 02_zero_shot_baseline.py")


if __name__ == "__main__":
    main()
