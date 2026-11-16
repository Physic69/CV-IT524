#!/usr/bin/env python3
"""
02_zero_shot_baseline.py
========================
Zero-shot SAM2 evaluation on LoveDA.

WHAT THIS SCRIPT DOES:
  1. Loads SAM2.1 (Hiera-Large) from HuggingFace via from_pretrained().
  2. Runs two zero-shot evaluation strategies on a subset of LoveDA images:

     Strategy A — "Automatic Mask Generator" (truly zero-shot):
       Uses SAM2AutomaticMaskGenerator to produce class-agnostic masks for the
       whole image.  Because SAM2 does NOT produce class labels, we use a
       simple majority-vote heuristic: for each generated mask, look at the
       ground-truth pixels under that mask and assign the most-frequent GT
       label.  This is a GENEROUS upper-bound on zero-shot performance —
       it tells us how well SAM2's mask SHAPES align with the GT even when
       the model is given perfect class assignments.

     Strategy B — "Oracle-prompted per-class" (not truly zero-shot):
       For each of the 7 classes in the GT mask, generate a foreground point
       prompt (centroid of the largest connected component) and a tight
       bounding-box prompt, then ask SAM2 to predict the mask for that class.
       This uses GT information to create prompts, so it is NOT a fair zero-
       shot baseline — it is a diagnostic tool showing how well SAM2 can
       delineate each class's region when told exactly where to look.

  3. Computes per-class IoU and mean IoU for each strategy on each domain.
  4. Saves example prediction visualizations and a summary JSON.

CRITICAL HONESTY NOTE:
  Both strategies use ground-truth information (Strategy A for class assignment,
  Strategy B for prompt generation).  Neither is a true zero-shot semantic
  segmentation baseline.  SAM2 is fundamentally a CLASS-AGNOSTIC prompted
  segmentor — it cannot do semantic segmentation without external class info.
  These results establish a CEILING on what zero-shot SAM2 can achieve, and
  they let us visually inspect mask quality before fine-tuning.

Run:
    python 02_zero_shot_baseline.py

Prerequisites:
    pip install sam-2           # Meta's official SAM2 package
    pip install huggingface_hub # for from_pretrained()
    (or: git clone https://github.com/facebookresearch/sam2 && pip install -e .)

Expected runtime: ~20-40 min on RTX 5090 for SUBSET_SIZE=20 per domain.
  (SAM2-Large is ~600M params; set_image() is the bottleneck at ~0.5s/image.)

Outputs saved to:  ./outputs/02_zero_shot/
"""

import os
import sys
import time
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# User-adjustable variables
# ---------------------------------------------------------------------------
DATA_ROOT       = "./data"
OUTPUT_DIR      = "./outputs/02_zero_shot"
SUBSET_SIZE     = 20          # images per domain to evaluate (set to None for full)
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# SAM2 model — using HuggingFace from_pretrained (auto-download)
SAM2_MODEL_ID   = "facebook/sam2.1-hiera-large"

# Automatic mask generator settings
AMG_POINTS_PER_SIDE   = 32    # grid density (32x32 = 1024 points)
AMG_PRED_IOU_THRESH   = 0.7   # lower than default 0.8 to get more masks
AMG_STABILITY_THRESH  = 0.90
AMG_POINTS_PER_BATCH  = 64    # GPU memory control

# Number of oracle-prompted visualisations to save
NUM_VIS         = 5

# ---------------------------------------------------------------------------
# LoveDA constants (must match 01_setup_and_explore.py)
# ---------------------------------------------------------------------------
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
NUM_CLASSES_WITH_NODATA = 8
NUM_CLASSES = 7             # classes 1-7
IGNORE_INDEX = 0

CLASS_COLOURS = np.array([
    [0,   0,   0],    # 0  no-data
    [128, 128, 128],  # 1  background
    [255,   0,   0],  # 2  building
    [255, 255,   0],  # 3  road
    [0,   0, 255],    # 4  water
    [139,  69,  19],  # 5  barren
    [0,  128,   0],   # 6  forest
    [0,  255,   0],   # 7  agriculture
], dtype=np.uint8)


def colorize_mask(mask_np):
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES_WITH_NODATA):
        rgb[mask_np == c] = CLASS_COLOURS[c]
    return rgb


# ---------------------------------------------------------------------------
# IoU computation  (with sanity test)
# ---------------------------------------------------------------------------
def compute_iou(pred: np.ndarray, gt: np.ndarray,
                num_classes: int = NUM_CLASSES,
                ignore_index: int = IGNORE_INDEX):
    """
    Compute per-class IoU and mean IoU.

    Args:
        pred:  (H, W) predicted label map, values in [0, num_classes_with_nodata-1]
        gt:    (H, W) ground truth label map
        num_classes: number of VALID classes (excluding ignore)
        ignore_index: label to ignore (0 = no-data)

    Returns:
        per_class_iou: dict  {class_id: iou}   (class_id = 1..7)
        mean_iou: float
    """
    valid_mask = (gt != ignore_index)
    per_class_iou = {}
    ious_for_mean = []

    for c in range(1, num_classes + 1):   # classes 1..7
        pred_c = (pred == c) & valid_mask
        gt_c   = (gt == c) & valid_mask
        intersection = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        if union == 0:
            # Class not present in either pred or gt — skip for mIoU
            per_class_iou[c] = float('nan')
        else:
            iou = float(intersection) / float(union)
            per_class_iou[c] = iou
            ious_for_mean.append(iou)

    mean_iou = float(np.nanmean(ious_for_mean)) if ious_for_mean else 0.0
    return per_class_iou, mean_iou


def sanity_test_iou():
    """Quick unit test for IoU computation."""
    print("\n[Sanity Test] IoU computation ...")
    # Perfect prediction
    gt   = np.array([[0, 1, 1, 2], [2, 3, 3, 0]], dtype=np.uint8)
    pred = np.array([[0, 1, 1, 2], [2, 3, 3, 0]], dtype=np.uint8)
    per_cls, miou = compute_iou(pred, gt)
    assert abs(miou - 1.0) < 1e-6, f"Perfect pred should give mIoU=1.0, got {miou}"

    # Completely wrong prediction (no overlap for any class)
    pred2 = np.array([[0, 2, 2, 3], [3, 1, 1, 0]], dtype=np.uint8)
    per_cls2, miou2 = compute_iou(pred2, gt)
    assert abs(miou2 - 0.0) < 1e-6, f"Zero overlap should give mIoU=0.0, got {miou2}"

    # Partial overlap: class 1 half correct
    gt3   = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    pred3 = np.array([[1, 1, 2, 2]], dtype=np.uint8)
    per_cls3, miou3 = compute_iou(pred3, gt3)
    # Class 1: intersection=2, union=4 → IoU=0.5
    # Class 2: intersection=0, union=2 → IoU=0.0
    assert abs(per_cls3[1] - 0.5) < 1e-6, f"Class 1 IoU should be 0.5, got {per_cls3[1]}"
    assert abs(per_cls3[2] - 0.0) < 1e-6, f"Class 2 IoU should be 0.0, got {per_cls3[2]}"

    print("  [✓] All IoU sanity checks passed.")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def get_file_pairs(root, split, scene, subset_size=None):
    img_dir  = os.path.join(root, split, scene, "images_png")
    mask_dir = os.path.join(root, split, scene, "masks_png")
    img_paths  = sorted(glob.glob(os.path.join(img_dir,  "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    assert len(img_paths) == len(mask_paths), (
        f"Mismatch in {split}/{scene}: {len(img_paths)} imgs vs {len(mask_paths)} masks"
    )
    pairs = list(zip(img_paths, mask_paths))
    if subset_size is not None:
        pairs = pairs[:subset_size]
    return pairs


def load_image_mask(img_path, mask_path):
    from PIL import Image
    img  = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    return img, mask


# ---------------------------------------------------------------------------
# Strategy A:  Automatic Mask Generator → majority-vote class assignment
# ---------------------------------------------------------------------------
def evaluate_amg_zero_shot(mask_generator, pairs, scene_name):
    """
    Run SAM2 Automatic Mask Generator on each image.
    Assign each generated mask the majority GT class under it.
    Assemble into a semantic prediction and compute IoU.
    """
    from tqdm import tqdm

    all_per_class_iou = []
    all_miou = []

    print(f"\n  [AMG] Evaluating {len(pairs)} {scene_name} images ...")
    for img_path, mask_path in tqdm(pairs, desc=f"    AMG/{scene_name}", ncols=80):
        img, gt_mask = load_image_mask(img_path, mask_path)

        # Generate masks (class-agnostic)
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            anns = mask_generator.generate(img)

        # Build semantic prediction by assigning each mask its majority GT class
        # Process masks in descending area order so smaller (more specific)
        # masks overwrite larger (background-like) ones
        h, w = gt_mask.shape
        pred_semantic = np.zeros((h, w), dtype=np.uint8)  # default = 0 (no-data)

        sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
        for ann in sorted_anns:
            seg = ann['segmentation']   # bool array (H, W)
            if seg.sum() == 0:
                continue
            # Find majority GT class under this mask (ignoring no-data=0)
            gt_under_mask = gt_mask[seg]
            gt_under_mask = gt_under_mask[gt_under_mask != IGNORE_INDEX]
            if len(gt_under_mask) == 0:
                continue
            majority_class = int(np.bincount(gt_under_mask).argmax())
            pred_semantic[seg] = majority_class

        per_cls, miou = compute_iou(pred_semantic, gt_mask)
        all_per_class_iou.append(per_cls)
        all_miou.append(miou)

    # Average per-class IoU across all images
    avg_per_class = {}
    for c in range(1, NUM_CLASSES + 1):
        vals = [d[c] for d in all_per_class_iou if not np.isnan(d.get(c, float('nan')))]
        avg_per_class[c] = float(np.mean(vals)) if vals else float('nan')
    avg_miou = float(np.mean(all_miou)) if all_miou else 0.0

    return avg_per_class, avg_miou


# ---------------------------------------------------------------------------
# Strategy B:  Oracle-prompted per-class masks
# ---------------------------------------------------------------------------
def generate_class_prompt(mask_np, class_id):
    """
    Given a GT mask and a class ID, generate a point prompt (centroid of
    largest connected component) and a bounding-box prompt for that class.

    Returns:
        point_coord: (x, y) or None if class not present
        bbox: [x1, y1, x2, y2] or None
    """
    import cv2
    binary = (mask_np == class_id).astype(np.uint8)
    if binary.sum() == 0:
        return None, None

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return None, None

    # Largest component (skip label 0 = background of binary image)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1  # +1 to account for skipped bg

    # Centroid of largest component
    cx, cy = centroids[largest_idx]
    point_coord = np.array([[cx, cy]], dtype=np.float32)  # (1, 2)

    # Bounding box of largest component
    x = stats[largest_idx, cv2.CC_STAT_LEFT]
    y = stats[largest_idx, cv2.CC_STAT_TOP]
    w = stats[largest_idx, cv2.CC_STAT_WIDTH]
    h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
    bbox = np.array([x, y, x + w, y + h], dtype=np.float32)

    return point_coord, bbox


def evaluate_oracle_prompted(predictor, pairs, scene_name):
    """
    For each image, for each class present in the GT:
      - Generate a point + box prompt from GT
      - Ask SAM2 to predict a binary mask
      - Assign that class label to predicted foreground pixels
    Assemble into a semantic prediction and compute IoU.
    """
    from tqdm import tqdm

    all_per_class_iou = []
    all_miou = []

    print(f"\n  [Oracle] Evaluating {len(pairs)} {scene_name} images ...")
    for img_path, mask_path in tqdm(pairs, desc=f"    Oracle/{scene_name}", ncols=80):
        img, gt_mask = load_image_mask(img_path, mask_path)
        h, w = gt_mask.shape

        # Set image once (expensive encoder pass)
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            predictor.set_image(img)

        pred_semantic = np.zeros((h, w), dtype=np.uint8)
        pred_scores   = np.full((h, w), -1.0, dtype=np.float32)  # for conflict resolution

        for class_id in range(1, NUM_CLASSES_WITH_NODATA):
            point_coord, bbox = generate_class_prompt(gt_mask, class_id)
            if point_coord is None:
                continue

            point_label = np.array([1], dtype=np.int32)  # foreground

            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                masks, scores, _ = predictor.predict(
                    point_coords=point_coord,
                    point_labels=point_label,
                    box=bbox,
                    multimask_output=True,
                )

            # Pick the mask with highest predicted IoU score
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]        # bool (H, W)
            best_score = scores[best_idx]

            # Assign class where this mask wins over previously assigned
            update_region = np.logical_and(best_mask, (best_score > pred_scores))
            pred_semantic[update_region] = class_id
            pred_scores[update_region] = best_score

        predictor.reset_predictor()

        per_cls, miou = compute_iou(pred_semantic, gt_mask)
        all_per_class_iou.append(per_cls)
        all_miou.append(miou)

    # Average
    avg_per_class = {}
    for c in range(1, NUM_CLASSES + 1):
        vals = [d[c] for d in all_per_class_iou if not np.isnan(d.get(c, float('nan')))]
        avg_per_class[c] = float(np.mean(vals)) if vals else float('nan')
    avg_miou = float(np.mean(all_miou)) if all_miou else 0.0

    return avg_per_class, avg_miou


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def save_prediction_vis(img, gt_mask, pred_mask, title, save_path):
    """Save a 3-panel comparison: image | GT | prediction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(colorize_mask(gt_mask))
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(colorize_mask(pred_mask))
    axes[2].set_title("SAM2 Prediction")
    axes[2].axis("off")

    legend_patches = [
        mpatches.Patch(color=CLASS_COLOURS[c] / 255.0, label=CLASS_NAMES[c])
        for c in range(NUM_CLASSES_WITH_NODATA)
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=NUM_CLASSES_WITH_NODATA, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_oracle_visualisations(predictor, pairs, scene_name, output_dir, num_vis=NUM_VIS):
    """Generate and save oracle-prompted prediction visualisations."""
    indices = np.linspace(0, len(pairs) - 1, min(num_vis, len(pairs)), dtype=int)
    vis_dir = os.path.join(output_dir, f"vis_oracle_{scene_name.lower()}")
    os.makedirs(vis_dir, exist_ok=True)

    for i, idx in enumerate(indices):
        img_path, mask_path = pairs[idx]
        img, gt_mask = load_image_mask(img_path, mask_path)
        h, w = gt_mask.shape

        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            predictor.set_image(img)

        pred_semantic = np.zeros((h, w), dtype=np.uint8)
        pred_scores   = np.full((h, w), -1.0, dtype=np.float32)

        for class_id in range(1, NUM_CLASSES_WITH_NODATA):
            point_coord, bbox = generate_class_prompt(gt_mask, class_id)
            if point_coord is None:
                continue
            point_label = np.array([1], dtype=np.int32)
            with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
                masks, scores, _ = predictor.predict(
                    point_coords=point_coord,
                    point_labels=point_label,
                    box=bbox,
                    multimask_output=True,
                )
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]
            update_region = np.logical_and(best_mask, (best_score > pred_scores))
            pred_semantic[update_region] = class_id
            pred_scores[update_region] = best_score

        predictor.reset_predictor()

        fname = os.path.basename(img_path)
        save_path = os.path.join(vis_dir, f"oracle_{i:02d}_{fname}")
        save_prediction_vis(
            img, gt_mask, pred_semantic,
            f"Oracle-Prompted SAM2 — {scene_name} — {fname}",
            save_path
        )
    print(f"  [✓] Oracle visualisations saved to {vis_dir}")


# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------
def print_results_table(results):
    """Pretty-print the results dictionary."""
    print("\n" + "=" * 72)
    print("  ZERO-SHOT SAM2 RESULTS SUMMARY")
    print("=" * 72)
    for strategy in ["AMG (majority-vote)", "Oracle-prompted"]:
        print(f"\n  Strategy: {strategy}")
        print(f"  {'Domain':<8s}", end="")
        for c in range(1, NUM_CLASSES + 1):
            print(f"  {CLASS_NAMES[c]:>11s}", end="")
        print(f"  {'mIoU':>8s}")
        print("  " + "-" * 100)

        key_prefix = "amg" if "AMG" in strategy else "oracle"
        for scene in ["Urban", "Rural"]:
            key = f"{key_prefix}_{scene.lower()}"
            if key not in results:
                continue
            r = results[key]
            print(f"  {scene:<8s}", end="")
            for c in range(1, NUM_CLASSES + 1):
                val = r["per_class_iou"].get(str(c), float('nan'))
                if np.isnan(val):
                    print(f"  {'N/A':>11s}", end="")
                else:
                    print(f"  {val*100:>10.2f}%", end="")
            print(f"  {r['mean_iou']*100:>7.2f}%")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print("  SAM2 Zero-Shot Baseline on LoveDA")
    print("  Script 02 of 08")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----- Sanity test IoU -----
    sanity_test_iou()

    # ----- Check data exists -----
    for split in ["Train", "Val"]:
        for scene in ["Urban", "Rural"]:
            d = os.path.join(DATA_ROOT, split, scene, "images_png")
            if not os.path.isdir(d):
                print(f"  [!] Missing: {d}")
                print(f"      Run 01_setup_and_explore.py first to download the data.")
                sys.exit(1)
    print("  [✓] LoveDA data found.")

    # ----- Load SAM2 model -----
    print(f"\n[Step 1] Loading SAM2 model: {SAM2_MODEL_ID} ...")
    print(f"         Device: {DEVICE}")

    # Import SAM2
    try:
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        print("\n  [!] Cannot import sam2. Install it with:")
        print("      git clone https://github.com/facebookresearch/sam2.git")
        print("      cd sam2 && pip install -e .")
        print("      (or: pip install sam-2)")
        sys.exit(1)

    sam2_model = build_sam2_hf(SAM2_MODEL_ID, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    print(f"  [✓] SAM2 model loaded. Parameters: {sum(p.numel() for p in sam2_model.parameters()):,}")

    # Build automatic mask generator
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=AMG_POINTS_PER_SIDE,
        pred_iou_thresh=AMG_PRED_IOU_THRESH,
        stability_score_thresh=AMG_STABILITY_THRESH,
        points_per_batch=AMG_POINTS_PER_BATCH,
        output_mode="binary_mask",
    )

    # ----- Evaluate on Val split (Urban + Rural) -----
    # We use Val split because the test split has no masks.
    results = {}

    for scene in ["Urban", "Rural"]:
        pairs = get_file_pairs(DATA_ROOT, "Val", scene, subset_size=SUBSET_SIZE)
        print(f"\n{'='*60}")
        print(f"  Evaluating: Val/{scene}  ({len(pairs)} images)")
        print(f"{'='*60}")

        # Strategy A: AMG
        print("\n  --- Strategy A: Automatic Mask Generator (majority-vote) ---")
        amg_per_cls, amg_miou = evaluate_amg_zero_shot(mask_generator, pairs, scene)
        results[f"amg_{scene.lower()}"] = {
            "per_class_iou": {str(k): v for k, v in amg_per_cls.items()},
            "mean_iou": amg_miou,
        }
        print(f"  AMG mIoU ({scene}): {amg_miou*100:.2f}%")

        # Strategy B: Oracle-prompted
        print("\n  --- Strategy B: Oracle-prompted per-class ---")
        oracle_per_cls, oracle_miou = evaluate_oracle_prompted(predictor, pairs, scene)
        results[f"oracle_{scene.lower()}"] = {
            "per_class_iou": {str(k): v for k, v in oracle_per_cls.items()},
            "mean_iou": oracle_miou,
        }
        print(f"  Oracle mIoU ({scene}): {oracle_miou*100:.2f}%")

        # Save visualisations for oracle strategy
        save_oracle_visualisations(predictor, pairs, scene, OUTPUT_DIR, num_vis=NUM_VIS)

    # ----- Print results table -----
    print_results_table(results)

    # ----- Save results JSON -----
    results["metadata"] = {
        "model": SAM2_MODEL_ID,
        "subset_size": SUBSET_SIZE,
        "split": "Val",
        "amg_points_per_side": AMG_POINTS_PER_SIDE,
        "amg_pred_iou_thresh": AMG_PRED_IOU_THRESH,
        "amg_stability_thresh": AMG_STABILITY_THRESH,
        "device": DEVICE,
        "notes": (
            "AMG uses majority-vote class assignment from GT masks — this is a "
            "generous upper-bound, not a true zero-shot result. "
            "Oracle-prompted uses GT-derived point+box prompts — also not truly "
            "zero-shot. Both strategies show how well SAM2 mask shapes align "
            "with LoveDA ground truth when given perfect class/location info."
        ),
    }
    results_path = os.path.join(OUTPUT_DIR, "zero_shot_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [✓] Results saved to {results_path}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DONE — total time: {elapsed:.1f}s")
    print(f"  Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*60}")
    print("\n  Next step: run  python 03_prompt_generation.py")


if __name__ == "__main__":
    main()
