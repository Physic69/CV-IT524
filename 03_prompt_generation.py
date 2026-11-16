#!/usr/bin/env python3
"""
03_prompt_generation.py
=======================
Automatic prompt generation from ground-truth masks for SAM2 fine-tuning.

SAM2 requires point or box prompts at inference time. Since LoveDA has no
built-in prompts, we must generate them from the GT masks. This script:

  1. Implements multiple prompt generation strategies:
     - Random foreground points (N points sampled inside each class region)
     - Centroid of largest connected component per class
     - Bounding box of each class region
     - Combined: centroid point + bounding box

  2. Visualises generated prompts overlaid on the image + mask to verify
     they land inside the correct class regions.

  3. Runs a statistical summary: how many classes per image have valid
     prompts, average component sizes, etc.

  4. Exports the prompt generation functions in a clean API that scripts
     04 and 05 will import directly.

Run:
    python 03_prompt_generation.py

Outputs saved to:  ./outputs/03_prompts/
"""

import os
import sys
import glob
import json
import time
import warnings

import numpy as np
import cv2
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# User-adjustable variables
# ---------------------------------------------------------------------------
DATA_ROOT   = "./data"
OUTPUT_DIR  = "./outputs/03_prompts"
NUM_VIS     = 8          # images to visualise per domain
NUM_POINTS  = 5          # random foreground points per class
SEED        = 42

# ---------------------------------------------------------------------------
# LoveDA constants
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "no-data", "background", "building", "road",
    "water", "barren", "forest", "agriculture",
]
NUM_CLASSES_WITH_NODATA = 8
NUM_CLASSES = 7
IGNORE_INDEX = 0

CLASS_COLOURS = np.array([
    [0,   0,   0],   [128,128,128], [255,  0,  0], [255,255,  0],
    [0,   0, 255],   [139, 69, 19], [0,  128,  0], [0,  255,  0],
], dtype=np.uint8)

# Distinct marker colours for prompt points (high contrast on any background)
PROMPT_COLOURS = [
    "none",       # 0 no-data
    "#AAAAAA",    # 1 background
    "#FF0000",    # 2 building
    "#FFD700",    # 3 road
    "#00BFFF",    # 4 water
    "#D2691E",    # 5 barren
    "#00FF00",    # 6 forest
    "#7CFC00",    # 7 agriculture
]


def colorize_mask(mask_np):
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES_WITH_NODATA):
        rgb[mask_np == c] = CLASS_COLOURS[c]
    return rgb


# ===================================================================
# PROMPT GENERATION API  (importable by scripts 04/05)
# ===================================================================

def generate_random_points(mask_np, class_id, num_points=5, rng=None):
    """
    Sample random foreground points inside a class region.

    Args:
        mask_np: (H, W) uint8 label mask
        class_id: integer class label (1-7)
        num_points: how many points to sample
        rng: numpy RandomState (for reproducibility)

    Returns:
        points: (N, 2) float32 array of (x, y) coords, or None
        labels: (N,) int32 array of 1s (foreground), or None
    """
    if rng is None:
        rng = np.random.RandomState(SEED)

    ys, xs = np.where(mask_np == class_id)
    if len(ys) == 0:
        return None, None

    n = min(num_points, len(ys))
    indices = rng.choice(len(ys), size=n, replace=False)
    points = np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)
    labels = np.ones(n, dtype=np.int32)
    return points, labels


def generate_centroid_point(mask_np, class_id):
    """
    Centroid of the largest connected component for a class.

    Returns:
        point: (1, 2) float32 array (x, y), or None
        label: (1,) int32 array [1], or None
    """
    binary = (mask_np == class_id).astype(np.uint8)
    if binary.sum() == 0:
        return None, None

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return None, None

    # Largest component (skip CC label 0 = background of binary image)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1
    cx, cy = centroids[largest_idx]

    # Safety: clamp to image bounds and verify the point is actually
    # inside the class region. If centroid falls outside (e.g. concave
    # shape), find the nearest pixel of this class instead.
    h, w = mask_np.shape
    cx_int = int(np.clip(round(cx), 0, w - 1))
    cy_int = int(np.clip(round(cy), 0, h - 1))
    if mask_np[cy_int, cx_int] != class_id:
        # Fallback: pick the pixel in this component closest to the centroid
        comp_ys, comp_xs = np.where(labels == largest_idx)
        dists = (comp_xs - cx) ** 2 + (comp_ys - cy) ** 2
        nearest = np.argmin(dists)
        cx, cy = float(comp_xs[nearest]), float(comp_ys[nearest])

    point = np.array([[cx, cy]], dtype=np.float32)
    label = np.array([1], dtype=np.int32)
    return point, label


def generate_bbox(mask_np, class_id):
    """
    Axis-aligned bounding box of the entire class region.

    Returns:
        bbox: (4,) float32 array [x1, y1, x2, y2] (XYXY), or None
    """
    ys, xs = np.where(mask_np == class_id)
    if len(ys) == 0:
        return None

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def generate_component_bbox(mask_np, class_id):
    """
    Bounding box of the LARGEST connected component for a class.

    Returns:
        bbox: (4,) float32 array [x1, y1, x2, y2], or None
    """
    binary = (mask_np == class_id).astype(np.uint8)
    if binary.sum() == 0:
        return None

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1

    x = stats[largest_idx, cv2.CC_STAT_LEFT]
    y = stats[largest_idx, cv2.CC_STAT_TOP]
    w = stats[largest_idx, cv2.CC_STAT_WIDTH]
    h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def generate_combined_prompt(mask_np, class_id, num_points=3, rng=None):
    """
    Combined prompt: centroid point + bounding box of largest component,
    plus random foreground points.

    This is the PRIMARY prompt strategy used in SAM2 fine-tuning (scripts 04/05).

    Returns:
        dict with keys:
            'point_coords': (N, 2) float32 or None
            'point_labels': (N,) int32 or None
            'box': (4,) float32 or None
        Returns None if class not present.
    """
    centroid_pt, centroid_lbl = generate_centroid_point(mask_np, class_id)
    rand_pts, rand_lbls = generate_random_points(mask_np, class_id, num_points, rng)
    bbox = generate_component_bbox(mask_np, class_id)

    if centroid_pt is None and rand_pts is None:
        return None

    # Merge centroid + random points
    all_pts = []
    all_lbls = []
    if centroid_pt is not None:
        all_pts.append(centroid_pt)
        all_lbls.append(centroid_lbl)
    if rand_pts is not None:
        all_pts.append(rand_pts)
        all_lbls.append(rand_lbls)

    point_coords = np.concatenate(all_pts, axis=0) if all_pts else None
    point_labels = np.concatenate(all_lbls, axis=0) if all_lbls else None

    return {
        'point_coords': point_coords,
        'point_labels': point_labels,
        'box': bbox,
    }


def generate_all_class_prompts(mask_np, num_points=3, rng=None):
    """
    Generate prompts for ALL classes present in a mask.

    Returns:
        dict: {class_id: prompt_dict} for each class with valid prompts.
               prompt_dict has keys 'point_coords', 'point_labels', 'box'.
    """
    prompts = {}
    for c in range(1, NUM_CLASSES_WITH_NODATA):
        prompt = generate_combined_prompt(mask_np, c, num_points, rng)
        if prompt is not None:
            prompts[c] = prompt
    return prompts


# ===================================================================
# VISUALISATION
# ===================================================================

def visualise_prompts_on_image(img, mask_np, prompts, title, save_path):
    """
    Draw generated prompts on the image and colorised mask side-by-side.

    Args:
        img: (H, W, 3) uint8
        mask_np: (H, W) uint8
        prompts: dict {class_id: {'point_coords', 'point_labels', 'box'}}
        title: figure title
        save_path: where to save
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Panel 1: Image with point prompts
    axes[0].imshow(img)
    axes[0].set_title("Image + Point Prompts")
    for cls_id, prompt in prompts.items():
        pts = prompt['point_coords']
        if pts is not None:
            colour = PROMPT_COLOURS[cls_id]
            axes[0].scatter(pts[:, 0], pts[:, 1], c=colour,
                           s=80, marker='*', edgecolors='white',
                           linewidths=0.8, zorder=5,
                           label=CLASS_NAMES[cls_id])
    axes[0].legend(fontsize=7, loc='upper right', framealpha=0.7)
    axes[0].axis("off")

    # Panel 2: Image with box prompts
    axes[1].imshow(img)
    axes[1].set_title("Image + Box Prompts")
    for cls_id, prompt in prompts.items():
        bbox = prompt['box']
        if bbox is not None:
            colour = PROMPT_COLOURS[cls_id]
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor=colour,
                                 facecolor='none', linestyle='--',
                                 label=CLASS_NAMES[cls_id])
            axes[1].add_patch(rect)
    axes[1].legend(fontsize=7, loc='upper right', framealpha=0.7)
    axes[1].axis("off")

    # Panel 3: Colorised mask with all prompts
    cmask = colorize_mask(mask_np)
    axes[2].imshow(cmask)
    axes[2].set_title("Mask + All Prompts")
    for cls_id, prompt in prompts.items():
        colour = PROMPT_COLOURS[cls_id]
        pts = prompt['point_coords']
        bbox = prompt['box']
        if pts is not None:
            axes[2].scatter(pts[:, 0], pts[:, 1], c=colour,
                           s=80, marker='*', edgecolors='white',
                           linewidths=0.8, zorder=5)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor=colour,
                                 facecolor='none', linestyle='--')
            axes[2].add_patch(rect)
    # Legend
    legend_patches = [
        mpatches.Patch(color=CLASS_COLOURS[c] / 255.0, label=CLASS_NAMES[c])
        for c in range(NUM_CLASSES_WITH_NODATA)
    ]
    axes[2].legend(handles=legend_patches, fontsize=6, loc='lower right',
                   framealpha=0.7, ncol=2)
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# VALIDATION: verify prompts land inside correct class regions
# ===================================================================

def validate_prompts(mask_np, prompts):
    """
    Check that every generated point prompt actually falls inside the
    correct class region. Returns (n_correct, n_total).
    """
    n_correct = 0
    n_total = 0
    h, w = mask_np.shape
    for cls_id, prompt in prompts.items():
        pts = prompt['point_coords']
        if pts is None:
            continue
        for (px, py) in pts:
            xi = int(np.clip(round(px), 0, w - 1))
            yi = int(np.clip(round(py), 0, h - 1))
            n_total += 1
            if mask_np[yi, xi] == cls_id:
                n_correct += 1
    return n_correct, n_total


# ===================================================================
# DATA LOADING
# ===================================================================

def get_file_pairs(root, split, scene, subset_size=None):
    img_dir  = os.path.join(root, split, scene, "images_png")
    mask_dir = os.path.join(root, split, scene, "masks_png")
    imgs  = sorted(glob.glob(os.path.join(img_dir,  "*.png")))
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    assert len(imgs) == len(masks), f"Mismatch {split}/{scene}"
    pairs = list(zip(imgs, masks))
    if subset_size is not None:
        pairs = pairs[:subset_size]
    return pairs


def load_image_mask(img_path, mask_path):
    img  = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    return img, mask


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("  Prompt Generation & Validation")
    print("  Script 03 of 08")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)

    # --- Validate on a few images per domain ---
    total_correct = 0
    total_points = 0
    stats_summary = {}

    for scene in ["Urban", "Rural"]:
        pairs = get_file_pairs(DATA_ROOT, "Train", scene)
        vis_pairs = pairs[:NUM_VIS]

        scene_correct = 0
        scene_total = 0
        classes_per_image = []

        print(f"\n  Processing Train/{scene} ({len(pairs)} total, "
              f"visualising {len(vis_pairs)}) ...")

        for i, (img_path, mask_path) in enumerate(pairs):
            img, mask = load_image_mask(img_path, mask_path)
            prompts = generate_all_class_prompts(mask, num_points=NUM_POINTS,
                                                  rng=rng)
            classes_per_image.append(len(prompts))

            nc, nt = validate_prompts(mask, prompts)
            scene_correct += nc
            scene_total += nt

            # Save visualisation for first NUM_VIS images
            if i < NUM_VIS:
                fname = os.path.basename(img_path).replace(".png", "")
                save_path = os.path.join(
                    OUTPUT_DIR,
                    f"prompts_{scene.lower()}_{i:02d}_{fname}.png"
                )
                visualise_prompts_on_image(
                    img, mask, prompts,
                    f"Generated Prompts — Train/{scene} — {fname}",
                    save_path
                )

        accuracy = scene_correct / max(scene_total, 1) * 100
        avg_classes = np.mean(classes_per_image)
        print(f"    Point placement accuracy: {scene_correct}/{scene_total} "
              f"({accuracy:.1f}%)")
        print(f"    Avg classes with prompts per image: {avg_classes:.1f}")

        total_correct += scene_correct
        total_points += scene_total

        stats_summary[scene] = {
            "num_images": len(pairs),
            "point_accuracy": round(accuracy, 2),
            "points_correct": scene_correct,
            "points_total": scene_total,
            "avg_classes_per_image": round(float(avg_classes), 2),
        }

    overall_acc = total_correct / max(total_points, 1) * 100
    print(f"\n  Overall point placement accuracy: {total_correct}/{total_points} "
          f"({overall_acc:.1f}%)")

    # --- Save summary ---
    stats_summary["overall_accuracy"] = round(overall_acc, 2)
    stats_summary["num_points_per_class"] = NUM_POINTS
    stats_summary["seed"] = SEED
    stats_path = os.path.join(OUTPUT_DIR, "prompt_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_summary, f, indent=2)
    print(f"  [✓] Stats saved to {stats_path}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DONE — total time: {elapsed:.1f}s")
    print(f"  Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*60}")
    print("\n  Next step: run  python 04_train_sam2_urban.py")
    print("  (Scripts 04/05 will import prompt functions from this file.)")


if __name__ == "__main__":
    main()
