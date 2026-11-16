#!/usr/bin/env python3
"""
04_train_sam2_urban.py
=====================
Fine-tune SAM2 (frozen encoder, trainable decoder) on LoveDA URBAN train set.
Evaluate on BOTH Urban (in-domain) and Rural (cross-domain) val sets.

Architecture:
  - Image encoder: FROZEN (preserves pretrained features)
  - Prompt encoder: FROZEN
  - Mask decoder: TRAINABLE
  - Training approach: per-class binary segmentation. For each class present
    in the GT mask, generate a prompt, predict a binary mask, compute BCE+Dice
    loss against the GT binary mask for that class.

Run:   python 04_train_sam2_urban.py
Outputs: ./outputs/04_sam2_urban/
"""

import os, sys, time, glob, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG — adjust these at the top
# ---------------------------------------------------------------------------
TRAIN_SCENE     = "Urban"          # <<< THIS IS THE ONLY DIFFERENCE vs script 05
CROSS_SCENE     = "Rural"
DATA_ROOT       = "./data"
OUTPUT_DIR      = "./outputs/04_sam2_urban"
SAM2_MODEL_ID   = "facebook/sam2.1-hiera-large"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Training hyperparameters
NUM_EPOCHS      = 20
BATCH_SIZE      = 1               # SAM2 processes one image at a time for prompts
LR              = 1e-4
WEIGHT_DECAY    = 1e-4
SAVE_EVERY      = 5               # checkpoint every N epochs
EVAL_EVERY      = 5               # evaluate every N epochs
NUM_POINTS      = 3               # prompt points per class
IMG_SIZE        = 1024             # LoveDA native resolution
SUBSET_SIZE     = None             # set to e.g. 50 for smoke-test; None=full
SEED            = 42

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["no-data","background","building","road","water","barren","forest","agriculture"]
NUM_CLASSES_WITH_NODATA = 8
NUM_CLASSES = 7
IGNORE_INDEX = 0
CLASS_COLOURS = np.array([
    [0,0,0],[128,128,128],[255,0,0],[255,255,0],
    [0,0,255],[139,69,19],[0,128,0],[0,255,0]], dtype=np.uint8)

# ---------------------------------------------------------------------------
# Import prompt generation from script 03
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
_pg = import_module("03_prompt_generation")
generate_centroid_point = _pg.generate_centroid_point
generate_random_points  = _pg.generate_random_points
generate_component_bbox = _pg.generate_component_bbox

# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------
def compute_iou(pred, gt):
    valid = (gt != IGNORE_INDEX)
    per_class = {}
    ious = []
    for c in range(1, NUM_CLASSES + 1):
        pc = (pred == c) & valid
        gc = (gt == c) & valid
        inter = (pc & gc).sum()
        union = (pc | gc).sum()
        if union == 0:
            per_class[c] = float('nan')
        else:
            iou = float(inter) / float(union)
            per_class[c] = iou
            ious.append(iou)
    miou = float(np.nanmean(ious)) if ious else 0.0
    return per_class, miou

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class LoveDASegDataset(Dataset):
    def __init__(self, root, split, scene, subset_size=None):
        img_dir  = os.path.join(root, split, scene, "images_png")
        mask_dir = os.path.join(root, split, scene, "masks_png")
        imgs  = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        assert len(imgs) == len(masks)
        self.pairs = list(zip(imgs, masks))
        if subset_size:
            self.pairs = self.pairs[:subset_size]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.pairs[idx][0]).convert("RGB"))
        mask = np.array(Image.open(self.pairs[idx][1]).convert("L"))
        return img, mask, self.pairs[idx][0]

# ---------------------------------------------------------------------------
# Loss: BCE + Dice
# ---------------------------------------------------------------------------
def bce_dice_loss(pred_logits, target_binary):
    """
    pred_logits: (1, H, W) float — raw logits from mask decoder
    target_binary: (1, H, W) float — 0/1 ground truth
    """
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_binary)
    pred_prob = torch.sigmoid(pred_logits)
    inter = (pred_prob * target_binary).sum()
    union = pred_prob.sum() + target_binary.sum()
    dice = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
    return bce + dice

# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------
def setup_sam2_for_finetuning(model):
    """Freeze everything except mask decoder."""
    # Freeze all
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze mask decoder
    for p in model.sam_mask_decoder.parameters():
        p.requires_grad = True
    # Print verification
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  === PARAMETER FREEZE VERIFICATION ===")
    print(f"  Total parameters:     {total:>12,}")
    print(f"  Trainable parameters: {trainable:>12,}")
    print(f"  Frozen parameters:    {total - trainable:>12,}")
    print(f"  Trainable fraction:   {trainable/total*100:.2f}%")
    print(f"\n  Trainable modules:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"    {name}: {list(p.shape)}")
    print(f"  ===================================\n")
    return trainable

# ---------------------------------------------------------------------------
# Training loop (one epoch)
# ---------------------------------------------------------------------------
def train_one_epoch(model, predictor, dataset, optimizer, epoch, rng):
    model.train()
    # Keep encoder in eval mode (frozen, BatchNorm should not update)
    model.image_encoder.eval()
    if hasattr(model, 'sam_prompt_encoder'):
        model.sam_prompt_encoder.eval()

    total_loss = 0.0
    n_samples = 0

    for idx in tqdm(range(len(dataset)), desc=f"  Epoch {epoch+1}", ncols=80):
        img, mask, _ = dataset[idx]
        h, w = mask.shape

        # Compute image embeddings (frozen — no grad)
        with torch.no_grad():
            predictor.set_image(img)

        epoch_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        n_classes_in_image = 0

        for class_id in range(1, NUM_CLASSES_WITH_NODATA):
            binary_gt = (mask == class_id)
            if binary_gt.sum() < 100:  # skip tiny regions
                continue

            # Generate prompt
            pt, pt_lbl = generate_centroid_point(mask, class_id)
            rpts, rlbls = generate_random_points(mask, class_id, NUM_POINTS, rng)
            bbox = generate_component_bbox(mask, class_id)

            if pt is None:
                continue

            # Merge points
            all_pts = [pt]
            all_lbls = [pt_lbl]
            if rpts is not None:
                all_pts.append(rpts)
                all_lbls.append(rlbls)
            point_coords = np.concatenate(all_pts, axis=0)
            point_labels = np.concatenate(all_lbls, axis=0)

            # Prepare prompts as tensors
            coord_tensor = torch.as_tensor(point_coords, dtype=torch.float, device=DEVICE)
            unnorm_coords = predictor._transforms.transform_coords(
                coord_tensor, normalize=True, orig_hw=predictor._orig_hw[0]
            )
            label_tensor = torch.as_tensor(point_labels, dtype=torch.int, device=DEVICE)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords = unnorm_coords[None, ...]
                label_tensor = label_tensor[None, ...]

            unnorm_box = None
            if bbox is not None:
                box_t = torch.as_tensor(bbox, dtype=torch.float, device=DEVICE)
                unnorm_box = predictor._transforms.transform_boxes(
                    box_t, normalize=True, orig_hw=predictor._orig_hw[0]
                )

            # Merge box into concat_points (same as SAM2 internal logic)
            concat_points = (unnorm_coords, label_tensor)
            if unnorm_box is not None:
                box_coords = unnorm_box.reshape(-1, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=DEVICE)
                box_labels = box_labels.repeat(box_coords.size(0), 1)
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)

            # Forward through prompt encoder + mask decoder (trainable)
            sparse_emb, dense_emb = model.sam_prompt_encoder(
                points=concat_points, boxes=None, masks=None
            )

            high_res_features = [
                feat_level[0].unsqueeze(0)
                for feat_level in predictor._features["high_res_feats"]
            ]

            low_res_masks, iou_preds, _, _ = model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][0].unsqueeze(0),
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )

            # Upscale to original resolution
            pred_masks = F.interpolate(
                low_res_masks, (h, w), mode="bilinear", align_corners=False
            )  # (1, 1, H, W)

            # Build GT binary tensor
            gt_binary = torch.as_tensor(
                binary_gt.astype(np.float32), device=DEVICE
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            loss = bce_dice_loss(pred_masks.squeeze(0), gt_binary.squeeze(0))
            epoch_loss = epoch_loss + loss
            n_classes_in_image += 1

        predictor.reset_predictor()

        if n_classes_in_image > 0:
            avg_loss = epoch_loss / n_classes_in_image
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            total_loss += avg_loss.item()
            n_samples += 1

    return total_loss / max(n_samples, 1)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, predictor, dataset, rng):
    model.eval()
    all_per_cls = []
    all_miou = []

    for idx in tqdm(range(len(dataset)), desc="  Evaluating", ncols=80):
        img, mask, _ = dataset[idx]
        h, w = mask.shape
        predictor.set_image(img)

        pred_semantic = np.zeros((h, w), dtype=np.uint8)
        pred_scores = np.full((h, w), -1.0, dtype=np.float32)

        for class_id in range(1, NUM_CLASSES_WITH_NODATA):
            pt, pt_lbl = generate_centroid_point(mask, class_id)
            bbox = generate_component_bbox(mask, class_id)
            if pt is None:
                continue

            masks, scores, _ = predictor.predict(
                point_coords=pt, point_labels=pt_lbl,
                box=bbox, multimask_output=True,
            )
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
            best_score = scores[best_idx]
            update = np.logical_and(best_mask, (best_score > pred_scores))
            pred_semantic[update] = class_id
            pred_scores[update] = best_score

        predictor.reset_predictor()
        pc, mi = compute_iou(pred_semantic, mask)
        all_per_cls.append(pc)
        all_miou.append(mi)

    avg_pc = {}
    for c in range(1, NUM_CLASSES + 1):
        vals = [d[c] for d in all_per_cls if not np.isnan(d.get(c, float('nan')))]
        avg_pc[c] = float(np.mean(vals)) if vals else float('nan')
    avg_miou = float(np.mean(all_miou)) if all_miou else 0.0
    return avg_pc, avg_miou

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print(f"  SAM2 Fine-tuning on LoveDA {TRAIN_SCENE}")
    print(f"  Script 04 of 08")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)
    torch.manual_seed(SEED)

    # --- Load SAM2 ---
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print(f"\n[1] Loading SAM2: {SAM2_MODEL_ID}")
    sam2_model = build_sam2_hf(SAM2_MODEL_ID, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)

    # --- Freeze encoder, unfreeze decoder ---
    setup_sam2_for_finetuning(sam2_model)

    # --- Datasets ---
    print(f"[2] Loading datasets (SUBSET_SIZE={SUBSET_SIZE})")
    train_ds = LoveDASegDataset(DATA_ROOT, "Train", TRAIN_SCENE, SUBSET_SIZE)
    val_in   = LoveDASegDataset(DATA_ROOT, "Val", TRAIN_SCENE, SUBSET_SIZE)
    val_cross= LoveDASegDataset(DATA_ROOT, "Val", CROSS_SCENE, SUBSET_SIZE)
    print(f"    Train: {len(train_ds)}, Val in-domain: {len(val_in)}, Val cross: {len(val_cross)}")

    # --- Optimizer (only trainable params) ---
    trainable_params = [p for p in sam2_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # --- Check for resume ---
    start_epoch = 0
    loss_history = []
    ckpt_path = os.path.join(OUTPUT_DIR, "latest_checkpoint.pt")
    if os.path.exists(ckpt_path):
        print(f"  Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        sam2_model.sam_mask_decoder.load_state_dict(ckpt['decoder_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        loss_history = ckpt.get('loss_history', [])
        print(f"  Resumed at epoch {start_epoch}")

    # --- Training loop ---
    print(f"\n[3] Training for {NUM_EPOCHS} epochs (starting from {start_epoch})")
    eval_results = {}

    for epoch in range(start_epoch, NUM_EPOCHS):
        ep_t0 = time.time()
        avg_loss = train_one_epoch(sam2_model, predictor, train_ds, optimizer, epoch, rng)
        scheduler.step()
        loss_history.append(avg_loss)
        ep_time = time.time() - ep_t0
        lr_now = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}  loss={avg_loss:.4f}  "
              f"lr={lr_now:.2e}  time={ep_time:.0f}s")

        # Save checkpoint
        if (epoch + 1) % SAVE_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            ckpt_data = {
                'epoch': epoch,
                'decoder_state': sam2_model.sam_mask_decoder.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'loss_history': loss_history,
            }
            torch.save(ckpt_data, ckpt_path)
            torch.save(ckpt_data, os.path.join(OUTPUT_DIR, f"checkpoint_ep{epoch+1:03d}.pt"))
            print(f"    [✓] Checkpoint saved")

        # Evaluate
        if (epoch + 1) % EVAL_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            print(f"  --- Evaluating at epoch {epoch+1} ---")
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pc_in, miou_in = evaluate(sam2_model, predictor, val_in, rng)
                pc_cross, miou_cross = evaluate(sam2_model, predictor, val_cross, rng)
            print(f"    In-domain  ({TRAIN_SCENE}) mIoU: {miou_in*100:.2f}%")
            print(f"    Cross-dom  ({CROSS_SCENE}) mIoU: {miou_cross*100:.2f}%")
            print(f"    Domain gap: {(miou_in - miou_cross)*100:.2f}%")
            eval_results[f"epoch_{epoch+1}"] = {
                "in_domain": {"per_class": {str(k):v for k,v in pc_in.items()}, "miou": miou_in},
                "cross_domain": {"per_class": {str(k):v for k,v in pc_cross.items()}, "miou": miou_cross},
            }

    # --- Save loss curve ---
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(loss_history)+1), loss_history, 'b-o', markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(f"SAM2 {TRAIN_SCENE} Training Loss")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Save results ---
    results = {
        "model": SAM2_MODEL_ID, "train_scene": TRAIN_SCENE,
        "cross_scene": CROSS_SCENE, "epochs": NUM_EPOCHS,
        "lr": LR, "subset_size": SUBSET_SIZE,
        "loss_history": loss_history, "eval_results": eval_results,
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DONE — {elapsed/60:.1f} min")
    print(f"  Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
