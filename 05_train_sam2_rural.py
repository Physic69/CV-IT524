#!/usr/bin/env python3
"""
05_train_sam2_rural.py
=====================
Fine-tune SAM2 (frozen encoder, trainable decoder) on LoveDA RURAL train set.
Evaluate on BOTH Rural (in-domain) and Urban (cross-domain) val sets.

This is identical to script 04 except TRAIN_SCENE="Rural" / CROSS_SCENE="Urban".
All architecture, loss, and training logic is shared.

Run:   python 05_train_sam2_rural.py
Outputs: ./outputs/05_sam2_rural/
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
# CONFIG — RURAL variant
# ---------------------------------------------------------------------------
TRAIN_SCENE     = "Rural"          # <<< ONLY DIFFERENCE from script 04
CROSS_SCENE     = "Urban"
DATA_ROOT       = "./data"
OUTPUT_DIR      = "./outputs/05_sam2_rural"
SAM2_MODEL_ID   = "facebook/sam2.1-hiera-large"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS      = 20              # same as script 04 for fair comparison
BATCH_SIZE      = 1
LR              = 1e-4
WEIGHT_DECAY    = 1e-4
SAVE_EVERY      = 5
EVAL_EVERY      = 5
NUM_POINTS      = 3
IMG_SIZE        = 1024
SUBSET_SIZE     = None
SEED            = 42

# ---------------------------------------------------------------------------
# Reuse ALL logic from script 04
# ---------------------------------------------------------------------------
CLASS_NAMES = ["no-data","background","building","road","water","barren","forest","agriculture"]
NUM_CLASSES_WITH_NODATA = 8
NUM_CLASSES = 7
IGNORE_INDEX = 0
CLASS_COLOURS = np.array([
    [0,0,0],[128,128,128],[255,0,0],[255,255,0],
    [0,0,255],[139,69,19],[0,128,0],[0,255,0]], dtype=np.uint8)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
_pg = import_module("03_prompt_generation")
generate_centroid_point = _pg.generate_centroid_point
generate_random_points  = _pg.generate_random_points
generate_component_bbox = _pg.generate_component_bbox

# Import shared functions from script 04
_s04 = import_module("04_train_sam2_urban")
compute_iou             = _s04.compute_iou
LoveDASegDataset        = _s04.LoveDASegDataset
bce_dice_loss           = _s04.bce_dice_loss
setup_sam2_for_finetuning = _s04.setup_sam2_for_finetuning
train_one_epoch         = _s04.train_one_epoch
evaluate                = _s04.evaluate

# ---------------------------------------------------------------------------
# MAIN (same logic as 04, different config)
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print(f"  SAM2 Fine-tuning on LoveDA {TRAIN_SCENE}")
    print(f"  Script 05 of 08")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)
    torch.manual_seed(SEED)

    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print(f"\n[1] Loading SAM2: {SAM2_MODEL_ID}")
    sam2_model = build_sam2_hf(SAM2_MODEL_ID, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    setup_sam2_for_finetuning(sam2_model)

    print(f"[2] Loading datasets (SUBSET_SIZE={SUBSET_SIZE})")
    train_ds = LoveDASegDataset(DATA_ROOT, "Train", TRAIN_SCENE, SUBSET_SIZE)
    val_in   = LoveDASegDataset(DATA_ROOT, "Val", TRAIN_SCENE, SUBSET_SIZE)
    val_cross= LoveDASegDataset(DATA_ROOT, "Val", CROSS_SCENE, SUBSET_SIZE)
    print(f"    Train: {len(train_ds)}, Val in-domain: {len(val_in)}, Val cross: {len(val_cross)}")

    trainable_params = [p for p in sam2_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

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

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(loss_history)+1), loss_history, 'b-o', markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(f"SAM2 {TRAIN_SCENE} Training Loss")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

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
