#!/usr/bin/env python3
"""
07_train_baseline_rural.py
==========================
Train DeepLabV3-ResNet50 on LoveDA RURAL, evaluate on Rural + Urban val sets.

Identical to script 06 except TRAIN_SCENE="Rural", CROSS_SCENE="Urban".
Reuses all training logic from script 06 via import.

Run:   python 07_train_baseline_rural.py
Outputs: ./outputs/07_baseline_rural/
"""

import os, sys, time, glob, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG — RURAL variant
# ---------------------------------------------------------------------------
TRAIN_SCENE     = "Rural"
CROSS_SCENE     = "Urban"
DATA_ROOT       = "./data"
OUTPUT_DIR      = "./outputs/07_baseline_rural"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS      = 20              # MUST match script 06 for fair comparison
BATCH_SIZE      = 8
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
SAVE_EVERY      = 5
EVAL_EVERY      = 5
IMG_SIZE        = 512
SUBSET_SIZE     = None
SEED            = 42
NUM_WORKERS     = 4

NUM_CLASSES_WITH_NODATA = 8
NUM_CLASSES = 7
IGNORE_INDEX = 0

# Import shared logic from script 06
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
_s06 = import_module("06_train_baseline_urban")
compute_iou     = _s06.compute_iou
LoveDADataset   = _s06.LoveDADataset
build_deeplabv3 = _s06.build_deeplabv3
train_one_epoch = _s06.train_one_epoch
evaluate_model  = _s06.evaluate_model

def main():
    t0 = time.time()
    print("=" * 60)
    print(f"  DeepLabV3 Training on LoveDA {TRAIN_SCENE}")
    print(f"  Script 07 of 08")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    print(f"\n[1] Loading datasets (img_size={IMG_SIZE}, subset={SUBSET_SIZE})")
    train_ds = LoveDADataset(DATA_ROOT, "Train", TRAIN_SCENE, IMG_SIZE, SUBSET_SIZE)
    val_in_ds = LoveDADataset(DATA_ROOT, "Val", TRAIN_SCENE, IMG_SIZE, SUBSET_SIZE)
    val_cross_ds = LoveDADataset(DATA_ROOT, "Val", CROSS_SCENE, IMG_SIZE, SUBSET_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_in_loader = DataLoader(val_in_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_cross_loader = DataLoader(val_cross_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True)
    print(f"    Train: {len(train_ds)}, Val in: {len(val_in_ds)}, Val cross: {len(val_cross_ds)}")

    print(f"\n[2] Building DeepLabV3-ResNet50 (num_classes={NUM_CLASSES_WITH_NODATA})")
    model = build_deeplabv3(NUM_CLASSES_WITH_NODATA).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    start_epoch = 0; loss_history = []
    ckpt_path = os.path.join(OUTPUT_DIR, "latest_checkpoint.pt")
    if os.path.exists(ckpt_path):
        print(f"  Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        loss_history = ckpt.get('loss_history', [])

    print(f"\n[3] Training for {NUM_EPOCHS} epochs (starting from {start_epoch})")
    eval_results = {}

    for epoch in range(start_epoch, NUM_EPOCHS):
        ep_t0 = time.time()
        avg_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        scheduler.step()
        loss_history.append(avg_loss)
        ep_time = time.time() - ep_t0
        lr_now = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}  loss={avg_loss:.4f}  "
              f"lr={lr_now:.2e}  time={ep_time:.0f}s")

        if (epoch + 1) % SAVE_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            ckpt_data = {
                'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'loss_history': loss_history,
            }
            torch.save(ckpt_data, ckpt_path)
            torch.save(ckpt_data, os.path.join(OUTPUT_DIR, f"checkpoint_ep{epoch+1:03d}.pt"))
            print(f"    [✓] Checkpoint saved")

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            print(f"  --- Evaluating at epoch {epoch+1} ---")
            pc_in, miou_in = evaluate_model(model, val_in_loader)
            pc_cross, miou_cross = evaluate_model(model, val_cross_loader)
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
    ax.plot(range(1, len(loss_history)+1), loss_history, 'r-o', markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"DeepLabV3 {TRAIN_SCENE} Training Loss"); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    results = {
        "model": "DeepLabV3-ResNet50 (ImageNet-pretrained backbone)",
        "train_scene": TRAIN_SCENE, "cross_scene": CROSS_SCENE,
        "epochs": NUM_EPOCHS, "lr": LR, "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE, "subset_size": SUBSET_SIZE,
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
