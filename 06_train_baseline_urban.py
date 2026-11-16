#!/usr/bin/env python3
"""
06_train_baseline_urban.py
==========================
Train DeepLabV3-ResNet50 on LoveDA URBAN, evaluate on Urban + Rural val sets.

HONESTY NOTE: torchvision's DeepLabV3 uses an ImageNet-pretrained ResNet-50
backbone. This is NOT truly "from scratch" — the backbone has seen ~1.2M
natural images. However, it has NOT seen any remote-sensing/aerial data.
This is standard practice and must be reported honestly.

The same number of epochs and comparable LR schedule as SAM2 scripts are used
to ensure a FAIR comparison. If you change epochs in 04/05, change them here too.

Run:   python 06_train_baseline_urban.py
Outputs: ./outputs/06_baseline_urban/
"""

import os, sys, time, glob, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TRAIN_SCENE     = "Urban"
CROSS_SCENE     = "Rural"
DATA_ROOT       = "./data"
OUTPUT_DIR      = "./outputs/06_baseline_urban"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Training — SAME epochs as SAM2 for fair comparison
NUM_EPOCHS      = 20
BATCH_SIZE      = 8               # DeepLabV3 is lighter than SAM2, can batch
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
SAVE_EVERY      = 5
EVAL_EVERY      = 5
IMG_SIZE        = 512             # resize for memory; adjustable
SUBSET_SIZE     = None            # set to e.g. 50 for smoke-test
SEED            = 42
NUM_WORKERS     = 4

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["no-data","background","building","road","water","barren","forest","agriculture"]
NUM_CLASSES_WITH_NODATA = 8
NUM_CLASSES = 7
IGNORE_INDEX = 0

# ---------------------------------------------------------------------------
# IoU
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
# Dataset with transforms
# ---------------------------------------------------------------------------
class LoveDADataset(Dataset):
    def __init__(self, root, split, scene, img_size=IMG_SIZE, subset_size=None):
        img_dir  = os.path.join(root, split, scene, "images_png")
        mask_dir = os.path.join(root, split, scene, "masks_png")
        imgs  = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        assert len(imgs) == len(masks)
        self.pairs = list(zip(imgs, masks))
        if subset_size:
            self.pairs = self.pairs[:subset_size]
        self.img_size = img_size
        # ImageNet normalization (required for pretrained backbone)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img = Image.open(self.pairs[idx][0]).convert("RGB")
        mask = Image.open(self.pairs[idx][1]).convert("L")

        # Resize
        img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # To tensor
        img_t = transforms.ToTensor()(img)          # (3, H, W) float [0,1]
        img_t = self.normalize(img_t)                # ImageNet-normalize
        mask_t = torch.from_numpy(np.array(mask)).long()  # (H, W)

        return img_t, mask_t

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_deeplabv3(num_classes):
    """
    Build DeepLabV3-ResNet50.

    IMPORTANT: This uses an ImageNet-pretrained ResNet-50 backbone.
    The classifier head is randomly initialized for our num_classes.
    This is NOT training from scratch — it's transfer learning from ImageNet.
    """
    from torchvision.models.segmentation import deeplabv3_resnet50

    # weights="DEFAULT" loads ImageNet-pretrained backbone
    model = deeplabv3_resnet50(weights=None, weights_backbone="DEFAULT",
                                num_classes=num_classes)

    # Print parameter info
    total = sum(p.numel() for p in model.parameters())
    print(f"  DeepLabV3-ResNet50 parameters: {total:,}")
    print(f"  NOTE: Backbone is ImageNet-pretrained (NOT from scratch).")
    print(f"        Classifier head is randomly initialized.")
    return model

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    n = 0
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    for imgs, masks in tqdm(loader, desc=f"  Epoch {epoch+1}", ncols=80, leave=False):
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        out = model(imgs)['out']  # (B, C, H, W)
        loss = criterion(out, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)

    return total_loss / max(n, 1)

@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    all_pc = []
    all_miou = []

    for imgs, masks in tqdm(loader, desc="  Evaluating", ncols=80, leave=False):
        imgs = imgs.to(DEVICE)
        out = model(imgs)['out']  # (B, C, H, W)
        preds = out.argmax(dim=1).cpu().numpy()  # (B, H, W)
        masks_np = masks.numpy()

        for i in range(preds.shape[0]):
            pc, mi = compute_iou(preds[i], masks_np[i])
            all_pc.append(pc)
            all_miou.append(mi)

    avg_pc = {}
    for c in range(1, NUM_CLASSES + 1):
        vals = [d[c] for d in all_pc if not np.isnan(d.get(c, float('nan')))]
        avg_pc[c] = float(np.mean(vals)) if vals else float('nan')
    return avg_pc, float(np.mean(all_miou)) if all_miou else 0.0

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print(f"  DeepLabV3 Training on LoveDA {TRAIN_SCENE}")
    print(f"  Script 06 of 08")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Datasets
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

    print(f"    Train: {len(train_ds)}, Val in-domain: {len(val_in_ds)}, "
          f"Val cross: {len(val_cross_ds)}")

    # Model
    print(f"\n[2] Building DeepLabV3-ResNet50 (num_classes={NUM_CLASSES_WITH_NODATA})")
    # num_classes = 8 (including no-data as class 0, ignored in loss)
    model = build_deeplabv3(NUM_CLASSES_WITH_NODATA).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Resume
    start_epoch = 0
    loss_history = []
    ckpt_path = os.path.join(OUTPUT_DIR, "latest_checkpoint.pt")
    if os.path.exists(ckpt_path):
        print(f"  Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        loss_history = ckpt.get('loss_history', [])
        print(f"  Resumed at epoch {start_epoch}")

    # Train
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
                'epoch': epoch,
                'model_state': model.state_dict(),
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

    # Loss curve
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(loss_history)+1), loss_history, 'r-o', markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"DeepLabV3 {TRAIN_SCENE} Training Loss")
    ax.grid(True, alpha=0.3)
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
