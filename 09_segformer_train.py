#!/usr/bin/env python3
"""
09_segformer_train.py
=====================
High-performance semantic segmentation on LoveDA using SegFormer-B5.
Target: 65-70% mIoU on BOTH Urban and Rural domains.

Architecture: SegFormer-B5 (Mix Transformer encoder + lightweight All-MLP decoder)
              ~82M params, initialised from ADE20k-pretrained weights via HuggingFace.

Key improvements over scripts 04-07:
  - SegFormer-B5: purpose-built segmentation transformer (not a prompted model)
  - ADE20k to LoveDA transfer (richer segmentation knowledge than ImageNet alone)
  - Combined Urban + Rural training data (~2500 images total)
  - Random 512x512 crops from native 1024x1024 images (2x training variety)
  - Strong augmentation: flips, rot90, colour jitter, gaussian noise, grid distortion
  - Class-weighted CE loss (rescues rare classes like barren / road)
  - OHEM: Online Hard Example Mining (focus on hardest pixels each step)
  - Differential LR (backbone 10x smaller than decoder)
  - AdamW + linear warmup + cosine decay (80 epochs)
  - Gradient accumulation (effective batch = 8)
  - Gradient clipping
  - bfloat16 mixed precision (RTX 5080 / Blackwell native)
  - Full-resolution 1024x1024 inference
  - TTA: Test-Time Augmentation (H-flip + V-flip ensemble, free +1-2%)
  - Saves best checkpoint by average mIoU across both domains

Prerequisites (install once):
  pip install transformers albumentations

Run:
  python 09_segformer_train.py

Smoke test: set SUBSET_SIZE = 50 below, ~15 min

Estimated time: 8-10 hrs on RTX 5080 for 80 epochs, full dataset.
Outputs: ./outputs/09_segformer/
"""

import os, sys, glob, json, time, math, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# USER-ADJUSTABLE CONFIG
# ============================================================
DATA_ROOT      = "./data"
OUTPUT_DIR     = "./outputs/09_segformer"
SEED           = 42
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# Architecture
MODEL_ID       = "nvidia/segformer-b5-finetuned-ade-640-640"

# Dataset
CROP_SIZE      = 512        # random crop size during training
SUBSET_SIZE    = None       # None = full dataset; set to 50 for smoke test

# Training
NUM_EPOCHS     = 350
BATCH_SIZE     = 4
GRAD_ACCUM     = 2          # effective batch = BATCH_SIZE * GRAD_ACCUM = 8
LR             = 6e-5       # decoder learning rate
LR_BACKBONE    = 6e-6       # backbone lr (10x smaller - already pretrained)
WEIGHT_DECAY   = 0.01
WARMUP_EPOCHS  = 5
GRAD_CLIP      = 1.0
NUM_WORKERS    = 4
USE_AMP        = True       # bfloat16 mixed precision

# OHEM
OHEM_RATIO     = 0.7        # keep hardest 70% of valid pixels for gradient

# Evaluation
EVAL_EVERY     = 5          # evaluate every N epochs
SAVE_EVERY     = 10         # save checkpoint every N epochs
USE_TTA        = True       # test-time augmentation (H-flip + V-flip ensemble)
EVAL_SUBSET    = None       # cap eval images for speed (None = all)

# ============================================================
# CONSTANTS
# ============================================================
NUM_CLASSES    = 8          # 0=ignore/no-data, 1-7=valid classes
IGNORE_INDEX   = 0
CLASS_NAMES    = ["no-data", "background", "building", "road",
                  "water", "barren", "forest", "agriculture"]

# ImageNet normalisation (SegFormer pretrained with these)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
# DEPENDENCY CHECK
# ============================================================
def check_deps():
    missing = []
    for pkg in ["transformers", "albumentations"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n[!] Missing packages: {missing}")
        print(f"    pip install {' '.join(missing)}")
        sys.exit(1)
    print("[OK] All dependencies available.")


# ============================================================
# AUGMENTATIONS
# ============================================================
def get_transforms(training: bool):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if training:
        return A.Compose([
            A.RandomCrop(CROP_SIZE, CROP_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.1, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])


# ============================================================
# DATASET
# ============================================================
class LoveDADataset(Dataset):
    def __init__(self, root, split, scenes, subset_size=None, training=True):
        self.transform = get_transforms(training)
        self.pairs = []

        for scene in scenes:
            img_dir  = os.path.join(root, split, scene, "images_png")
            mask_dir = os.path.join(root, split, scene, "masks_png")
            imgs  = sorted(glob.glob(os.path.join(img_dir,  "*.png")))
            masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
            assert len(imgs) > 0,           f"No images found at {img_dir}"
            assert len(imgs) == len(masks), f"Image/mask count mismatch in {scene}"
            self.pairs.extend(list(zip(imgs, masks)))

        if subset_size:
            self.pairs = self.pairs[:subset_size]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img  = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        out   = self.transform(image=img, mask=mask)
        img_t = out["image"].float()
        msk_t = out["mask"].long()
        return img_t, msk_t, img_path


# ============================================================
# CLASS WEIGHTS
# ============================================================
def compute_class_weights(pairs, n_sample=500):
    print(f"  Counting pixel frequencies (sampling {min(n_sample, len(pairs))} masks) ...")
    counts  = np.zeros(NUM_CLASSES, dtype=np.int64)
    indices = np.random.choice(len(pairs), min(n_sample, len(pairs)), replace=False)

    for i in tqdm(indices, desc="  Pixel count", ncols=70, leave=False):
        _, mask_path = pairs[i]
        mask = np.array(Image.open(mask_path).convert("L"))
        for c in range(NUM_CLASSES):
            counts[c] += int((mask == c).sum())

    counts = counts.astype(np.float64)
    counts[IGNORE_INDEX] = 1
    freq    = counts / counts.sum()
    weights = 1.0 / (freq + 1e-6)
    weights[IGNORE_INDEX] = 0.0
    valid_w = weights[1:]
    weights[1:] = valid_w / valid_w.mean()

    print("  Class weights:")
    for c in range(1, NUM_CLASSES):
        print(f"    [{c}] {CLASS_NAMES[c]:<15}  w={weights[c]:.3f}   pixels={counts[c]:,.0f}")
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


# ============================================================
# LOSS: CLASS-WEIGHTED CE + OHEM
# ============================================================
class OHEMLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=IGNORE_INDEX, ohem_ratio=OHEM_RATIO):
        super().__init__()
        self.weight       = class_weights
        self.ignore_index = ignore_index
        self.ohem_ratio   = ohem_ratio

    def forward(self, logits, labels):
        loss_map = F.cross_entropy(
            logits, labels,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid        = (labels != self.ignore_index)
        valid_losses = loss_map[valid]

        if valid_losses.numel() == 0:
            return loss_map.mean()

        k = max(1, int(self.ohem_ratio * valid_losses.numel()))
        hard_losses, _ = valid_losses.topk(k)
        return hard_losses.mean()


# ============================================================
# MODEL
# ============================================================
def build_model():
    from transformers import SegformerForSemanticSegmentation

    print(f"  Loading: {MODEL_ID}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_ID,
        num_labels=NUM_CLASSES,
        id2label={i: n for i, n in enumerate(CLASS_NAMES)},
        label2id={n: i for i, n in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True,
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")
    return model


# ============================================================
# LR SCHEDULE: linear warmup + cosine decay
# ============================================================
def get_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-6, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# INFERENCE: full-resolution + optional TTA
# ============================================================
@torch.no_grad()
def predict_image(model, img_np, tta=USE_TTA):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    h, w = img_np.shape[:2]
    norm = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])

    def _fwd(img):
        t      = norm(image=img)["image"].unsqueeze(0).to(DEVICE)
        logits = model(pixel_values=t).logits
        logits = F.interpolate(logits, (h, w), mode="bilinear", align_corners=False)
        return logits.softmax(dim=1)

    probs = _fwd(img_np)
    if tta:
        probs = probs + _fwd(np.fliplr(img_np).copy()).flip(dims=[-1])
        probs = probs + _fwd(np.flipud(img_np).copy()).flip(dims=[-2])
        probs = probs / 3.0

    return probs.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.int64)


# ============================================================
# METRICS
# ============================================================
def compute_iou(pred, gt):
    valid   = (gt != IGNORE_INDEX)
    per_cls = {}
    ious    = []
    for c in range(1, NUM_CLASSES):
        pred_c = (pred == c) & valid
        gt_c   = (gt   == c) & valid
        inter  = int((pred_c & gt_c).sum())
        union  = int((pred_c | gt_c).sum())
        if union == 0:
            per_cls[c] = float("nan")
        else:
            iou = inter / union
            per_cls[c] = iou
            ious.append(iou)
    miou = float(np.nanmean(ious)) if ious else 0.0
    return per_cls, miou


# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def evaluate(model, dataset, desc="Eval", subset=None):
    model.eval()
    pairs = dataset.pairs
    if subset:
        pairs = pairs[:subset]

    all_per_cls = []
    all_miou    = []

    for img_path, mask_path in tqdm(pairs, desc=f"  {desc}", ncols=80, leave=False):
        img_np  = np.array(Image.open(img_path).convert("RGB"))
        mask_np = np.array(Image.open(mask_path).convert("L"))

        pred       = predict_image(model, img_np)
        pc, mi     = compute_iou(pred, mask_np)
        all_per_cls.append(pc)
        all_miou.append(mi)

    avg_pc = {}
    for c in range(1, NUM_CLASSES):
        vals      = [d[c] for d in all_per_cls if not np.isnan(d.get(c, float("nan")))]
        avg_pc[c] = float(np.mean(vals)) if vals else float("nan")

    avg_miou = float(np.nanmean(all_miou)) if all_miou else 0.0
    return avg_pc, avg_miou


# ============================================================
# TRAINING LOOP (one epoch)
# ============================================================
def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, epoch):
    model.train()
    model.segformer.eval()   # keep pretrained BN stats frozen

    total_loss = 0.0
    n_batches  = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"  Epoch {epoch+1:>3}", ncols=90)
    for step, (imgs, masks, _) in enumerate(pbar):
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=USE_AMP):
            logits = model(pixel_values=imgs).logits
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode="bilinear", align_corners=False)
            loss   = loss_fn(logits, masks) / GRAD_ACCUM

        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * GRAD_ACCUM
        n_batches  += 1
        pbar.set_postfix(
            loss=f"{total_loss / n_batches:.4f}",
            lr=f"{optimizer.param_groups[1]['lr']:.2e}",
        )

    return total_loss / n_batches


# ============================================================
# PLOT CURVES
# ============================================================
def plot_curves(history, out_dir):
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(history["loss"], color="steelblue")
        ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss"); ax1.grid(True)

        epochs = history["eval_epochs"]
        ax2.plot(epochs, [v*100 for v in history["urban_miou"]],
                 label="Urban", marker="o", color="orangered")
        ax2.plot(epochs, [v*100 for v in history["rural_miou"]],
                 label="Rural", marker="s", color="forestgreen")
        ax2.set_title("Validation mIoU (%)"); ax2.set_xlabel("Epoch")
        ax2.set_ylabel("mIoU (%)"); ax2.legend(); ax2.grid(True)

        plt.tight_layout()
        path = os.path.join(out_dir, "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] Curves saved -> {path}")
    except Exception as e:
        print(f"  [!] Could not save curves: {e}")


# ============================================================
# MAIN
# ============================================================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 62)
    print("  SegFormer-B5 -- LoveDA Urban + Rural Segmentation")
    print("  Script 09  |  Target: 65-70% mIoU")
    print("=" * 62)

    check_deps()
    gpu_name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    print(f"\n  Device : {DEVICE} ({gpu_name})")

    # Datasets
    print("\n[1] Loading datasets ...")
    train_ds     = LoveDADataset(DATA_ROOT, "train", ["Urban", "Rural"],
                                 SUBSET_SIZE, training=True)
    val_urban_ds = LoveDADataset(DATA_ROOT, "val",   ["Urban"],
                                 SUBSET_SIZE, training=False)
    val_rural_ds = LoveDADataset(DATA_ROOT, "val",   ["Rural"],
                                 SUBSET_SIZE, training=False)

    print(f"  Train (Urban+Rural) : {len(train_ds):>5} images")
    print(f"  Val Urban           : {len(val_urban_ds):>5} images")
    print(f"  Val Rural           : {len(val_rural_ds):>5} images")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )

    # Class weights
    print("\n[2] Computing class weights ...")
    class_weights = compute_class_weights(train_ds.pairs)

    # Model
    print("\n[3] Building model ...")
    model = build_model().to(DEVICE)

    # Loss
    loss_fn = OHEMLoss(class_weights)

    # Optimiser with differential LR
    backbone_params = list(model.segformer.parameters())
    decoder_params  = list(model.decode_head.parameters())
    optimizer = AdamW(
        [{"params": backbone_params, "lr": LR_BACKBONE},
         {"params": decoder_params,  "lr": LR}],
        weight_decay=WEIGHT_DECAY,
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = get_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, steps_per_epoch)

    # Training
    print(f"\n[4] Training for {NUM_EPOCHS} epochs ...")
    print(f"    Batch={BATCH_SIZE}, GradAccum={GRAD_ACCUM} "
          f"(eff={BATCH_SIZE*GRAD_ACCUM}), AMP={USE_AMP}, TTA={USE_TTA}\n")

    history = {"loss": [], "eval_epochs": [], "urban_miou": [], "rural_miou": []}
    best_avg_miou  = 0.0
    best_ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pth")

    for epoch in range(NUM_EPOCHS):
        t0   = time.time()
        loss = train_one_epoch(model, train_loader, optimizer,
                               scheduler, loss_fn, epoch)
        elapsed = time.time() - t0

        history["loss"].append(loss)
        print(f"  Epoch {epoch+1:>3}/{NUM_EPOCHS}  "
              f"loss={loss:.4f}  "
              f"lr={optimizer.param_groups[1]['lr']:.2e}  "
              f"time={elapsed:.0f}s")

        # Evaluate
        if (epoch + 1) % EVAL_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            print(f"\n  -- Evaluating at epoch {epoch+1} --")
            urban_pc, urban_miou = evaluate(model, val_urban_ds, "Urban", EVAL_SUBSET)
            rural_pc, rural_miou = evaluate(model, val_rural_ds, "Rural", EVAL_SUBSET)
            avg_miou = (urban_miou + rural_miou) / 2.0

            print(f"    Urban mIoU : {urban_miou*100:.2f}%")
            print(f"    Rural mIoU : {rural_miou*100:.2f}%")
            print(f"    Avg   mIoU : {avg_miou*100:.2f}%")
            print(f"    Gap        : {abs(urban_miou-rural_miou)*100:.2f}%\n")

            history["eval_epochs"].append(epoch + 1)
            history["urban_miou"].append(urban_miou)
            history["rural_miou"].append(rural_miou)

            if avg_miou > best_avg_miou:
                best_avg_miou = avg_miou
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  [BEST] Avg mIoU = {best_avg_miou*100:.2f}%  (saved)\n")

        # Periodic checkpoint
        if (epoch + 1) % SAVE_EVERY == 0:
            ckpt = os.path.join(OUTPUT_DIR, f"checkpoint_ep{epoch+1:03d}.pth")
            torch.save({
                "epoch": epoch + 1, "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(), "history": history,
                "best_avg_miou": best_avg_miou,
            }, ckpt)
            print(f"  [OK] Checkpoint -> {ckpt}\n")

    # Final evaluation
    print("\n[5] Final evaluation (best checkpoint) ...")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
    urban_pc, urban_miou = evaluate(model, val_urban_ds, "Final Urban")
    rural_pc, rural_miou = evaluate(model, val_rural_ds, "Final Rural")

    print("\n" + "=" * 68)
    print("  FINAL RESULTS")
    print("=" * 68)
    print(f"  {'Class':<15} {'Urban IoU':>12} {'Rural IoU':>12}")
    print("  " + "-" * 42)
    for c in range(1, NUM_CLASSES):
        u = urban_pc.get(c, float("nan"))
        r = rural_pc.get(c, float("nan"))
        print(f"  {CLASS_NAMES[c]:<15} {u*100:>11.2f}%  {r*100:>11.2f}%")
    print("  " + "-" * 42)
    print(f"  {'mIoU':<15} {urban_miou*100:>11.2f}%  {rural_miou*100:>11.2f}%")
    print("=" * 68)
    print(f"  Best avg mIoU : {best_avg_miou*100:.2f}%")
    print(f"  Domain gap    : {abs(urban_miou-rural_miou)*100:.2f}%")
    print("=" * 68)

    # Save results
    results = {
        "model": "SegFormer-B5", "train_data": "Urban+Rural",
        "num_epochs": NUM_EPOCHS, "best_avg_miou": float(best_avg_miou),
        "urban": {"miou": float(urban_miou),
                  "per_class": {CLASS_NAMES[c]: float(v) for c, v in urban_pc.items()}},
        "rural": {"miou": float(rural_miou),
                  "per_class": {CLASS_NAMES[c]: float(v) for c, v in rural_pc.items()}},
        "history": history,
    }
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [OK] Results -> {results_path}")

    plot_curves(history, OUTPUT_DIR)

    print(f"\n{'='*62}")
    print(f"  DONE  --  Best avg mIoU: {best_avg_miou*100:.2f}%")
    print(f"  Outputs -> {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
