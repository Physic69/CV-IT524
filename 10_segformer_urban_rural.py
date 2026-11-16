#!/usr/bin/env python3
"""
10_segformer_urban_rural.py
===========================
Single script that trains TWO separate SegFormer-B5 models:
  1. Urban model  -> trained on Urban, best for Urban segmentation
  2. Rural model  -> trained on Rural, best for Rural segmentation

Run it and come back in ~12-13 hours. It handles everything automatically.

Fixes vs 09_segformer_train.py:
  - FIXED: LR scheduler was stepping at wrong rate (now correctly uses
    steps_per_epoch = len(loader) // GRAD_ACCUM)
  - Separate Urban and Rural models instead of combined (better per-domain accuracy)
  - Quick eval during training (100-image subset, no TTA) for speed
  - Full eval with TTA only at end of each model
  - Prints a final side-by-side comparison table

Prerequisites:
  pip install transformers albumentations

Run:
  python 10_segformer_urban_rural.py

Estimated time: ~12-13 hrs on RTX 5080 (250 epochs x 2 models)
Outputs:
  ./outputs/10_urban/    <- Urban model checkpoints + results
  ./outputs/10_rural/    <- Rural model checkpoints + results
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
# CONFIG  (adjust here if needed)
# ============================================================
DATA_ROOT     = "./data"
BASE_OUT      = "./outputs"
SEED          = 42
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID      = "nvidia/segformer-b5-finetuned-ade-640-640"

# Dataset
CROP_SIZE     = 512
SUBSET_SIZE   = None    # None = full; set 50 for smoke test

# Training
NUM_EPOCHS    = 250     # per model
BATCH_SIZE    = 4
GRAD_ACCUM    = 2       # effective batch = 8
LR            = 6e-5
LR_BACKBONE   = 6e-6   # 10x smaller for pretrained backbone
WEIGHT_DECAY  = 0.01
WARMUP_EPOCHS = 8
GRAD_CLIP     = 1.0
NUM_WORKERS   = 4
USE_AMP       = True

# OHEM
OHEM_RATIO    = 0.7

# Evaluation
EVAL_EVERY    = 10      # evaluate every N epochs during training
EVAL_SUBSET   = 100     # images per val scene during training (fast)
                        # set None to eval full val set every time (slower)

# ============================================================
# CONSTANTS
# ============================================================
NUM_CLASSES   = 8
IGNORE_INDEX  = 0
CLASS_NAMES   = ["no-data","background","building","road",
                 "water","barren","forest","agriculture"]

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
        print(f"[!] Missing packages: {missing}")
        print(f"    pip install {' '.join(missing)}")
        sys.exit(1)
    print("[OK] Dependencies: transformers + albumentations found.")


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
            assert len(imgs) > 0,           f"No images at {img_dir}"
            assert len(imgs) == len(masks), f"Count mismatch in {scene}"
            self.pairs.extend(list(zip(imgs, masks)))
        if subset_size:
            self.pairs = self.pairs[:subset_size]

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img  = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        out  = self.transform(image=img, mask=mask)
        return out["image"].float(), out["mask"].long(), img_path


# ============================================================
# CLASS WEIGHTS
# ============================================================
def compute_class_weights(pairs, n_sample=400):
    print(f"  Computing class weights (sampling {min(n_sample,len(pairs))} images)...")
    counts  = np.zeros(NUM_CLASSES, dtype=np.int64)
    indices = np.random.choice(len(pairs), min(n_sample, len(pairs)), replace=False)
    for i in tqdm(indices, desc="  Counting", ncols=70, leave=False):
        mask = np.array(Image.open(pairs[i][1]).convert("L"))
        for c in range(NUM_CLASSES):
            counts[c] += int((mask == c).sum())
    counts = counts.astype(np.float64)
    counts[IGNORE_INDEX] = 1
    freq    = counts / counts.sum()
    weights = 1.0 / (freq + 1e-6)
    weights[IGNORE_INDEX] = 0.0
    weights[1:] = weights[1:] / weights[1:].mean()
    print("  Weights: " +
          ", ".join(f"{CLASS_NAMES[c]}={weights[c]:.2f}" for c in range(1, NUM_CLASSES)))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


# ============================================================
# LOSS: Class-weighted CE + OHEM
# ============================================================
class OHEMLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=IGNORE_INDEX, ohem_ratio=OHEM_RATIO):
        super().__init__()
        self.weight       = class_weights
        self.ignore_index = ignore_index
        self.ohem_ratio   = ohem_ratio

    def forward(self, logits, labels):
        loss_map = F.cross_entropy(
            logits, labels, weight=self.weight,
            ignore_index=self.ignore_index, reduction="none")
        valid        = (labels != self.ignore_index)
        valid_losses = loss_map[valid]
        if valid_losses.numel() == 0:
            return loss_map.mean()
        k = max(1, int(self.ohem_ratio * valid_losses.numel()))
        hard, _ = valid_losses.topk(k)
        return hard.mean()


# ============================================================
# MODEL
# ============================================================
def build_model():
    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_ID,
        num_labels=NUM_CLASSES,
        id2label={i: n for i, n in enumerate(CLASS_NAMES)},
        label2id={n: i for i, n in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"  SegFormer-B5: {total:,} parameters")
    return model


# ============================================================
# LR SCHEDULER  (FIXED: uses actual scheduler steps, not loader steps)
# ============================================================
def get_scheduler(optimizer, warmup_epochs, total_epochs, scheduler_steps_per_epoch):
    """
    scheduler_steps_per_epoch = len(loader) // GRAD_ACCUM
    (scheduler.step() is called once per gradient update, not per batch)
    """
    warmup_steps = warmup_epochs * scheduler_steps_per_epoch
    total_steps  = total_epochs  * scheduler_steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-6, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(1e-7, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# INFERENCE  (full-resolution, optional TTA)
# ============================================================
@torch.no_grad()
def predict_image(model, img_np, use_tta=False):
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
    if use_tta:
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
    return per_cls, float(np.nanmean(ious)) if ious else 0.0


# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def evaluate(model, dataset, desc="Eval", subset=None, use_tta=False):
    model.eval()
    pairs = dataset.pairs
    if subset and subset < len(pairs):
        idx   = np.random.choice(len(pairs), subset, replace=False)
        pairs = [pairs[i] for i in idx]

    all_pc, all_mi = [], []
    for img_path, mask_path in tqdm(pairs, desc=f"  {desc}", ncols=80, leave=False):
        img_np  = np.array(Image.open(img_path).convert("RGB"))
        mask_np = np.array(Image.open(mask_path).convert("L"))
        pred    = predict_image(model, img_np, use_tta=use_tta)
        pc, mi  = compute_iou(pred, mask_np)
        all_pc.append(pc)
        all_mi.append(mi)

    avg_pc = {}
    for c in range(1, NUM_CLASSES):
        vals      = [d[c] for d in all_pc if not np.isnan(d.get(c, float("nan")))]
        avg_pc[c] = float(np.mean(vals)) if vals else float("nan")
    return avg_pc, float(np.nanmean(all_mi)) if all_mi else 0.0


# ============================================================
# TRAINING LOOP (one epoch)
# ============================================================
def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, epoch, total_epochs):
    model.train()
    model.segformer.eval()   # keep pretrained BN stats frozen
    total_loss, n_batches = 0.0, 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"  Epoch {epoch+1:>3}/{total_epochs}", ncols=90)
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
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}",
                         lr=f"{optimizer.param_groups[1]['lr']:.2e}")

    return total_loss / n_batches


# ============================================================
# PLOT
# ============================================================
def plot_curves(history, out_dir, title):
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(title, fontsize=13)

        ax1.plot(history["loss"], color="steelblue")
        ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss"); ax1.grid(True)

        ep = history["eval_epochs"]
        ax2.plot(ep, [v*100 for v in history["primary_miou"]],
                 label="Primary (in-domain)", marker="o", color="orangered")
        ax2.plot(ep, [v*100 for v in history["cross_miou"]],
                 label="Cross-domain", marker="s", color="steelblue", linestyle="--")
        ax2.set_title("Validation mIoU"); ax2.set_xlabel("Epoch")
        ax2.set_ylabel("mIoU (%)"); ax2.legend(); ax2.grid(True)

        plt.tight_layout()
        path = os.path.join(out_dir, "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] Curves -> {path}")
    except Exception as e:
        print(f"  [!] Could not save curves: {e}")


# ============================================================
# TRAIN ONE FULL MODEL (urban or rural)
# ============================================================
def train_model(model_name, train_scene, primary_val_scene, cross_val_scene, out_dir):
    """
    Trains a SegFormer-B5 on train_scene, evaluates on both val scenes.
    model_name      : "Urban" or "Rural"
    train_scene     : "Urban" or "Rural"
    primary_val_scene: the in-domain val scene
    cross_val_scene : the cross-domain val scene
    """
    os.makedirs(out_dir, exist_ok=True)
    banner = f"  Training {model_name} Model  (train={train_scene})"
    print("\n" + "=" * 62)
    print(banner)
    print("=" * 62)

    # Datasets
    print("\n[A] Loading datasets ...")
    train_ds       = LoveDADataset(DATA_ROOT, "train", [train_scene],
                                   SUBSET_SIZE, training=True)
    val_primary_ds = LoveDADataset(DATA_ROOT, "val", [primary_val_scene],
                                   SUBSET_SIZE, training=False)
    val_cross_ds   = LoveDADataset(DATA_ROOT, "val", [cross_val_scene],
                                   SUBSET_SIZE, training=False)

    print(f"  Train {train_scene:<6}: {len(train_ds):>5} images")
    print(f"  Val   {primary_val_scene:<6}: {len(val_primary_ds):>5} images  (primary / in-domain)")
    print(f"  Val   {cross_val_scene:<6}: {len(val_cross_ds):>5} images  (cross-domain)")

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    # Class weights
    print("\n[B] Computing class weights ...")
    class_weights = compute_class_weights(train_ds.pairs)

    # Model
    print("\n[C] Building model ...")
    model = build_model().to(DEVICE)

    # Loss + optimiser
    loss_fn = OHEMLoss(class_weights)
    backbone_params = list(model.segformer.parameters())
    decoder_params  = list(model.decode_head.parameters())
    optimizer = AdamW(
        [{"params": backbone_params, "lr": LR_BACKBONE},
         {"params": decoder_params,  "lr": LR}],
        weight_decay=WEIGHT_DECAY)

    # FIXED scheduler: steps = gradient updates per epoch, NOT batches
    sched_steps_per_epoch = max(1, len(loader) // GRAD_ACCUM)
    scheduler = get_scheduler(optimizer, WARMUP_EPOCHS,
                               NUM_EPOCHS, sched_steps_per_epoch)

    print(f"\n[D] Training {NUM_EPOCHS} epochs  "
          f"(batch={BATCH_SIZE}, gradaccum={GRAD_ACCUM}, "
          f"eff_batch={BATCH_SIZE*GRAD_ACCUM})")
    print(f"    Scheduler steps/epoch = {sched_steps_per_epoch} "
          f"(FIXED from {len(loader)} raw steps)\n")

    history  = {"loss":[], "eval_epochs":[], "primary_miou":[], "cross_miou":[]}
    best_miou, best_path = 0.0, os.path.join(out_dir, "best_model.pth")

    for epoch in range(NUM_EPOCHS):
        t0   = time.time()
        loss = train_one_epoch(model, loader, optimizer, scheduler,
                               loss_fn, epoch, NUM_EPOCHS)
        elapsed = time.time() - t0
        history["loss"].append(loss)

        cur_lr = optimizer.param_groups[1]["lr"]
        print(f"  Epoch {epoch+1:>3}/{NUM_EPOCHS}  "
              f"loss={loss:.4f}  lr={cur_lr:.2e}  time={elapsed:.0f}s")

        # Quick eval during training (subset, no TTA)
        if (epoch + 1) % EVAL_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            print(f"\n  -- Quick eval @ epoch {epoch+1} "
                  f"(subset={EVAL_SUBSET}, TTA=off) --")
            _, pm = evaluate(model, val_primary_ds, primary_val_scene,
                             EVAL_SUBSET, use_tta=False)
            _, cm = evaluate(model, val_cross_ds,   cross_val_scene,
                             EVAL_SUBSET, use_tta=False)

            print(f"    {primary_val_scene} mIoU (primary/in-domain) : {pm*100:.2f}%")
            print(f"    {cross_val_scene} mIoU (cross-domain)   : {cm*100:.2f}%\n")

            history["eval_epochs"].append(epoch + 1)
            history["primary_miou"].append(pm)
            history["cross_miou"].append(cm)

            if pm > best_miou:   # save best IN-DOMAIN performance
                best_miou = pm
                torch.save(model.state_dict(), best_path)
                print(f"  [BEST] In-domain {primary_val_scene} mIoU = {best_miou*100:.2f}%  (saved)\n")

        # Periodic checkpoint
        if (epoch + 1) % 50 == 0:
            ckpt = os.path.join(out_dir, f"checkpoint_ep{epoch+1:03d}.pth")
            torch.save({"epoch": epoch+1, "model_state": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "history": history, "best_miou": best_miou}, ckpt)
            print(f"  [OK] Checkpoint -> {ckpt}\n")

    # Final full evaluation with TTA
    print(f"\n[E] Final full evaluation (TTA=on) ...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    primary_pc, primary_miou = evaluate(model, val_primary_ds,
                                         f"Final {primary_val_scene}", None, True)
    cross_pc,   cross_miou   = evaluate(model, val_cross_ds,
                                         f"Final {cross_val_scene}",   None, True)

    # Per-class table
    print(f"\n  {'Class':<15} {primary_val_scene+' IoU':>12} {cross_val_scene+' IoU':>12}")
    print("  " + "-" * 43)
    for c in range(1, NUM_CLASSES):
        p = primary_pc.get(c, float("nan"))
        x = cross_pc.get(c, float("nan"))
        print(f"  {CLASS_NAMES[c]:<15} {p*100:>11.2f}%  {x*100:>11.2f}%")
    print("  " + "-" * 43)
    print(f"  {'mIoU':<15} {primary_miou*100:>11.2f}%  {cross_miou*100:>11.2f}%")
    print(f"  Domain gap: {abs(primary_miou - cross_miou)*100:.2f}%")

    # Save results JSON
    results = {
        "model_name": model_name, "train_scene": train_scene,
        "num_epochs": NUM_EPOCHS, "best_primary_miou": float(best_miou),
        "final": {
            primary_val_scene.lower(): {
                "miou": float(primary_miou),
                "per_class": {CLASS_NAMES[c]: float(v)
                              for c, v in primary_pc.items()}},
            cross_val_scene.lower(): {
                "miou": float(cross_miou),
                "per_class": {CLASS_NAMES[c]: float(v)
                              for c, v in cross_pc.items()}},
        },
        "history": history,
    }
    rpath = os.path.join(out_dir, "results.json")
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [OK] Results -> {rpath}")
    plot_curves(history, out_dir, f"SegFormer-B5 | Trained on {train_scene}")

    # Return final mIoUs for summary table
    return primary_miou, cross_miou, primary_pc, cross_pc


# ============================================================
# MAIN
# ============================================================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 62)
    print("  SegFormer-B5  --  Urban & Rural Separate Models")
    print(f"  {NUM_EPOCHS} epochs each  |  Target: 52-58% mIoU")
    print("=" * 62)

    check_deps()
    gpu = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    print(f"\n  Device : {DEVICE} ({gpu})")
    print(f"  Total epochs: {NUM_EPOCHS} x 2 models = {NUM_EPOCHS*2}\n")

    t_global = time.time()

    # ── MODEL 1: Urban ────────────────────────────────────────
    urban_pri, urban_cross, urban_pc, _ = train_model(
        model_name        = "Urban",
        train_scene       = "Urban",
        primary_val_scene = "Urban",
        cross_val_scene   = "Rural",
        out_dir           = os.path.join(BASE_OUT, "10_urban"),
    )

    # ── MODEL 2: Rural ────────────────────────────────────────
    rural_pri, rural_cross, rural_pc, _ = train_model(
        model_name        = "Rural",
        train_scene       = "Rural",
        primary_val_scene = "Rural",
        cross_val_scene   = "Urban",
        out_dir           = os.path.join(BASE_OUT, "10_rural"),
    )

    total_time = (time.time() - t_global) / 3600

    # ── Final Comparison Table ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    print(f"  {'Model':<20} {'Urban mIoU':>12} {'Rural mIoU':>12} {'Domain Gap':>12}")
    print("  " + "-" * 58)
    print(f"  {'Urban Model':<20} {urban_pri*100:>11.2f}%  "
          f"{urban_cross*100:>11.2f}%  {abs(urban_pri-urban_cross)*100:>10.2f}%")
    print(f"  {'Rural Model':<20} {rural_cross*100:>11.2f}%  "
          f"{rural_pri*100:>11.2f}%  {abs(rural_pri-rural_cross)*100:>10.2f}%")
    print("=" * 70)

    print(f"\n  Total training time: {total_time:.1f} hours")
    print(f"  Urban model best  -> ./outputs/10_urban/best_model.pth")
    print(f"  Rural model best  -> ./outputs/10_rural/best_model.pth")
    print("\n  For Urban images: use outputs/10_urban/best_model.pth")
    print("  For Rural images: use outputs/10_rural/best_model.pth")
    print("\n  Done!\n")


if __name__ == "__main__":
    main()
