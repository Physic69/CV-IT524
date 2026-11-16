#!/usr/bin/env python3
"""
11_prithvi_finetune_combined.py
===============================
Fine-tune IBM/NASA Prithvi-100M for LoveDA semantic segmentation.
Target: 65-72% mIoU.

Trains a single COMBINED model on both Urban + Rural datasets (~2500 images).
We learned from SegFormer that splitting the dataset causes severe overfitting 
because the model is too large for just 1150 images.

Architecture:
  Prithvi ViT encoder (frozen → gradually unfreezes)
  + UPerNet-style FPN decoder  
  + 1x1 conv segmentation head (8 classes)

Run:
  python 11_prithvi_finetune_combined.py

Outputs:
  ./outputs/11_prithvi_combined/
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
sys.path.insert(0, "./prithvi_weights")

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT       = "./data"
BASE_OUT        = "./outputs"
PRITHVI_WEIGHTS = "./prithvi_weights/Prithvi_EO_V1_100M.pt"
SEED            = 42
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
CROP_SIZE       = 224
SUBSET_SIZE     = None

# Training
NUM_EPOCHS      = 200
BATCH_SIZE      = 4
GRAD_ACCUM      = 2
LR              = 2e-4
LR_ENCODER      = 2e-5
WEIGHT_DECAY    = 0.05
WARMUP_EPOCHS   = 10
GRAD_CLIP       = 1.0
NUM_WORKERS     = 4
USE_AMP         = True

FREEZE_ENCODER_EPOCHS = 20
OHEM_RATIO            = 0.7

EVAL_EVERY    = 10
EVAL_SUBSET   = 100
SAVE_EVERY    = 50

# ============================================================
# CONSTANTS
# ============================================================
NUM_CLASSES  = 8
IGNORE_INDEX = 0
CLASS_NAMES  = ["no-data","background","building","road",
                "water","barren","forest","agriculture"]
PRITHVI_MEAN = np.array([775.2290, 1080.9928, 1228.5855], dtype=np.float32)
PRITHVI_STD  = np.array([1281.5261, 1270.0298, 1399.4803], dtype=np.float32)
SCALE_TO_PRITHVI = 10000.0 / 255.0


# ============================================================
# DEPENDENCY CHECK
# ============================================================
def check_deps():
    missing = []
    for pkg in ["einops", "timm", "albumentations"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[!] Missing: {missing}")
        print(f"    pip install {' '.join(missing)}")
        sys.exit(1)
    if not os.path.exists(PRITHVI_WEIGHTS):
        print(f"[!] Prithvi weights not found at {PRITHVI_WEIGHTS}")
        sys.exit(1)
    print("[OK] All dependencies and weights found.")


# ============================================================
# AUGMENTATION
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
            A.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.05, p=0.6),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])


# ============================================================
# DATASET
# ============================================================
class LoveDADataset(Dataset):
    def __init__(self, root, split, scenes, subset_size=None, training=True):
        self.transform   = get_transforms(training)
        self.pairs       = []
        for scene in scenes:
            img_dir  = os.path.join(root, split, scene, "images_png")
            mask_dir = os.path.join(root, split, scene, "masks_png")
            imgs  = sorted(glob.glob(os.path.join(img_dir,  "*.png")))
            masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
            assert len(imgs) > 0,            f"No images at {img_dir}"
            assert len(imgs) == len(masks),  f"Count mismatch in {scene}"
            self.pairs.extend(list(zip(imgs, masks)))
        if subset_size:
            self.pairs = self.pairs[:subset_size]

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img  = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        out  = self.transform(image=img, mask=mask)
        img_t = out["image"] * SCALE_TO_PRITHVI
        mean  = torch.tensor(PRITHVI_MEAN).view(3,1,1)
        std   = torch.tensor(PRITHVI_STD).view(3,1,1)
        img_t = (img_t - mean) / std
        return img_t.float(), out["mask"].long(), img_path


# ============================================================
# CLASS WEIGHTS
# ============================================================
def compute_class_weights(pairs, n_sample=400):
    print(f"  Computing class weights ({min(n_sample,len(pairs))} sample masks)...")
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
    print("  Weights: " + " | ".join(f"{CLASS_NAMES[c]}={weights[c]:.2f}" for c in range(1, NUM_CLASSES)))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


# ============================================================
# OHEM LOSS
# ============================================================
class OHEMLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=IGNORE_INDEX, ohem_ratio=OHEM_RATIO):
        super().__init__()
        self.weight       = class_weights
        self.ignore_index = ignore_index
        self.ohem_ratio   = ohem_ratio

    def forward(self, logits, labels):
        loss_map = F.cross_entropy(logits, labels, weight=self.weight,
                                   ignore_index=self.ignore_index, reduction="none")
        valid        = (labels != self.ignore_index)
        valid_losses = loss_map[valid]
        if valid_losses.numel() == 0:
            return loss_map.mean()
        k = max(1, int(self.ohem_ratio * valid_losses.numel()))
        hard, _ = valid_losses.topk(k)
        return hard.mean()


# ============================================================
# PRITHVI SEGMENTATION MODEL
# ============================================================
class PrithviSeg(nn.Module):
    FPN_LAYERS = [2, 5, 8, 11]
    def __init__(self, num_classes, img_size=224, in_chans=3, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        from prithvi_mae import PrithviViT
        self.encoder = PrithviViT(img_size=img_size, patch_size=(1,16,16), num_frames=1,
                                  in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        self.laterals = nn.ModuleList([nn.Conv2d(embed_dim, 256, kernel_size=1) for _ in self.FPN_LAYERS])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                          nn.BatchNorm2d(256), nn.ReLU(inplace=True)) for _ in self.FPN_LAYERS
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.head = nn.Conv2d(256, num_classes, kernel_size=1)
        self._init_decoder()

    def _init_decoder(self):
        for m in [*self.laterals, *self.fpn_convs, self.fusion, self.head]:
            for layer in (m if isinstance(m, nn.Sequential) else [m]):
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out")
                    if layer.bias is not None: nn.init.zeros_(layer.bias)

    def freeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = False
        print("  [Encoder FROZEN]")

    def unfreeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = True
        print("  [Encoder UNFROZEN — full fine-tuning]")

    def forward(self, x):
        B, C, H, W = x.shape
        all_features = self.encoder.forward_features(x)
        tapped = [all_features[i] for i in self.FPN_LAYERS]
        spatial = self.encoder.prepare_features_for_image_model(tapped)
        lat = [self.laterals[i](spatial[i]) for i in range(4)]
        for i in range(2, -1, -1):
            lat[i] = lat[i] + F.interpolate(lat[i+1], size=lat[i].shape[-2:], mode="bilinear", align_corners=False)
        fpn_out = [self.fpn_convs[i](lat[i]) for i in range(4)]
        target_hw = fpn_out[0].shape[-2:]
        fused = torch.cat([F.interpolate(f, target_hw, mode="bilinear", align_corners=False) for f in fpn_out], dim=1)
        logits = self.head(self.fusion(fused))
        return F.interpolate(logits, (H, W), mode="bilinear", align_corners=False)


# ============================================================
# LOAD PRETRAINED WEIGHTS
# ============================================================
def load_prithvi_weights(model, weight_path):
    print(f"  Loading Prithvi weights from {weight_path} ...")
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    adapted = {}
    for k, v in state.items():
        if k.startswith("encoder."): adapted[k] = v
        else: adapted["encoder." + k] = v

    pe_key = "encoder.patch_embed.proj.weight"
    alt_key = "patch_embed.proj.weight"
    if pe_key in adapted:
        w = adapted[pe_key]
        adapted[pe_key] = w[:, :3, ...]
        print(f"  patch_embed adapted: {tuple(w.shape)} → {tuple(adapted[pe_key].shape)}")
    elif alt_key in state:
        w = state[alt_key]
        adapted["encoder." + alt_key] = w[:, :3, ...]
        print(f"  patch_embed adapted: {tuple(w.shape)} → 3-channel")

    # The checkpoint was trained on 3 frames (589 pos_embed tokens).
    # We are using 1 frame (197 pos_embed tokens).
    # Since pos_embed is fixed sin-cos, we just delete it from the checkpoint
    # and let the model keep its natively generated 197-token pos_embed.
    if "encoder.pos_embed" in adapted:
        del adapted["encoder.pos_embed"]
    if "pos_embed" in adapted:
        del adapted["pos_embed"]

    missing, unexpected = model.load_state_dict(adapted, strict=False)
    encoder_missing = [k for k in missing if "encoder." in k]
    if encoder_missing:
        print(f"  [!] Unexpected missing encoder keys ({len(encoder_missing)}): {encoder_missing[:3]}...")
    else:
        print(f"  [OK] All encoder weights loaded. Decoder random-init ({len(missing)} keys).")


# ============================================================
# LR SCHEDULER
# ============================================================
def get_scheduler(optimizer, warmup_epochs, total_epochs, sched_steps_per_epoch):
    warmup_steps = warmup_epochs * sched_steps_per_epoch
    total_steps  = total_epochs  * sched_steps_per_epoch
    def lr_lambda(step):
        if step < warmup_steps: return max(1e-6, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(1e-7, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# INFERENCE
# ============================================================
@torch.no_grad()
def predict_image(model, img_np, use_tta=False):
    H, W = img_np.shape[:2]
    def _prep(img):
        img_f  = img.astype(np.float32) / 255.0 * SCALE_TO_PRITHVI
        img_t  = torch.tensor(img_f.transpose(2,0,1))
        mean   = torch.tensor(PRITHVI_MEAN).view(3,1,1)
        std    = torch.tensor(PRITHVI_STD).view(3,1,1)
        return ((img_t - mean) / std).unsqueeze(0).to(DEVICE)
    def _fwd(img):
        t = _prep(img)
        logits = model(t)
        return F.interpolate(logits, (H, W), mode="bilinear", align_corners=False).softmax(dim=1)

    probs = _fwd(img_np)
    if use_tta:
        probs = probs + _fwd(np.fliplr(img_np).copy()).flip(dims=[-1])
        probs = probs + _fwd(np.flipud(img_np).copy()).flip(dims=[-2])
        probs = probs / 3.0
    return probs.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.int64)


# ============================================================
# EVALUATE
# ============================================================
def compute_iou(pred, gt):
    valid = (gt != IGNORE_INDEX)
    per_cls = {}; ious = []
    for c in range(1, NUM_CLASSES):
        pc = (pred == c) & valid; gc = (gt == c) & valid
        inter = int((pc & gc).sum()); union = int((pc | gc).sum())
        if union == 0: per_cls[c] = float("nan")
        else: per_cls[c] = inter / union; ious.append(inter / union)
    return per_cls, float(np.nanmean(ious)) if ious else 0.0

@torch.no_grad()
def evaluate(model, dataset, desc="Eval", subset=None, use_tta=False):
    model.eval()
    pairs = dataset.pairs
    if subset and subset < len(pairs):
        idx = np.random.choice(len(pairs), subset, replace=False)
        pairs = [pairs[i] for i in idx]
    all_pc, all_mi = [], []
    for ip, mp in tqdm(pairs, desc=f"  {desc}", ncols=80, leave=False):
        img_np  = np.array(Image.open(ip).convert("RGB"))
        mask_np = np.array(Image.open(mp).convert("L"))
        pred    = predict_image(model, img_np, use_tta=use_tta)
        pc, mi  = compute_iou(pred, mask_np)
        all_pc.append(pc); all_mi.append(mi)
    avg_pc = {}
    for c in range(1, NUM_CLASSES):
        vals = [d[c] for d in all_pc if not np.isnan(d.get(c, float("nan")))]
        avg_pc[c] = float(np.mean(vals)) if vals else float("nan")
    return avg_pc, float(np.nanmean(all_mi)) if all_mi else 0.0


# ============================================================
# TRAIN LOOP
# ============================================================
def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, epoch, total_epochs):
    model.train()
    total_loss = 0.0; n_batches = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"  Epoch {epoch+1:>3}/{total_epochs}", ncols=90)
    for step, (imgs, masks, _) in enumerate(pbar):
        imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=USE_AMP):
            logits = model(imgs)
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss   = loss_fn(logits, masks) / GRAD_ACCUM
        loss.backward()
        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        total_loss += loss.item() * GRAD_ACCUM; n_batches += 1
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}", lr=f"{optimizer.param_groups[-1]['lr']:.2e}")
    return total_loss / n_batches

def plot_curves(history, out_dir, title):
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(title)
        ax1.plot(history["loss"], color="steelblue")
        ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch"); ax1.grid(True)
        ep = history["eval_epochs"]
        ax2.plot(ep, [v*100 for v in history["urban_miou"]], label="Urban", marker="o", color="orangered")
        ax2.plot(ep, [v*100 for v in history["rural_miou"]], label="Rural", marker="s", color="forestgreen")
        ax2.set_title("mIoU (%)"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e: print(f"  [!] Could not plot: {e}")


# ============================================================
# MAIN
# ============================================================
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    out_dir = os.path.join(BASE_OUT, "11_prithvi_combined")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 64)
    print("  Prithvi-100M Fine-tune -- COMBINED (Urban+Rural)")
    print("=" * 64)
    check_deps()
    print(f"\n  Device: {DEVICE}")

    print("\n[A] Datasets ...")
    train_ds = LoveDADataset(DATA_ROOT, "train", ["Urban", "Rural"], SUBSET_SIZE, True)
    val_urb  = LoveDADataset(DATA_ROOT, "val", ["Urban"], SUBSET_SIZE, False)
    val_rur  = LoveDADataset(DATA_ROOT, "val", ["Rural"], SUBSET_SIZE, False)
    print(f"  Train: {len(train_ds)}, Val Urban: {len(val_urb)}, Val Rural: {len(val_rur)}")

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    print("\n[B] Class weights ...")
    loss_fn = OHEMLoss(compute_class_weights(train_ds.pairs))

    print("\n[C] Building model ...")
    model = PrithviSeg(num_classes=NUM_CLASSES).to(DEVICE)
    load_prithvi_weights(model, PRITHVI_WEIGHTS)
    model.freeze_encoder()

    optimizer = AdamW([
        {"params": list(model.encoder.parameters()), "lr": LR_ENCODER},
        {"params": list(model.laterals.parameters())+list(model.fpn_convs.parameters())+list(model.fusion.parameters())+list(model.head.parameters()), "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    sched_steps = max(1, len(loader) // GRAD_ACCUM)
    scheduler = get_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, sched_steps)

    history = {"loss":[], "eval_epochs":[], "urban_miou":[], "rural_miou":[]}
    best_avg  = 0.0
    best_path = os.path.join(out_dir, "best_model.pth")
    encoder_unfrozen = False
    
    t_start = time.time()
    for epoch in range(NUM_EPOCHS):
        if epoch == FREEZE_ENCODER_EPOCHS and not encoder_unfrozen:
            model.unfreeze_encoder()
            scheduler = get_scheduler(optimizer, 0, NUM_EPOCHS - FREEZE_ENCODER_EPOCHS, max(1, len(loader) // GRAD_ACCUM))
            encoder_unfrozen = True

        loss = train_one_epoch(model, loader, optimizer, scheduler, loss_fn, epoch, NUM_EPOCHS)
        history["loss"].append(loss)
        print(f"  Epoch {epoch+1:>3}  loss={loss:.4f}  time={time.time()-t_start:.0f}s")

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            _, um = evaluate(model, val_urb, "Urban", EVAL_SUBSET, False)
            _, rm = evaluate(model, val_rur, "Rural", EVAL_SUBSET, False)
            avg = (um+rm)/2
            print(f"    Urban: {um*100:.2f}% | Rural: {rm*100:.2f}% | Avg: {avg*100:.2f}%\n")
            history["eval_epochs"].append(epoch + 1)
            history["urban_miou"].append(um); history["rural_miou"].append(rm)
            if avg > best_avg:
                best_avg = avg
                torch.save(model.state_dict(), best_path)
                print(f"  [BEST] Avg mIoU = {best_avg*100:.2f}% (saved)\n")

        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save({"epoch": epoch+1, "model_state": model.state_dict(), "history": history}, os.path.join(out_dir, f"ckpt_ep{epoch+1:03d}.pth"))

    print("\n[E] Final Full Eval (TTA=on) ...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    u_pc, u_miou = evaluate(model, val_urb, "Final Urban", None, True)
    r_pc, r_miou = evaluate(model, val_rur, "Final Rural", None, True)

    print("\n  ========================================")
    print(f"  Urban mIoU: {u_miou*100:.2f}%  |  Rural mIoU: {r_miou*100:.2f}%")
    print("  ========================================")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"best_avg": float(best_avg), "urban": float(u_miou), "rural": float(r_miou), "history": history}, f)
    plot_curves(history, out_dir, "Prithvi-100M Combined")
    print(f"\n  Done in {(time.time()-t_start)/3600:.1f} hours!\n")

if __name__ == "__main__": main()
