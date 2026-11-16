#!/usr/bin/env python3
"""
11_prithvi_finetune.py
======================
Fine-tune IBM/NASA Prithvi-100M for LoveDA semantic segmentation.
Target: 65-72% mIoU (Urban + Rural separate models).

Why Prithvi beats SegFormer-B5 for this task:
  - Pretrained on NASA HLS satellite imagery (same modality as LoveDA!)
  - ViT-Large (768-dim, 12 layers) — already understands terrain, vegetation,
    buildings, water from a satellite view
  - We simply teach it the 7 LoveDA class labels on top

Architecture:
  Prithvi ViT encoder (frozen → gradually unfreezes)
  + UPerNet-style FPN decoder  
  + 1x1 conv segmentation head (8 classes)

Channel adaptation:
  Prithvi expects 6 bands (B02 B03 B04 B05 B06 B07).
  LoveDA is RGB (3 bands).
  We slice the patch_embed weights to keep only the first 3 channels (B02=Blue,
  B03=Green, B04=Red which maps perfectly to RGB).

Run:
  pip install einops timm
  python 11_prithvi_finetune.py

Estimated time: ~10-12 hrs on RTX 5080 (200 epochs x 2 models)
Outputs:
  ./outputs/11_prithvi_urban/
  ./outputs/11_prithvi_rural/
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
sys.path.insert(0, "./prithvi_weights")   # for prithvi_mae.py

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT       = "./data"
BASE_OUT        = "./outputs"
PRITHVI_WEIGHTS = "./prithvi_weights/Prithvi_EO_V1_100M.pt"
SEED            = 42
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
CROP_SIZE       = 224     # Prithvi native patch size — multiples are fine
SUBSET_SIZE     = None    # None = full; 50 for smoke test

# Training
NUM_EPOCHS      = 200     # per model
BATCH_SIZE      = 4
GRAD_ACCUM      = 2       # effective batch = 8
LR              = 2e-4    # decoder LR
LR_ENCODER      = 2e-5   # encoder LR (10x smaller)
WEIGHT_DECAY    = 0.05
WARMUP_EPOCHS   = 10
GRAD_CLIP       = 1.0
NUM_WORKERS     = 4
USE_AMP         = True

# Encoder unfreezing schedule
FREEZE_ENCODER_EPOCHS = 20   # freeze encoder for first N epochs, then unfreeze
OHEM_RATIO            = 0.7

# Evaluation
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
# Prithvi per-channel stats (B02 B03 B04 only — first 3 of 6 bands)
PRITHVI_MEAN = np.array([775.2290, 1080.9928, 1228.5855], dtype=np.float32)
PRITHVI_STD  = np.array([1281.5261, 1270.0298, 1399.4803], dtype=np.float32)
# We still normalise LoveDA RGB with these to keep scale consistent.
# LoveDA pixel values are 0-255; Prithvi trained on 0~10000 reflectance.
# We rescale: pixel / 255 * 10000 then normalise with Prithvi stats.
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
            A.ToFloat(max_value=255.0),   # 0-1
            ToTensorV2(),                 # (C, H, W) float
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
        # Scale to Prithvi reflectance range and normalise
        img_t = out["image"] * SCALE_TO_PRITHVI                          # (3,H,W) float
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
    print("  Weights: " +
          " | ".join(f"{CLASS_NAMES[c]}={weights[c]:.2f}" for c in range(1, NUM_CLASSES)))
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
    """
    Prithvi ViT encoder + FPN neck + segmentation head.

    Encoder output: list of (B, T*embed_dim, H/16, W/16) feature maps
    We pick 4 layers (3, 5, 8, 11) for a 4-scale FPN, then decode.
    """

    FPN_LAYERS = [2, 5, 8, 11]   # 0-indexed block indices to tap

    def __init__(self, num_classes, img_size=224, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        from prithvi_mae import PrithviViT

        self.encoder = PrithviViT(
            img_size    = img_size,
            patch_size  = (1, 16, 16),
            num_frames  = 1,
            in_chans    = in_chans,
            embed_dim   = embed_dim,
            depth       = depth,
            num_heads   = num_heads,
        )

        # FPN lateral convolutions (compress embed_dim → 256)
        self.laterals = nn.ModuleList([
            nn.Conv2d(embed_dim, 256, kernel_size=1) for _ in self.FPN_LAYERS
        ])
        # FPN output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ) for _ in self.FPN_LAYERS
        ])
        # Fusion + head
        self.fusion = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(256, num_classes, kernel_size=1)

        self._init_decoder()

    def _init_decoder(self):
        for m in [*self.laterals, *self.fpn_convs, self.fusion, self.head]:
            for layer in (m if isinstance(m, nn.Sequential) else [m]):
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        print("  [Encoder FROZEN]")

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        print("  [Encoder UNFROZEN — full fine-tuning]")

    def forward(self, x):
        # x: (B, 3, H, W)
        B, C, H, W = x.shape
        # Prithvi expects (B, C, T, H, W) with T=1, or we use forward_features
        # which adds time dim automatically when input_size[0]==1
        all_features = self.encoder.forward_features(x)
        # all_features: list of length=depth, each (B, 1+N, embed_dim)
        # where N = (H/16)*(W/16)

        # Tap 4 layers
        tapped = [all_features[i] for i in self.FPN_LAYERS]
        # Convert to spatial maps via prepare_features_for_image_model
        spatial = self.encoder.prepare_features_for_image_model(tapped)
        # spatial: list of (B, embed_dim, H/16, W/16) since T=1

        h_feat = H // 16
        w_feat = W // 16

        # Lateral projections
        lat = [self.laterals[i](spatial[i]) for i in range(4)]

        # Top-down FPN
        for i in range(2, -1, -1):
            lat[i] = lat[i] + F.interpolate(lat[i+1], size=lat[i].shape[-2:],
                                              mode="bilinear", align_corners=False)

        # FPN output convs
        fpn_out = [self.fpn_convs[i](lat[i]) for i in range(4)]

        # Upsample all to same scale (largest = lat[0] = H/16)
        target_hw = fpn_out[0].shape[-2:]
        fused = torch.cat([F.interpolate(f, target_hw, mode="bilinear", align_corners=False)
                           for f in fpn_out], dim=1)   # (B, 256*4, h, w)

        fused = self.fusion(fused)            # (B, 256, h, w)
        logits = self.head(fused)             # (B, num_classes, h, w)

        # Upsample back to input resolution
        logits = F.interpolate(logits, (H, W), mode="bilinear", align_corners=False)
        return logits


# ============================================================
# LOAD PRETRAINED WEIGHTS (6-chan → 3-chan adaptation)
# ============================================================
def load_prithvi_weights(model, weight_path):
    print(f"  Loading Prithvi weights from {weight_path} ...")
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)

    # The .pt file may be wrapped in a "model" key or plain
    state = ckpt.get("model", ckpt)

    # Rename keys: Prithvi MAE encoder keys start with "encoder."
    # Our model uses self.encoder.xxx
    adapted = {}
    for k, v in state.items():
        if k.startswith("encoder."):
            adapted[k] = v
        else:
            # Might be bare keys (PrithviViT saved directly)
            adapted["encoder." + k] = v

    # --- Channel adaptation: 6-chan patch_embed → 3-chan ---
    pe_key = "encoder.patch_embed.proj.weight"
    alt_key = "patch_embed.proj.weight"   # in case no prefix

    if pe_key in adapted:
        w = adapted[pe_key]                      # (embed_dim, 6, 1, 16, 16)
        adapted[pe_key] = w[:, :3, ...]          # keep first 3 bands (B02=B, B03=G, B04=R)
        print(f"  patch_embed adapted: {tuple(w.shape)} → {tuple(adapted[pe_key].shape)}")
    elif alt_key in state:
        w = state[alt_key]
        adapted["encoder." + alt_key] = w[:, :3, ...]
        print(f"  patch_embed adapted: {tuple(w.shape)} → 3-channel")

    if "encoder.pos_embed" in adapted:
        del adapted["encoder.pos_embed"]
    if "pos_embed" in adapted:
        del adapted["pos_embed"]

    missing, unexpected = model.load_state_dict(adapted, strict=False)
    # Decoder params (laterals, fpn, head) will be in missing — that's expected
    encoder_missing = [k for k in missing if "encoder." in k]
    if encoder_missing:
        print(f"  [!] Unexpected missing encoder keys ({len(encoder_missing)}): "
              f"{encoder_missing[:3]}...")
    else:
        print(f"  [OK] All encoder weights loaded. "
              f"Decoder random-init ({len(missing)} keys).")


# ============================================================
# LR SCHEDULER (FIXED: actual gradient steps)
# ============================================================
def get_scheduler(optimizer, warmup_epochs, total_epochs, sched_steps_per_epoch):
    warmup_steps = warmup_epochs * sched_steps_per_epoch
    total_steps  = total_epochs  * sched_steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-6, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(1e-7, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# INFERENCE
# ============================================================
@torch.no_grad()
def predict_image(model, img_np, use_tta=False):
    H, W = img_np.shape[:2]
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    def _prep(img):
        img_f  = img.astype(np.float32) / 255.0 * SCALE_TO_PRITHVI
        img_t  = torch.tensor(img_f.transpose(2,0,1))  # (3,H,W)
        mean   = torch.tensor(PRITHVI_MEAN).view(3,1,1)
        std    = torch.tensor(PRITHVI_STD).view(3,1,1)
        return ((img_t - mean) / std).unsqueeze(0).to(DEVICE)   # (1,3,H,W)

    def _fwd(img):
        t      = _prep(img)
        logits = model(t)
        logits = F.interpolate(logits, (H, W), mode="bilinear", align_corners=False)
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
    per_cls = {}; ious = []
    for c in range(1, NUM_CLASSES):
        pc = (pred == c) & valid; gc = (gt == c) & valid
        inter = int((pc & gc).sum()); union = int((pc | gc).sum())
        if union == 0:
            per_cls[c] = float("nan")
        else:
            per_cls[c] = inter / union; ious.append(inter / union)
    return per_cls, float(np.nanmean(ious)) if ious else 0.0


# ============================================================
# EVALUATE
# ============================================================
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
# TRAIN ONE EPOCH
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
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode="bilinear", align_corners=False)
            loss   = loss_fn(logits, masks) / GRAD_ACCUM

        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * GRAD_ACCUM; n_batches += 1
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}",
                         lr=f"{optimizer.param_groups[-1]['lr']:.2e}")

    return total_loss / n_batches


# ============================================================
# PLOT
# ============================================================
def plot_curves(history, out_dir, title):
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(title)
        ax1.plot(history["loss"], color="steelblue")
        ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch"); ax1.grid(True)
        ep = history["eval_epochs"]
        ax2.plot(ep, [v*100 for v in history["primary_miou"]],
                 label="Primary", marker="o", color="orangered")
        ax2.plot(ep, [v*100 for v in history["cross_miou"]],
                 label="Cross-domain", marker="s", color="steelblue", linestyle="--")
        ax2.set_title("mIoU (%)"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)
        plt.tight_layout()
        path = os.path.join(out_dir, "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] Curves -> {path}")
    except Exception as e:
        print(f"  [!] Could not plot: {e}")


# ============================================================
# TRAIN ONE FULL MODEL
# ============================================================
def train_model(model_name, train_scene, primary_scene, cross_scene, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "=" * 64)
    print(f"  Prithvi-100M | {model_name} Model | train={train_scene}")
    print("=" * 64)

    # Datasets
    print("\n[A] Datasets ...")
    train_ds = LoveDADataset(DATA_ROOT, "train", [train_scene], SUBSET_SIZE, True)
    val_pri  = LoveDADataset(DATA_ROOT, "val",   [primary_scene], SUBSET_SIZE, False)
    val_crs  = LoveDADataset(DATA_ROOT, "val",   [cross_scene],   SUBSET_SIZE, False)
    print(f"  Train: {len(train_ds)}, Val {primary_scene}: {len(val_pri)}, "
          f"Val {cross_scene}: {len(val_crs)}")

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    # Class weights
    print("\n[B] Class weights ...")
    class_weights = compute_class_weights(train_ds.pairs)
    loss_fn = OHEMLoss(class_weights)

    # Model
    print("\n[C] Building Prithvi-Seg model ...")
    model = PrithviSeg(num_classes=NUM_CLASSES).to(DEVICE)
    load_prithvi_weights(model, PRITHVI_WEIGHTS)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    # Freeze encoder initially
    model.freeze_encoder()

    # Optimiser (separate param groups)
    encoder_params = list(model.encoder.parameters())
    decoder_params = (list(model.laterals.parameters()) +
                      list(model.fpn_convs.parameters()) +
                      list(model.fusion.parameters()) +
                      list(model.head.parameters()))

    optimizer = AdamW([
        {"params": encoder_params, "lr": LR_ENCODER},
        {"params": decoder_params, "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    sched_steps = max(1, len(loader) // GRAD_ACCUM)
    scheduler   = get_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, sched_steps)

    print(f"\n[D] Training {NUM_EPOCHS} epochs | "
          f"batch={BATCH_SIZE} | eff_batch={BATCH_SIZE*GRAD_ACCUM}")
    print(f"    Encoder frozen for first {FREEZE_ENCODER_EPOCHS} epochs\n")

    history = {"loss":[], "eval_epochs":[], "primary_miou":[], "cross_miou":[]}
    best_miou  = 0.0
    best_path  = os.path.join(out_dir, "best_model.pth")
    encoder_unfrozen = False

    for epoch in range(NUM_EPOCHS):

        # Unfreeze encoder after warmup
        if epoch == FREEZE_ENCODER_EPOCHS and not encoder_unfrozen:
            model.unfreeze_encoder()
            # Reset scheduler so encoder gets proper cosine decay too
            sched_steps2 = max(1, len(loader) // GRAD_ACCUM)
            scheduler = get_scheduler(optimizer, 0,
                                       NUM_EPOCHS - FREEZE_ENCODER_EPOCHS,
                                       sched_steps2)
            encoder_unfrozen = True

        t0   = time.time()
        loss = train_one_epoch(model, loader, optimizer, scheduler,
                               loss_fn, epoch, NUM_EPOCHS)
        elapsed = time.time() - t0
        history["loss"].append(loss)
        print(f"  Epoch {epoch+1:>3}/{NUM_EPOCHS}  loss={loss:.4f}  "
              f"lr={optimizer.param_groups[-1]['lr']:.2e}  time={elapsed:.0f}s")

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            print(f"\n  -- Quick eval @ epoch {epoch+1} --")
            _, pm = evaluate(model, val_pri, primary_scene, EVAL_SUBSET, False)
            _, cm = evaluate(model, val_crs, cross_scene,   EVAL_SUBSET, False)
            print(f"    {primary_scene} mIoU: {pm*100:.2f}%  |  "
                  f"{cross_scene} mIoU: {cm*100:.2f}%\n")
            history["eval_epochs"].append(epoch + 1)
            history["primary_miou"].append(pm)
            history["cross_miou"].append(cm)
            if pm > best_miou:
                best_miou = pm
                torch.save(model.state_dict(), best_path)
                print(f"  [BEST] {primary_scene} mIoU = {best_miou*100:.2f}%  (saved)\n")

        if (epoch + 1) % SAVE_EVERY == 0:
            ckpt = os.path.join(out_dir, f"checkpoint_ep{epoch+1:03d}.pth")
            torch.save({"epoch": epoch+1, "model_state": model.state_dict(),
                        "history": history, "best_miou": best_miou}, ckpt)
            print(f"  [OK] Checkpoint -> {ckpt}\n")

    # Final eval with TTA
    print(f"\n[E] Final full eval (TTA=on) ...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    pri_pc, pri_miou = evaluate(model, val_pri, f"Final {primary_scene}", None, True)
    crs_pc, crs_miou = evaluate(model, val_crs, f"Final {cross_scene}",   None, True)

    print(f"\n  {'Class':<15} {primary_scene+' IoU':>12} {cross_scene+' IoU':>12}")
    print("  " + "-"*43)
    for c in range(1, NUM_CLASSES):
        p = pri_pc.get(c, float("nan")); x = crs_pc.get(c, float("nan"))
        print(f"  {CLASS_NAMES[c]:<15} {p*100:>11.2f}%  {x*100:>11.2f}%")
    print("  " + "-"*43)
    print(f"  {'mIoU':<15} {pri_miou*100:>11.2f}%  {crs_miou*100:>11.2f}%")
    print(f"  Domain gap: {abs(pri_miou-crs_miou)*100:.2f}%")

    results = {
        "model": "Prithvi-100M + FPN", "train_scene": train_scene,
        "num_epochs": NUM_EPOCHS, "best_primary_miou": float(best_miou),
        "final": {
            primary_scene.lower(): {"miou": float(pri_miou),
                                    "per_class": {CLASS_NAMES[c]: float(v)
                                                  for c,v in pri_pc.items()}},
            cross_scene.lower():   {"miou": float(crs_miou),
                                    "per_class": {CLASS_NAMES[c]: float(v)
                                                  for c,v in crs_pc.items()}},
        },
        "history": history,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    plot_curves(history, out_dir, f"Prithvi-100M | Trained on {train_scene}")
    return pri_miou, crs_miou


# ============================================================
# MAIN
# ============================================================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 64)
    print("  Prithvi-100M Fine-tune  --  LoveDA Urban & Rural")
    print(f"  {NUM_EPOCHS} epochs each  |  Target: 65-72% mIoU")
    print("=" * 64)

    check_deps()
    gpu = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"
    print(f"\n  Device: {DEVICE} ({gpu})\n")

    t_start = time.time()

    # Model 1: Urban specialist
    u_pri, u_crs = train_model(
        model_name    = "Urban",
        train_scene   = "Urban",
        primary_scene = "Urban",
        cross_scene   = "Rural",
        out_dir       = os.path.join(BASE_OUT, "11_prithvi_urban"),
    )

    # Model 2: Rural specialist
    r_pri, r_crs = train_model(
        model_name    = "Rural",
        train_scene   = "Rural",
        primary_scene = "Rural",
        cross_scene   = "Urban",
        out_dir       = os.path.join(BASE_OUT, "11_prithvi_rural"),
    )

    total_hrs = (time.time() - t_start) / 3600

    print("\n" + "=" * 70)
    print("  PRITHVI FINAL RESULTS")
    print("=" * 70)
    print(f"  {'Model':<20} {'Urban mIoU':>12} {'Rural mIoU':>12}")
    print("  " + "-"*46)
    print(f"  {'Urban Specialist':<20} {u_pri*100:>11.2f}%  {u_crs*100:>11.2f}%")
    print(f"  {'Rural Specialist':<20} {r_crs*100:>11.2f}%  {r_pri*100:>11.2f}%")
    print("=" * 70)
    print(f"\n  Total time: {total_hrs:.1f} hours")
    print("  Best model weights:")
    print("    Urban -> ./outputs/11_prithvi_urban/best_model.pth")
    print("    Rural -> ./outputs/11_prithvi_rural/best_model.pth")
    print("\n  Done!\n")


if __name__ == "__main__":
    main()
