"""
Task 4 — Exp 3 (T4-Exp3):
5-Fold End-to-End Fine-Tuned Domain-Invariant Architecture #3 (mobilenetv3_large_100.ra_in1k)
with Per-Image Algebraic De-Tinting, Heavy Domain Augmentation, FP32 Softmax (Zero NaNs),
and 3-View De-Tinted + Border-Cropped TTA (Zero Upside-Down Vertical Flip TTA Bug).
"""

import os
import io
import time
import random
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

SEED = 42
N_SPLITS = 5
NUM_CLASSES = 5
IMG_HEIGHT = 128
IMG_WIDTH = 256
BATCH_SIZE = 128
EPOCHS = 7
LR_BACKBONE = 4.5e-4
LR_HEAD = 1.5e-3
WEIGHT_DECAY = 1e-2
MODEL_NAME = "mobilenetv3_large_100.ra_in1k"

BASE_DIR = r"c:\!KMUTT\Year 3\Semeter 1\ML\Kaggle\cpe342-ML009"
DATA_DIR = os.path.join(BASE_DIR, "data", "task4")
TASK4_DIR = os.path.join(BASE_DIR, "task4")
SUB_DIR = os.path.join(BASE_DIR, "submissions")


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def preload_jpeg_bytes(img_dir: str, file_names) -> list[bytes]:
    def _read_one(fname):
        with open(os.path.join(img_dir, fname), "rb") as f:
            return f.read()
    with ThreadPoolExecutor(max_workers=16) as ex:
        return list(ex.map(_read_one, file_names))


class InMemoryJPEGDataset(Dataset):
    def __init__(self, jpeg_bytes_list: list[bytes], labels=None):
        self.jpeg_bytes = jpeg_bytes_list
        self.labels = labels

    def __len__(self):
        return len(self.jpeg_bytes)

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.jpeg_bytes[idx])).convert("RGB")
        arr = np.array(img, dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        if self.labels is None:
            return tensor
        return tensor, self.labels[idx]


def detint_batch_gpu(x: torch.Tensor, blend_w: float = 0.70) -> torch.Tensor:
    """
    Per-image, per-channel non-border color-cast & darkness neutralization on GPU.
    Inverts linear channel shifts I'_c = alpha_c * I_c + beta_c over active pixels (max >= 12/255).
    """
    active = (x.max(dim=1, keepdim=True).values >= (12.0 / 255.0)).float()
    n_act = active.sum(dim=(2, 3), keepdim=True).clamp_(min=16.0)
    mu_c = (x * active).sum(dim=(2, 3), keepdim=True) / n_act
    var_c = ((x - mu_c) ** 2 * active).sum(dim=(2, 3), keepdim=True) / n_act
    std_c = var_c.sqrt().clamp_(min=0.06)

    x_rest = ((x - mu_c) / std_c) * 0.23 + 0.45
    x_rest = torch.where(active > 0.5, x_rest, x).clamp_(0.0, 1.0)
    return ((1.0 - blend_w) * x + blend_w * x_rest).clamp_(0.0, 1.0)


def gpu_preprocess_and_augment(
    imgs_uint8: torch.Tensor,
    is_train: bool,
    detint_w: float = 0.70,
    center_crop_ratio: float = 1.0,
    img_h: int = IMG_HEIGHT,
    img_w: int = IMG_WIDTH,
) -> torch.Tensor:
    x = imgs_uint8.float().div_(255.0)
    if center_crop_ratio < 0.999:
        H, W = x.shape[2], x.shape[3]
        ch = int(H * (1.0 - center_crop_ratio) * 0.5)
        cw = int(W * (1.0 - center_crop_ratio) * 0.5)
        x = x[:, :, ch:H - ch, cw:W - cw]

    if (x.shape[2], x.shape[3]) != (img_h, img_w):
        x = F.interpolate(x, size=(img_h, img_w), mode="bilinear", align_corners=False)

    if is_train:
        B = x.size(0)
        device = x.device
        # Apply strong domain-matched synthetic RGB tint & darkness BEFORE de-tinting so network learns residual invariance
        alpha = 1.0 + (torch.rand(B, 3, 1, 1, device=device) - 0.5) * 0.85  # [0.575, 1.425] heavy tint
        beta = (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.36         # [-0.18, 0.18] darkness/brightness
        x = (x * alpha + beta).clamp_(0.0, 1.0)

        # Now apply de-tinting with randomized blend weight in [0.55, 0.85]
        x = detint_batch_gpu(x, blend_w=0.70)

        # Horizontal flip (p=0.5) & vertical flip (p=0.18)
        h_mask = torch.rand(B, 1, 1, 1, device=device) < 0.5
        x = torch.where(h_mask, torch.flip(x, dims=[3]), x)
        v_mask = torch.rand(B, 1, 1, 1, device=device) < 0.18
        x = torch.where(v_mask, torch.flip(x, dims=[2]), x)

        # Random grayscale (p=0.12) & Gaussian noise (p=0.35)
        gray_mask = torch.rand(B, 1, 1, 1, device=device) < 0.12
        x_gray = x.mean(dim=1, keepdim=True).expand_as(x)
        x = torch.where(gray_mask, x_gray, x)

        noise_mask = (torch.rand(B, 1, 1, 1, device=device) < 0.35).float()
        x = (x + noise_mask * torch.randn_like(x) * 0.045).clamp_(0.0, 1.0)

        # Random Cutout
        if random.random() < 0.5:
            eh = random.randint(12, 30)
            ew = random.randint(20, 52)
            ey = random.randint(0, img_h - eh)
            ex = random.randint(0, img_w - ew)
            erase_mask = torch.rand(B, 1, 1, 1, device=device) < 0.25
            x[:, :, ey:ey + eh, ex:ex + ew] = torch.where(
                erase_mask, torch.zeros_like(x[:, :, ey:ey + eh, ex:ex + ew]), x[:, :, ey:ey + eh, ex:ex + ew]
            )
    else:
        x = detint_batch_gpu(x, blend_w=detint_w)

    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def apply_mixup(imgs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(imgs.size(0), device=imgs.device)
    return lam * imgs + (1.0 - lam) * imgs[perm], labels, labels[perm], lam


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, use_mixup: bool = True):
    model.train()
    running_loss = 0.0
    for imgs_u8, labels in loader:
        imgs_u8 = imgs_u8.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        imgs = gpu_preprocess_and_augment(imgs_u8, is_train=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            if use_mixup and random.random() < 0.35:
                imgs, y_a, y_b, lam = apply_mixup(imgs, labels, alpha=0.2)
                logits = model(imgs)
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def predict_probs(model, loader, device, use_tta: bool = False):
    model.eval()
    all_probs = []
    for batch in loader:
        imgs_u8 = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs_u8 = imgs_u8.to(device, non_blocking=True)
        v1 = gpu_preprocess_and_augment(imgs_u8, is_train=False, detint_w=0.70, center_crop_ratio=1.0)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            # Always cast logits to float32 before softmax to prevent FP16 NaN overflow!
            p1 = torch.softmax(model(v1).float(), dim=-1)
            if use_tta:
                p2 = torch.softmax(model(torch.flip(v1, dims=[3])).float(), dim=-1)
                v3 = gpu_preprocess_and_augment(imgs_u8, is_train=False, detint_w=0.85, center_crop_ratio=0.90)
                p3 = torch.softmax(model(v3).float(), dim=-1)
                probs = 0.45 * p1 + 0.30 * p2 + 0.25 * p3
            else:
                probs = p1
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def main():
    seed_everything(SEED + 202)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"TASK 4 EXP-3: 5-FOLD DE-TINTED {MODEL_NAME} TRAINING")
    print("=" * 80)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    y_all = train_df["label"].to_numpy().astype(np.int64)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = np.zeros(len(train_df), dtype=int)
    for f, (_, va_idx) in enumerate(skf.split(train_df, y_all)):
        folds[va_idx] = f

    t0 = time.time()
    train_jpegs = preload_jpeg_bytes(os.path.join(DATA_DIR, "train"), train_df["file_name"].values)
    test_jpegs = preload_jpeg_bytes(os.path.join(DATA_DIR, "test"), test_df["file_name"].values)
    print(f"Loaded JPEGs in {time.time() - t0:.1f}s")

    class_counts = np.bincount(y_all)
    class_weights = (len(train_df) / (NUM_CLASSES * class_counts)).astype(np.float32)
    soft_class_weights = np.sqrt(class_weights)
    soft_class_weights /= soft_class_weights.mean()

    test_ds = InMemoryJPEGDataset(test_jpegs, labels=None)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True)

    oof_probs = np.zeros((len(train_df), NUM_CLASSES), dtype=np.float32)
    test_probs = np.zeros((len(test_df), NUM_CLASSES), dtype=np.float32)

    for fold in range(N_SPLITS):
        f_start = time.time()
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]

        tr_loader = DataLoader(
            InMemoryJPEGDataset([train_jpegs[i] for i in tr_idx], y_all[tr_idx]),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True
        )
        va_loader = DataLoader(
            InMemoryJPEGDataset([train_jpegs[i] for i in va_idx], y_all[va_idx]),
            batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True
        )

        model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES, drop_rate=0.2).to(device)
        weight_tensor = torch.tensor(soft_class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.05)

        head_params = [p for n, p in model.named_parameters() if "classifier" in n or "fc" in n or "head" in n]
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ], weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[LR_BACKBONE, LR_HEAD],
            steps_per_epoch=len(tr_loader), epochs=EPOCHS, pct_start=0.15, anneal_strategy="cos"
        )
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        best_val_f1 = -1.0
        best_state = None
        for ep in range(1, EPOCHS + 1):
            use_mix = ep <= (EPOCHS - 2)
            loss = train_one_epoch(model, tr_loader, criterion, optimizer, scheduler, scaler, device, use_mixup=use_mix)
            val_p = predict_probs(model, va_loader, device, use_tta=False)
            val_f1 = f1_score(y_all[va_idx], val_p.argmax(axis=1), average="macro")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Fold {fold+1} Ep {ep:02d}/{EPOCHS:02d} | Loss={loss:.4f} | Val F1={val_f1:.4f}")

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        oof_fold = predict_probs(model, va_loader, device, use_tta=True)
        oof_probs[va_idx] = oof_fold
        tta_f1 = f1_score(y_all[va_idx], oof_fold.argmax(axis=1), average="macro")
        print(f"-> Fold {fold+1} Best Val F1={best_val_f1:.4f} | 3-View De-Tinted TTA F1={tta_f1:.4f} ({time.time()-f_start:.1f}s)")

        test_probs += predict_probs(model, test_loader, device, use_tta=True) / N_SPLITS
        del model, optimizer, scheduler, scaler, best_state
        torch.cuda.empty_cache()

    assert not np.isnan(oof_probs).any(), "NaNs in OOF!"
    assert not np.isnan(test_probs).any(), "NaNs in Test!"
    np.save(os.path.join(TASK4_DIR, "oof_probs_mobilenetv3_detinted.npy"), oof_probs)
    np.save(os.path.join(TASK4_DIR, "test_probs_mobilenetv3_detinted.npy"), test_probs)
    print(f"\n>>> {MODEL_NAME} 5-Fold OOF Macro F1 = {f1_score(y_all, oof_probs.argmax(1), average='macro'):.5f} <<<")


if __name__ == "__main__":
    main()
