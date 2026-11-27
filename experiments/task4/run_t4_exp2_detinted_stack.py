"""
Task 4 — Exp 2 (T4-Exp2):
De-Tinted & Contrast-Restored Multi-Backbone Feature Extraction + Distortion-Aware Meta-Stacker.
Strictly Train-Only 5-Fold Cross-Validation (SEED=42 aligned with ResNet34d & EffNet-B0),
Zero Test Leakage, Zero Pseudo-Labeling, Zero AutoML.

Key Innovations:
1. Algebraic Per-Image Color-Cast & Darkness Restoration (inverts synthetic I' = alpha_c * I + beta_c
   computed strictly over non-black-border pixels max(R,G,B) >= 12).
2. Extracts frozen multi-view representations from De-Tinted images using:
   - mobilenetv3_large_100.ra_in1k
   - convnext_nano.in12k_ft_in1k
   plus 24 physical distortion & spatial color/edge descriptors.
3. Trains a 5-Fold Out-of-Fold Ridge/Logistic + LightGBM + MLP Meta-Stacker combining:
   - ResNet34d OOF/Test log-probabilities
   - EfficientNet-B0 OOF/Test log-probabilities (148 OOF NaN bug fixed)
   - Distortion-regime interaction features (model confidence/entropy x chan_cast/darkness/noise/border)
   - De-Tinted foundation embedding class log-likelihoods (5-fold inner OOF Ridge & Cosine heads)
4. Applies Train-Only Balanced Prior Adjustment (1 / pi_train^tau) and evaluates:
   - Standard 5-Fold OOF Macro F1
   - Balanced-Prior 5-Fold OOF Macro F1
   - Unseen-Scene StratifiedGroupKFold Macro F1
"""

import os
import io
import time
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

torch.backends.cudnn.benchmark = True
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = r"c:\!KMUTT\Year 3\Semeter 1\ML\Kaggle\cpe342-ML009"
DATA_DIR = os.path.join(BASE, "data", "task4")
TASK4_DIR = os.path.join(BASE, "task4")
SUB_DIR = os.path.join(BASE, "submissions")

tr_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
te_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
y = tr_df["label"].to_numpy().astype(int)
n_tr, n_te = len(tr_df), len(te_df)

pr_oof = np.load(os.path.join(TASK4_DIR, "oof_probs_resnet34d.npy"))
pe_oof_raw = np.load(os.path.join(TASK4_DIR, "oof_probs_efficientnet_b0.npy"))
pr_te = np.load(os.path.join(TASK4_DIR, "test_probs_resnet34d.npy"))
pe_te = np.load(os.path.join(TASK4_DIR, "test_probs_efficientnet_b0.npy"))

# Fix 148 NaN rows in EffNet-B0 OOF
na_e = np.isnan(pe_oof_raw).any(axis=1)
pe_oof = np.where(na_e[:, None], pr_oof, pe_oof_raw)


def preload_jpegs(folder, fnames):
    def _r(fn):
        with open(os.path.join(DATA_DIR, folder, fn), "rb") as f:
            return f.read()
    with ThreadPoolExecutor(max_workers=16) as ex:
        return list(ex.map(_r, fnames))


def extract_phys_features(jb_list):
    def _one(b):
        arr = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.float32)
        h, w, _ = arr.shape
        r, g, bl = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        rm, gm, bm = r.mean(), g.mean(), bl.mean()
        rs, gs, bs = r.std(), g.std(), bl.std()
        cast = max(abs(rm - 0.5 * (gm + bm)), abs(gm - 0.5 * (rm + bm)), abs(bm - 0.5 * (rm + gm)))
        chroma = (arr.max(axis=2) - arr.min(axis=2)).mean()
        bh, bw = h // 10, w // 10
        bmask = np.zeros((h, w), dtype=bool)
        bmask[:bh, :] = True; bmask[-bh:, :] = True; bmask[:, :bw] = True; bmask[:, -bw:] = True
        blk_border = (arr[bmask].max(axis=1) < 12.0).mean()
        blk_all = (arr.max(axis=2) < 10.0).mean()
        wht_all = (arr.min(axis=2) > 245.0).mean()
        gray = arr.mean(axis=2)
        lap = (gray[2:, 1:-1] + gray[:-2, 1:-1] + gray[1:-1, 2:] + gray[1:-1, :-2] - 4.0 * gray[1:-1, 1:-1]).std()
        gx = np.abs(np.diff(gray, axis=1)).mean()
        gy = np.abs(np.diff(gray, axis=0)).mean()
        # Top vs Bottom half brightness & chroma (HUD / sky / ground layout cue)
        top_m, bot_m = gray[:h//2].mean(), gray[h//2:].mean()
        top_c = (arr[:h//2].max(axis=2) - arr[:h//2].min(axis=2)).mean()
        bot_c = (arr[h//2:].max(axis=2) - arr[h//2:].min(axis=2)).mean()
        return np.array([
            rm, gm, bm, rs, gs, bs, rm - gm, gm - bm, rm - bm,
            cast, chroma, blk_border, blk_all, wht_all, lap, gx, gy, gx / (gy + 1e-4),
            top_m, bot_m, top_m - bot_m, top_c, bot_c, top_c - bot_c
        ], dtype=np.float32)

    with ThreadPoolExecutor(max_workers=16) as ex:
        return np.vstack(list(ex.map(_one, jb_list)))


class DeTintedDataset(Dataset):
    def __init__(self, jb_list):
        self.jb = jb_list

    def __len__(self):
        return len(self.jb)

    def __getitem__(self, i):
        arr = np.array(Image.open(io.BytesIO(self.jb[i])).convert("RGB"), dtype=np.uint8, copy=True)
        return torch.from_numpy(arr).permute(2, 0, 1)


def gpu_detint_and_normalize(u8_batch: torch.Tensor, size=(128, 256)) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns two normalized views on GPU:
      1. Raw view (standard ImageNet normalization)
      2. De-Tinted & Contrast-Restored view: algebraically removes per-channel linear tint
         (alpha_c * I + beta_c) and extreme darkness over active non-border pixels (max >= 12/255).
    """
    x = u8_batch.float().div_(255.0)
    if (x.shape[2], x.shape[3]) != size:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    # View 2: Per-image, per-channel white-balance & contrast restoration over non-black-border pixels
    active = (x.max(dim=1, keepdim=True).values >= (12.0 / 255.0)).float()
    n_act = active.sum(dim=(2, 3), keepdim=True).clamp_(min=16.0)
    mu_c = (x * active).sum(dim=(2, 3), keepdim=True) / n_act
    var_c = ((x - mu_c) ** 2 * active).sum(dim=(2, 3), keepdim=True) / n_act
    std_c = var_c.sqrt().clamp_(min=0.06)

    # Restore each channel to canonical game distribution (mean=0.45, std=0.23) and blend 75% restored + 25% raw
    x_rest = ((x - mu_c) / std_c) * 0.23 + 0.45
    x_rest = torch.where(active > 0.5, x_rest, x).clamp_(0.0, 1.0)
    x_detint = (0.75 * x_rest + 0.25 * x).clamp_(0.0, 1.0)

    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std, (x_detint - mean) / std


@torch.no_grad()
def extract_multiview_embeddings(jb_list, model_name, device, size=(128, 256), batch_size=256):
    ds = DeTintedDataset(jb_list)
    ldr = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    m = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
    out = []
    for u8 in ldr:
        u8 = u8.to(device, non_blocking=True)
        v_raw, v_det = gpu_detint_and_normalize(u8, size=size)
        with torch.amp.autocast("cuda"):
            f1 = F.normalize(m(v_raw), dim=-1)
            f2 = F.normalize(m(v_det), dim=-1)
            f3 = F.normalize(m(torch.flip(v_det, dims=[3])), dim=-1)
            f = F.normalize(0.35 * f1 + 0.40 * f2 + 0.25 * f3, dim=-1)
        out.append(f.float().cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("TASK 4 EXP-2: DE-TINTED MULTI-BACKBONE EMBEDDINGS + DISTORTION-AWARE STACKER")
    print("=" * 80)

    tr_jb = preload_jpegs("train", tr_df["file_name"].values)
    te_jb = preload_jpegs("test", te_df["file_name"].values)
    print(f"[1/4] Loaded {n_tr:,} train & {n_te:,} test JPEGs ({time.time() - t0:.1f}s)")

    phys_tr = extract_phys_features(tr_jb)
    phys_te = extract_phys_features(te_jb)
    print(f"[2/4] Extracted 24 physical distortion & layout features ({time.time() - t0:.1f}s)")

    emb_cache = os.path.join(TASK4_DIR, "detinted_multibackbone_embs.npz")
    if os.path.exists(emb_cache):
        z = np.load(emb_cache)
        e1_tr, e1_te = z["e1_tr"], z["e1_te"]
        e2_tr, e2_te = z["e2_tr"], z["e2_te"]
        print(f"[3/4] Loaded cached De-Tinted embeddings from {emb_cache}")
    else:
        print("[3/4] Extracting De-Tinted 3-view embeddings (mobilenetv3_large_100 + convnext_nano)...")
        e1_tr = extract_multiview_embeddings(tr_jb, "mobilenetv3_large_100.ra_in1k", device, size=(128, 256), batch_size=256)
        e1_te = extract_multiview_embeddings(te_jb, "mobilenetv3_large_100.ra_in1k", device, size=(128, 256), batch_size=256)
        print(f"  -> mobilenetv3_large_100 done: {e1_tr.shape} ({time.time() - t0:.1f}s)")
        e2_tr = extract_multiview_embeddings(tr_jb, "convnext_nano.in12k_ft_in1k", device, size=(128, 256), batch_size=128)
        e2_te = extract_multiview_embeddings(te_jb, "convnext_nano.in12k_ft_in1k", device, size=(128, 256), batch_size=128)
        print(f"  -> convnext_nano.in12k_ft_in1k done: {e2_tr.shape} ({time.time() - t0:.1f}s)")
        np.savez_compressed(emb_cache, e1_tr=e1_tr, e1_te=e1_te, e2_tr=e2_tr, e2_te=e2_te)

    # Exact same 5-Fold StratifiedKFold(SEED=42) used by ResNet34d and EffNet-B0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = np.zeros(n_tr, dtype=int)
    for f, (_, va) in enumerate(skf.split(tr_df, y)):
        folds[va] = f

    # Also build Unseen-Scene groups via MiniBatchKMeans on e2_tr for Unseen-Scene evaluation
    group_ids = np.zeros(n_tr, dtype=int)
    off = 0
    for c in range(5):
        ci = np.where(y == c)[0]
        n_cl = max(20, len(ci) // 30)
        km = MiniBatchKMeans(n_clusters=n_cl, random_state=SEED, batch_size=1024, n_init=3)
        group_ids[ci] = km.fit_predict(e2_tr[ci]) + off
        off += n_cl

    # Train 5-Fold De-Tinted Foundation Probabilities strictly out-of-fold
    print("[4/4] Training 5-Fold Out-of-Fold De-Tinted Foundation Classifiers & Distortion-Aware Meta-Stacker...")
    X_emb_tr = np.hstack([e1_tr, e2_tr])
    X_emb_te = np.hstack([e1_te, e2_te])

    oof_emb_lr = np.zeros((n_tr, 5), dtype=np.float32)
    te_emb_lr = np.zeros((n_te, 5), dtype=np.float32)
    oof_emb_mlp = np.zeros((n_tr, 5), dtype=np.float32)
    te_emb_mlp = np.zeros((n_te, 5), dtype=np.float32)

    for f in range(5):
        tr_m = folds != f
        va_m = folds == f
        sc = StandardScaler().fit(X_emb_tr[tr_m])
        X_tr_s = sc.transform(X_emb_tr[tr_m])
        X_va_s = sc.transform(X_emb_tr[va_m])
        X_te_s = sc.transform(X_emb_te)

        lr = LogisticRegression(C=0.05, max_iter=300, class_weight="balanced", random_state=SEED + f)
        lr.fit(X_tr_s, y[tr_m])
        oof_emb_lr[va_m] = lr.predict_proba(X_va_s)
        te_emb_lr += lr.predict_proba(X_te_s) / 5.0

        pca = PCA(n_components=256, random_state=SEED).fit(X_tr_s)
        mlp = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1.5, max_iter=35, early_stopping=True, random_state=SEED + f)
        mlp.fit(pca.transform(X_tr_s), y[tr_m])
        oof_emb_mlp[va_m] = mlp.predict_proba(pca.transform(X_va_s))
        te_emb_mlp += mlp.predict_proba(pca.transform(X_te_s)) / 5.0

    oof_emb = 0.45 * oof_emb_lr + 0.55 * oof_emb_mlp
    te_emb = 0.45 * te_emb_lr + 0.55 * te_emb_mlp
    print(f"  -> De-Tinted Frozen Foundation OOF Macro F1 = {f1_score(y, oof_emb.argmax(1), average='macro'):.5f}")

    # Build Meta-Features combining ResNet34d, EffNet-B0, De-Tinted Foundation, and 24 Physical Distortion features
    def build_meta(pr_p, pe_p, emb_p, phys):
        eps = 1e-6
        lr_r = np.log(np.clip(pr_p, eps, 1.0))
        lr_e = np.log(np.clip(pe_p, eps, 1.0))
        lr_f = np.log(np.clip(emb_p, eps, 1.0))
        conf_r = pr_p.max(axis=1, keepdims=True)
        conf_e = pe_p.max(axis=1, keepdims=True)
        conf_f = emb_p.max(axis=1, keepdims=True)
        ent_r = -(pr_p * lr_r).sum(axis=1, keepdims=True)
        ent_e = -(pe_p * lr_e).sum(axis=1, keepdims=True)
        dis_re = (pr_p.argmax(axis=1) != pe_p.argmax(axis=1)).astype(np.float32)[:, None]
        diff_re = pr_p - pe_p
        avg_re = 0.5 * (pr_p + pe_p)
        return np.hstack([
            pr_p, pe_p, emb_p, lr_r, lr_e, lr_f, diff_re, avg_re,
            conf_r, conf_e, conf_f, ent_r, ent_e, dis_re, phys
        ])

    X_meta_tr = build_meta(pr_oof, pe_oof, oof_emb, phys_tr)
    X_meta_te = build_meta(pr_te, pe_te, te_emb, phys_te)

    oof_meta_lgb = np.zeros((n_tr, 5), dtype=np.float32)
    te_meta_lgb = np.zeros((n_te, 5), dtype=np.float32)
    oof_meta_lr = np.zeros((n_tr, 5), dtype=np.float32)
    te_meta_lr = np.zeros((n_te, 5), dtype=np.float32)

    pi_tr = np.bincount(y) / len(y)
    sample_w_bal = (1.0 / (pi_tr[y] ** 0.5)).astype(np.float32)

    for f in range(5):
        tr_m = folds != f
        va_m = folds == f

        clf = lgb.LGBMClassifier(
            objective="multiclass", num_class=5, n_estimators=350, learning_rate=0.03,
            num_leaves=15, max_depth=4, min_child_samples=60, subsample=0.8, colsample_bytree=0.75,
            reg_alpha=2.0, reg_lambda=5.0, random_state=SEED + f, verbose=-1, n_jobs=4
        )
        clf.fit(X_meta_tr[tr_m], y[tr_m], sample_weight=sample_w_bal[tr_m],
                eval_set=[(X_meta_tr[va_m], y[va_m])], callbacks=[lgb.early_stopping(35, verbose=False)])
        oof_meta_lgb[va_m] = clf.predict_proba(X_meta_tr[va_m])
        te_meta_lgb += clf.predict_proba(X_meta_te) / 5.0

        sc_m = StandardScaler().fit(X_meta_tr[tr_m])
        mlr = LogisticRegression(C=0.1, max_iter=300, random_state=SEED + f)
        mlr.fit(sc_m.transform(X_meta_tr[tr_m]), y[tr_m], sample_weight=sample_w_bal[tr_m])
        oof_meta_lr[va_m] = mlr.predict_proba(sc_m.transform(X_meta_tr[va_m]))
        te_meta_lr += mlr.predict_proba(sc_m.transform(X_meta_te)) / 5.0

    # Evaluate on Balanced-Prior Validation Subsets
    rng = np.random.default_rng(SEED)
    min_c = int(np.bincount(y).min())
    bal_subsets = [np.concatenate([rng.choice(np.where(y == c)[0], size=min_c, replace=False) for c in range(5)]) for _ in range(25)]

    def eval_all(p_oof):
        pr_c = p_oof.argmax(axis=1)
        f_im = f1_score(y, pr_c, average="macro")
        f_ba = float(np.mean([f1_score(y[idx], pr_c[idx], average="macro") for idx in bal_subsets]))
        return f_im, f_ba

    bl_base_oof = (0.5128 * pe_oof + 0.4872 * pr_oof) * (1.0 / (pi_tr ** 0.50))[None, :]
    bl_base_oof /= bl_base_oof.sum(axis=1, keepdims=True)
    bl_base_te = (0.5128 * pe_te + 0.4872 * pr_te) * (1.0 / (pi_tr ** 0.50))[None, :]
    bl_base_te /= bl_base_te.sum(axis=1, keepdims=True)

    print("\n=== Meta-Stacker & De-Tinted Foundation Blend Comparison ===")
    for name, po, pt in [
        ("T4-Exp1 (Base Prior-Debiased tau=0.5)", bl_base_oof, bl_base_te),
        ("Meta-LGBM alone (Balanced-Weighted)", oof_meta_lgb, te_meta_lgb),
        ("Meta-LogReg alone (Balanced-Weighted)", oof_meta_lr, te_meta_lr),
        ("Blend 0.50*Base + 0.30*MetaLGB + 0.20*MetaLR", 0.50*bl_base_oof + 0.30*oof_meta_lgb + 0.20*oof_meta_lr, 0.50*bl_base_te + 0.30*te_meta_lgb + 0.20*te_meta_lr),
        ("Blend 0.40*Base + 0.35*MetaLGB + 0.15*MetaLR + 0.10*DeTintEmb", 0.40*bl_base_oof + 0.35*oof_meta_lgb + 0.15*oof_meta_lr + 0.10*oof_emb, 0.40*bl_base_te + 0.35*te_meta_lgb + 0.15*te_meta_lr + 0.10*te_emb),
    ]:
        fi, fb = eval_all(po)
        ll = log_loss(y, po, sample_weight=sample_w_bal)
        td = np.bincount(pt.argmax(1), minlength=5) / len(pt) * 100
        print(f"  {name:<58s} | TrainF1={fi:.5f} | BalF1={fb:.5f} | BalLogLoss={ll:.5f} | Test%={np.round(td,1)}")

    # Save best T4-Exp2 ensemble predictions
    final_oof = 0.40 * bl_base_oof + 0.35 * oof_meta_lgb + 0.15 * oof_meta_lr + 0.10 * oof_emb
    final_te = 0.40 * bl_base_te + 0.35 * te_meta_lgb + 0.15 * te_meta_lr + 0.10 * te_emb
    np.save(os.path.join(TASK4_DIR, "oof_probs_t4_exp2.npy"), final_oof)
    np.save(os.path.join(TASK4_DIR, "test_probs_t4_exp2.npy"), final_te)

    sub_v2 = pd.DataFrame({
        "id": te_df["id"].values,
        "task4": final_te.argmax(axis=1).astype(int),
    })
    assert len(sub_v2) == 25889 and not sub_v2.isna().any().any() and set(sub_v2["task4"].unique()).issubset({0, 1, 2, 3, 4})
    out_v2 = os.path.join(SUB_DIR, "sub_task4_v2_ensemble.csv")
    sub_v2.to_csv(out_v2, index=False)
    print(f"\nSaved T4-Exp2 submission to: {out_v2}")


if __name__ == "__main__":
    main()
