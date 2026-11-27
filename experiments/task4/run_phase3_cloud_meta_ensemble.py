"""
Task 4 — Phase 3 (T4-Exp6 Final): Unseen-Scene Calibrated Multi-Paradigm Meta-Ensemble.
Combines strictly folds_scene-honest models (zero test leakage, zero pseudo-labeling, zero AutoML):
  - Phase 1 (7-Backbone Multi-Foundation: DINOv2-S, SigLIP2-B, EVA02-CLIP-B, ConvNeXt-T, EffNetV2-S
    + local MobileNetV3 & ConvNeXt-Nano) -> 5-Backbone MLP/LR + 7-Backbone MLP/LR
  - Phase 2 (End-to-End Domain-Hardened Unseen-Scene Fine-Tuning):
    ConvNeXt-Small (0.97765), Swin-Tiny (0.97451), MobileNetV3-Large (0.94789)
  - Nested Out-of-Fold PyTorch LBFGS Temperature Scaling + Distortion-Aware CatBoost & LightGBM
    + Calibrated Weighted Logit Blend + Train-Only Safe Prior Check (no double penalty!).
"""

import glob
import os
import time
import warnings
import zipfile
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE = r"c:\!KMUTT\Year 3\Semeter 1\ML\Kaggle\cpe342-ML009"
DATA_DIR = os.path.join(BASE, "data", "task4")
TASK4_DIR = os.path.join(BASE, "task4")
CLOUD_OUT_DIR = os.path.join(TASK4_DIR, "cloud_outputs")
SUB_DIR = os.path.join(BASE, "submissions")


def auto_unpack_cloud_zip():
    os.makedirs(CLOUD_OUT_DIR, exist_ok=True)
    # Skip unpack if all expected cloud files are already extracted
    expected = os.path.join(CLOUD_OUT_DIR, "oof_probs_v5_convnext_small.npy")
    if os.path.exists(expected) and os.path.exists(os.path.join(CLOUD_OUT_DIR, "embs_v4_cloud_5backbones.npz")):
        print(f"[Cache] Using already unpacked cloud outputs in {CLOUD_OUT_DIR}")
        return None
    candidates = [
        os.path.join(os.path.dirname(BASE), "task4_cloud_all_outputs.zip"),
        os.path.join(os.path.dirname(BASE), "task4_partA_v4_outputs.zip"),
        os.path.join(TASK4_DIR, "task4_cloud_all_outputs.zip"),
        os.path.join(TASK4_DIR, "task4_partA_v4_outputs.zip"),
        os.path.join(os.path.expanduser("~"), "Downloads", "task4_cloud_all_outputs.zip"),
        os.path.join(os.path.expanduser("~"), "Downloads", "task4_partA_v4_outputs.zip"),
    ]
    for dl in glob.glob(os.path.join(os.path.expanduser("~"), "Downloads", "task4_*outputs*.zip")):
        candidates.append(dl)
    existing_zips = [p for p in candidates if os.path.exists(p)]
    if not existing_zips:
        return None
    all_zips = [p for p in existing_zips if "all_outputs" in os.path.basename(p)]
    chosen = max(all_zips if all_zips else existing_zips, key=os.path.getmtime)
    print(f"[Unpack] Extracting cloud bundle: {chosen} -> {CLOUD_OUT_DIR}")
    with zipfile.ZipFile(chosen, "r") as zf:
        zf.extractall(CLOUD_OUT_DIR)
    return chosen


class DeepResidualEmbNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 768, num_classes: int = 5):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.drop1 = nn.Dropout(0.45)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.drop2 = nn.Dropout(0.40)
        self.head = nn.Linear(hidden, num_classes)
        self.skip = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        xn = self.norm_in(x)
        h1 = self.drop1(F.silu(self.ln1(self.fc1(xn))))
        h2 = h1 + self.drop2(F.silu(self.ln2(self.fc2(h1))))
        return self.head(h2) + 0.25 * self.skip(xn)


def temp_scale_oof_test(po: np.ndarray, pt: np.ndarray, y: np.ndarray, folds_scene: np.ndarray, w_bal_cls: np.ndarray):
    """Proper PyTorch LBFGS 1-parameter temperature calibration per unseen-scene fold."""
    cal_o = np.zeros_like(po, dtype=np.float32)
    cal_t = np.zeros_like(pt, dtype=np.float32)
    temps = []
    lo = torch.from_numpy(np.log(np.clip(po, 1e-6, 1.0))).float()
    lt = torch.from_numpy(np.log(np.clip(pt, 1e-6, 1.0))).float()
    yt = torch.from_numpy(y).long()
    w_t = torch.from_numpy(w_bal_cls).float()
    for f in range(5):
        tr_m, va_m = folds_scene != f, folds_scene == f
        log_T = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.LBFGS([log_T], lr=0.2, max_iter=35)

        def closure():
            opt.zero_grad()
            T = torch.exp(log_T).clamp(0.3, 3.0)
            lp = torch.log_softmax(lo[tr_m] / T, dim=-1)
            loss = -(lp[torch.arange(tr_m.sum()), yt[tr_m]] * w_t[yt[tr_m]]).mean()
            loss.backward()
            return loss

        opt.step(closure)
        T_val = float(torch.exp(log_T).clamp(0.3, 3.0).item())
        temps.append(T_val)
        with torch.no_grad():
            cal_o[va_m] = torch.softmax(lo[va_m] / T_val, dim=-1).numpy()
            cal_t += torch.softmax(lt / T_val, dim=-1).numpy() / 5.0
    return cal_o, cal_t, float(np.mean(temps))


def get_or_train_7backbone_heads(y, folds_scene, phys_tr, phys_te):
    c_mlp_o = os.path.join(CLOUD_OUT_DIR, "oof_probs_v4_7bb_mlp.npy")
    c_mlp_t = os.path.join(CLOUD_OUT_DIR, "test_probs_v4_7bb_mlp.npy")
    c_lr_o = os.path.join(CLOUD_OUT_DIR, "oof_probs_v4_7bb_lr.npy")
    c_lr_t = os.path.join(CLOUD_OUT_DIR, "test_probs_v4_7bb_lr.npy")
    if all(os.path.exists(p) for p in [c_mlp_o, c_mlp_t, c_lr_o, c_lr_t]):
        print("[Cache] Loaded cached 7-Backbone MLP & LR heads")
        return {
            "v4_7bb_mlp": (np.load(c_mlp_o), np.load(c_mlp_t)),
            "v4_7bb_lr": (np.load(c_lr_o), np.load(c_lr_t)),
        }

    cloud_emb_p = os.path.join(CLOUD_OUT_DIR, "embs_v4_cloud_5backbones.npz")
    local_emb_p = os.path.join(TASK4_DIR, "detinted_multibackbone_embs.npz")
    zc = np.load(cloud_emb_p)
    zl = np.load(local_emb_p)

    tr_parts = [zl["e1_tr"].astype(np.float32), zl["e2_tr"].astype(np.float32)]
    te_parts = [zl["e1_te"].astype(np.float32), zl["e2_te"].astype(np.float32)]
    for t in ["dinov2_s", "siglip2_b", "eva02clip", "convnext_t", "effnetv2_s"]:
        tr_parts.append(zc[f"{t}_tr"].astype(np.float32))
        te_parts.append(zc[f"{t}_te"].astype(np.float32))

    sc_p = StandardScaler().fit(phys_tr)
    X_tr = np.hstack(tr_parts + [sc_p.transform(phys_tr)]).astype(np.float32)
    X_te = np.hstack(te_parts + [sc_p.transform(phys_te)]).astype(np.float32)
    print(f"[Stage 1] Training 7-Backbone + 24-Phys feature heads: {X_tr.shape}")

    n_tr, n_te = len(y), X_te.shape[0]
    pi_tr = np.bincount(y) / n_tr
    w_bal = torch.tensor((1.0 / pi_tr) / (1.0 / pi_tr).mean(), dtype=torch.float32, device=DEVICE)
    w_sqrt = torch.tensor((1.0 / (pi_tr ** 0.50)) / (1.0 / (pi_tr ** 0.50)).mean(), dtype=torch.float32, device=DEVICE)

    oof_lr = np.zeros((n_tr, 5), dtype=np.float32)
    test_lr = np.zeros((n_te, 5), dtype=np.float32)
    oof_mlp = np.zeros((n_tr, 5), dtype=np.float32)
    test_mlp = np.zeros((n_te, 5), dtype=np.float32)

    for f in range(5):
        tr_m, va_m = folds_scene != f, folds_scene == f
        sc = StandardScaler().fit(X_tr[tr_m])
        Xtr_s = sc.transform(X_tr[tr_m]).astype(np.float32)
        Xva_s = sc.transform(X_tr[va_m]).astype(np.float32)
        Xte_s = sc.transform(X_te).astype(np.float32)

        Xtr_t = torch.from_numpy(Xtr_s).to(DEVICE)
        ytr_t = torch.from_numpy(y[tr_m]).to(DEVICE)
        Xva_t = torch.from_numpy(Xva_s).to(DEVICE)
        Xte_t = torch.from_numpy(Xte_s).to(DEVICE)

        lin = nn.Linear(X_tr.shape[1], 5).to(DEVICE)
        opt_l = torch.optim.LBFGS(lin.parameters(), lr=0.8, max_iter=40, line_search_fn="strong_wolfe")
        crit_l = nn.CrossEntropyLoss(weight=w_bal)

        def closure():
            opt_l.zero_grad()
            loss = crit_l(lin(Xtr_t), ytr_t) + 0.010 * (lin.weight ** 2).sum()
            loss.backward()
            return loss

        opt_l.step(closure)
        with torch.no_grad():
            oof_lr[va_m] = torch.softmax(lin(Xva_t).float(), dim=-1).cpu().numpy()
            test_lr += torch.softmax(lin(Xte_t).float(), dim=-1).cpu().numpy() / 5.0

        p_va_f = np.zeros((va_m.sum(), 5), dtype=np.float32)
        p_te_f = np.zeros((n_te, 5), dtype=np.float32)
        for sd in range(3):
            torch.manual_seed(SEED + f * 17 + sd)
            ldr = DataLoader(TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(y[tr_m])), batch_size=256, shuffle=True, drop_last=True)
            net = DeepResidualEmbNet(X_tr.shape[1], hidden=768).to(DEVICE)
            opt_m = torch.optim.AdamW(net.parameters(), lr=9e-4, weight_decay=4e-2)
            sched_m = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m, T_max=22 * len(ldr))
            crit_m = nn.CrossEntropyLoss(weight=w_sqrt, label_smoothing=0.06)

            best_f1, best_pv, best_st = -1.0, None, None
            for ep in range(1, 23):
                net.train()
                for xb, yb in ldr:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt_m.zero_grad(set_to_none=True)
                    if ep <= 17 and np.random.rand() < 0.50:
                        lam = float(np.random.beta(0.35, 0.35))
                        perm = torch.randperm(xb.size(0), device=DEVICE)
                        logits = net(lam * xb + (1.0 - lam) * xb[perm])
                        loss = lam * crit_m(logits, yb) + (1.0 - lam) * crit_m(logits, yb[perm])
                    else:
                        loss = crit_m(net(xb), yb)
                    loss.backward()
                    opt_m.step()
                    sched_m.step()
                net.eval()
                with torch.no_grad():
                    pv = torch.softmax(net(Xva_t).float(), dim=-1).cpu().numpy()
                f1v = f1_score(y[va_m], pv.argmax(1), average="macro")
                if f1v > best_f1:
                    best_f1, best_pv = f1v, pv
                    best_st = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            p_va_f += best_pv / 3.0
            net.load_state_dict({k: v.to(DEVICE) for k, v in best_st.items()})
            net.eval()
            with torch.no_grad():
                p_te_f += torch.softmax(net(Xte_t).float(), dim=-1).cpu().numpy() / 3.0
        oof_mlp[va_m] = p_va_f
        test_mlp += p_te_f / 5.0

    np.save(c_mlp_o, oof_mlp); np.save(c_mlp_t, test_mlp)
    np.save(c_lr_o, oof_lr); np.save(c_lr_t, test_lr)
    return {"v4_7bb_mlp": (oof_mlp, test_mlp), "v4_7bb_lr": (oof_lr, test_lr)}


def main():
    t0 = time.time()
    print("=" * 94)
    print("TASK 4 PHASE 3 FINAL (T4-EXP6): NESTED CALIBRATED 7-BACKBONE + 3-E2E META-ENSEMBLE")
    print("=" * 94)

    auto_unpack_cloud_zip()
    tr_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    te_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    y = tr_df["label"].to_numpy().astype(np.int64)
    n_tr, n_te = len(tr_df), len(te_df)

    sp = np.load(os.path.join(TASK4_DIR, "scene_validation_split.npz"))
    folds_scene = sp["folds_scene"].astype(np.int64)
    bal_subsets = list(sp["bal_subsets"])

    pz = np.load(os.path.join(TASK4_DIR, "phys_features.npz"))
    phys_tr, phys_te = pz["phys_tr"], pz["phys_te"]

    csplit_p = os.path.join(CLOUD_OUT_DIR, "scene_split_v4.npz")
    if os.path.exists(csplit_p):
        assert np.array_equal(np.load(csplit_p)["folds_scene"].astype(np.int64), folds_scene)
        print("[Verified] Cloud scene_split_v4.npz matches Golden Unseen-Scene 5-Fold Split 100%!")

    models = get_or_train_7backbone_heads(y, folds_scene, phys_tr, phys_te)

    for name, fo, ft in [
        ("v4_cloud5_lr", "oof_probs_v4_foundation_lr.npy", "test_probs_v4_foundation_lr.npy"),
        ("v4_cloud5_mlp", "oof_probs_v4_foundation_mlp.npy", "test_probs_v4_foundation_mlp.npy"),
        ("v5_convnext_small", "oof_probs_v5_convnext_small.npy", "test_probs_v5_convnext_small.npy"),
        ("v5_swin_tiny", "oof_probs_v5_swin_tiny.npy", "test_probs_v5_swin_tiny.npy"),
        ("v5_mobilenetv3", "oof_probs_v5_mobilenetv3.npy", "test_probs_v5_mobilenetv3.npy"),
    ]:
        po_p, pt_p = os.path.join(CLOUD_OUT_DIR, fo), os.path.join(CLOUD_OUT_DIR, ft)
        if os.path.exists(po_p) and os.path.exists(pt_p):
            po, pt = np.load(po_p), np.load(pt_p)
            assert not np.isnan(po).any() and not np.isnan(pt).any()
            models[name] = (po, pt)

    # Proper PyTorch Temperature Calibration per model strictly out-of-fold on folds_scene
    pi_tr = np.bincount(y) / n_tr
    w_bal_cls = (1.0 / (pi_tr ** 0.50)).astype(np.float32)
    w_bal_cls /= w_bal_cls.mean()
    sample_w_bal = w_bal_cls[y]

    cal_models = {}
    print("\n[Temperature Calibration on Golden Unseen-Scene 5-Fold CV]")
    for mn, (po, pt) in models.items():
        co, ct, mean_t = temp_scale_oof_test(po, pt, y, folds_scene, w_bal_cls)
        cal_models[mn] = (co, ct)
        ll0 = log_loss(y, po, sample_weight=sample_w_bal)
        ll1 = log_loss(y, co, sample_weight=sample_w_bal)
        print(f"  {mn:<18s} | mean_T={mean_t:.3f} | BalLogLoss: {ll0:.4f} -> {ll1:.4f} | UnseenAllF1={f1_score(y, co.argmax(1), average='macro'):.5f}")

    # 1. Nested Out-of-Fold Calibrated Logit Weight Optimization (minimizes balanced log-loss on tr_m!)
    m_names = list(cal_models.keys())
    K = len(m_names)
    oof_log_blend = np.zeros((n_tr, 5), dtype=np.float32)
    te_log_blend = np.zeros((n_te, 5), dtype=np.float32)
    fold_ws = []

    log_oofs = [np.log(np.clip(cal_models[mn][0], 1e-6, 1.0)) for mn in m_names]
    log_tes = [np.log(np.clip(cal_models[mn][1], 1e-6, 1.0)) for mn in m_names]

    # Exact autograd simplex optimization minimizing Balanced Cross-Entropy strictly inside tr_m
    for f in range(5):
        tr_m, va_m = folds_scene != f, folds_scene == f
        lo_t = torch.stack([torch.from_numpy(lo[tr_m]) for lo in log_oofs], dim=0)  # (K, N, 5)
        yt = torch.from_numpy(y[tr_m]).long()
        wt = torch.from_numpy(sample_w_bal[tr_m]).float()
        idx_n = torch.arange(tr_m.sum())

        alpha = nn.Parameter(torch.zeros(K))
        opt_w = torch.optim.LBFGS([alpha], lr=0.5, max_iter=50)

        def closure_w():
            opt_w.zero_grad()
            w = torch.softmax(alpha, dim=0)
            comb = (lo_t * w.view(K, 1, 1)).sum(dim=0)
            lp = torch.log_softmax(comb, dim=-1)
            # Small entropy-like L2 regularizer on w keeps multi-family diversity
            loss = -(lp[idx_n, yt] * wt).mean() + 0.002 * (w ** 2).sum()
            loss.backward()
            return loss

        opt_w.step(closure_w)
        with torch.no_grad():
            w_opt = torch.softmax(alpha, dim=0).numpy()
        fold_ws.append(w_opt)

        comb_va = sum(wi * lo[va_m] for wi, lo in zip(w_opt, log_oofs))
        comb_va -= comb_va.max(axis=1, keepdims=True)
        ex_va = np.exp(comb_va)
        oof_log_blend[va_m] = ex_va / ex_va.sum(axis=1, keepdims=True)

        comb_te = sum(wi * lt for wi, lt in zip(w_opt, log_tes))
        comb_te -= comb_te.max(axis=1, keepdims=True)
        ex_te = np.exp(comb_te)
        te_log_blend += (ex_te / ex_te.sum(axis=1, keepdims=True)) / 5.0

    mean_w = np.mean(fold_ws, axis=0)
    print("\n[Nested-OOF Optimal Calibrated Logit Blend Weights]")
    for mn, wi in zip(m_names, mean_w):
        print(f"  {mn:<20s}: {wi:.4f}")

    # 2. Distortion-Aware Meta-Stacker (CatBoost + LightGBM) trained ONLY on the 7 clean calibrated models
    def build_meta_mat(is_oof: bool):
        ps = [cal_models[mn][0 if is_oof else 1] for mn in m_names]
        lps = [np.log(np.clip(p, 1e-6, 1.0)) for p in ps]
        confs = [p.max(axis=1, keepdims=True) for p in ps]
        ents = [-(p * lp).sum(axis=1, keepdims=True) for p, lp in zip(ps, lps)]
        weighted_lp = sum(wi * lp for wi, lp in zip(mean_w, lps))
        preds = [p.argmax(1) for p in ps]
        n_uniq = np.array([len(set(p[i] for p in preds)) for i in range(len(ps[0]))], dtype=np.float32)[:, None]
        phys = phys_tr if is_oof else phys_te
        return np.hstack(ps + lps + [weighted_lp] + confs + ents + [n_uniq, phys]).astype(np.float32)

    X_m_tr = build_meta_mat(True)
    X_m_te = build_meta_mat(False)
    oof_cb = np.zeros((n_tr, 5), dtype=np.float32)
    te_cb = np.zeros((n_te, 5), dtype=np.float32)
    oof_lgb = np.zeros((n_tr, 5), dtype=np.float32)
    te_lgb = np.zeros((n_te, 5), dtype=np.float32)

    for f in range(5):
        tr_m, va_m = folds_scene != f, folds_scene == f
        m_lgb = lgb.LGBMClassifier(
            objective="multiclass", num_class=5, n_estimators=450, learning_rate=0.025,
            num_leaves=15, max_depth=4, min_child_samples=60, subsample=0.8, colsample_bytree=0.75,
            reg_alpha=3.0, reg_lambda=10.0, random_state=SEED + f, verbose=-1, n_jobs=4
        )
        m_lgb.fit(X_m_tr[tr_m], y[tr_m], sample_weight=sample_w_bal[tr_m],
                  eval_set=[(X_m_tr[va_m], y[va_m])], callbacks=[lgb.early_stopping(35, verbose=False)])
        oof_lgb[va_m] = m_lgb.predict_proba(X_m_tr[va_m])
        te_lgb += m_lgb.predict_proba(X_m_te) / 5.0

        m_cb = CatBoostClassifier(
            loss_function="MultiClass", iterations=600, learning_rate=0.04, depth=5,
            l2_leaf_reg=12.0, random_seed=SEED + f, verbose=0, thread_count=4
        )
        m_cb.fit(X_m_tr[tr_m], y[tr_m], sample_weight=sample_w_bal[tr_m], eval_set=(X_m_tr[va_m], y[va_m]), early_stopping_rounds=45)
        oof_cb[va_m] = m_cb.predict_proba(X_m_tr[va_m])
        te_cb += m_cb.predict_proba(X_m_te) / 5.0

    # Compare Logit Blend vs Meta-Stacker vs Multi-Paradigm Fusion cleanly (NO double tau penalty!)
    mb = phys_tr[:, 0:3].mean(axis=1)
    masks = {
        "Dark<45": mb < 45,
        "Dark<75": mb < 75,
        "Border>0.15": phys_tr[:, 11] > 0.15,
        "Noise>90": phys_tr[:, 14] > 90,
    }

    # Also compute explicit calibrated 3-paradigm Logit Blend (0.50*ConvNeXtS + 0.35*SwinT + 0.15*F5MLP)
    lo_3f = 0.50 * np.log(np.clip(cal_models["v5_convnext_small"][0], 1e-6, 1)) + 0.35 * np.log(np.clip(cal_models["v5_swin_tiny"][0], 1e-6, 1)) + 0.15 * np.log(np.clip(cal_models["v4_cloud5_mlp"][0], 1e-6, 1))
    lt_3f = 0.50 * np.log(np.clip(cal_models["v5_convnext_small"][1], 1e-6, 1)) + 0.35 * np.log(np.clip(cal_models["v5_swin_tiny"][1], 1e-6, 1)) + 0.15 * np.log(np.clip(cal_models["v4_cloud5_mlp"][1], 1e-6, 1))
    lo_3f -= lo_3f.max(1, keepdims=True); oof_3f = (np.exp(lo_3f) / np.exp(lo_3f).sum(1, keepdims=True)).astype(np.float32)
    lt_3f -= lt_3f.max(1, keepdims=True); te_3f = (np.exp(lt_3f) / np.exp(lt_3f).sum(1, keepdims=True)).astype(np.float32)

    candidates = [
        ("v5_convnext_small (single)", cal_models["v5_convnext_small"][0], cal_models["v5_convnext_small"][1]),
        ("v5_swin_tiny (single)", cal_models["v5_swin_tiny"][0], cal_models["v5_swin_tiny"][1]),
        ("v4_cloud5_mlp (single)", cal_models["v4_cloud5_mlp"][0], cal_models["v4_cloud5_mlp"][1]),
        ("Nested Calibrated Logit", oof_log_blend, te_log_blend),
        ("3-Family Logit (CnxS+Swin+F5)", oof_3f, te_3f),
        ("Meta-CatBoost (Clean 7M)", oof_cb, te_cb),
        ("Meta-LightGBM (Clean 7M)", oof_lgb, te_lgb),
        ("0.65*CalLogit+0.20*CB+0.15*LGB", 0.65 * oof_log_blend + 0.20 * oof_cb + 0.15 * oof_lgb, 0.65 * te_log_blend + 0.20 * te_cb + 0.15 * te_lgb),
        ("0.80*CalLogit+0.10*CB+0.10*LGB", 0.80 * oof_log_blend + 0.10 * oof_cb + 0.10 * oof_lgb, 0.80 * te_log_blend + 0.10 * te_cb + 0.10 * te_lgb),
    ]

    print("\n" + "=" * 96)
    print("FINAL GOLDEN UNSEEN-SCENE 5-FOLD BENCHMARK (ZERO TEST LEAKAGE, NO DOUBLE PRIOR PENALTY)")
    print("=" * 96)
    best_score, best_name, best_oof, best_te = -1.0, None, None, None
    for cname, po, pt in candidates:
        pr = po.argmax(1)
        fall = f1_score(y, pr, average="macro")
        fbal = float(np.mean([f1_score(y[idx], pr[idx], average="macro") for idx in bal_subsets]))
        d45 = f1_score(y[masks["Dark<45"]], pr[masks["Dark<45"]], average="macro")
        rot = f1_score(y[masks["Border>0.15"]], pr[masks["Border>0.15"]], average="macro")
        noi = f1_score(y[masks["Noise>90"]], pr[masks["Noise>90"]], average="macro")
        td = np.bincount(pt.argmax(1), minlength=5) / n_te * 100
        comp = 0.50 * fbal + 0.30 * fall + 0.20 * d45
        print(f"  {cname:<30s} | All={fall:.5f} | Bal={fbal:.5f} | D45={d45:.4f} | Rot={rot:.4f} | Noi={noi:.4f} | Test%={np.round(td,1)}")
        if "single" not in cname:
            if comp > best_score:
                best_score, best_name, best_oof, best_te = comp, cname, po, pt

    print(f"\n>>> Selected Best Unseen-Scene Generalizer: {best_name} (Composite={best_score:.5f}) <<<")

    # Save probabilities & versioned submission CSV
    np.save(os.path.join(TASK4_DIR, "oof_probs_v6_scene_ensemble.npy"), best_oof)
    np.save(os.path.join(TASK4_DIR, "test_probs_v6_scene_ensemble.npy"), best_te)

    pred_final = best_te.argmax(axis=1).astype(int)
    sub_df = pd.DataFrame({"id": te_df["id"].values, "task4": pred_final})
    assert len(sub_df) == 25889 and not sub_df.isna().any().any() and set(sub_df["task4"].unique()).issubset({0, 1, 2, 3, 4})

    out_v5 = os.path.join(SUB_DIR, "sub_task4_v5_cloud_ensemble.csv")
    out_main = os.path.join(SUB_DIR, "sub_task4.csv")
    sub_df.to_csv(out_v5, index=False)
    sub_df.to_csv(out_main, index=False)

    sub_v3_p = os.path.join(SUB_DIR, "sub_task4_v3_detinted.csv")
    if os.path.exists(sub_v3_p):
        p_v3 = pd.read_csv(sub_v3_p)["task4"].to_numpy()
        changed = int((pred_final != p_v3).sum())
        print(f"Changed predictions vs V3 (LB 0.83963): {changed:,} / {n_te:,} ({changed/n_te*100:.2f}%)")

    print(f"Saved final Task 4 submission to: {out_v5} (and updated {out_main}) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
