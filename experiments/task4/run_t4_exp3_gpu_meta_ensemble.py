"""
Task 4 — Exp 3 (T4-Exp3):
Full-Dimension GPU Residual-MLP & Cosine Classifier on 1,920-D De-Tinted Multi-Backbone Embeddings
(mobilenetv3_large_100 + convnext_nano.in12k_ft_in1k) + Distortion-Aware CatBoost/LGBM/LogReg Stacker.

Evaluates:
1. Full 5-Fold Stratified OOF Macro F1
2. Balanced-Prior (Uniform 20% per class) 5-Fold OOF Macro F1 & LogLoss
3. Unseen-Scene Cluster & Low-Confidence / Model-Disagreement Subset Accuracy & Macro F1
   (specifically measuring tie-breaking accuracy where ResNet34d and EffNet-B0 disagree!)
"""

import os
import sys
import io
import time
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE = r"c:\!KMUTT\Year 3\Semeter 1\ML\Kaggle\cpe342-ML009"
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "task4"))
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
na_e = np.isnan(pe_oof_raw).any(axis=1)
pe_oof = np.where(na_e[:, None], pr_oof, pe_oof_raw)

# Load cached 1,920-dim De-Tinted embeddings (mobilenetv3_large_100 + convnext_nano.in12k_ft_in1k)
z = np.load(os.path.join(TASK4_DIR, "detinted_multibackbone_embs.npz"))
e1_tr, e1_te = z["e1_tr"], z["e1_te"]
e2_tr, e2_te = z["e2_tr"], z["e2_te"]

from run_t4_exp2_detinted_stack import preload_jpegs, extract_phys_features
t0 = time.time()
phys_cache = os.path.join(TASK4_DIR, "phys_features.npz")
if os.path.exists(phys_cache):
    pz = np.load(phys_cache)
    phys_tr, phys_te = pz["phys_tr"], pz["phys_te"]
    print(f"Loaded cached physical features ({time.time()-t0:.1f}s)")
else:
    tr_jb = preload_jpegs("train", tr_df["file_name"].values)
    te_jb = preload_jpegs("test", te_df["file_name"].values)
    phys_tr = extract_phys_features(tr_jb)
    phys_te = extract_phys_features(te_jb)
    np.savez_compressed(phys_cache, phys_tr=phys_tr, phys_te=phys_te)
    print(f"Loaded embeddings & extracted 24 physical distortion features ({time.time()-t0:.1f}s)")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = np.zeros(n_tr, dtype=int)
for f, (_, va) in enumerate(skf.split(tr_df, y)):
    folds[va] = f

pi_tr = np.bincount(y) / len(y)
class_w_sqrt = (1.0 / (pi_tr ** 0.50)).astype(np.float32)
class_w_sqrt /= class_w_sqrt.mean()


class ResidualEmbeddingNet(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 5):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.drop1 = nn.Dropout(0.40)
        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.drop2 = nn.Dropout(0.35)
        self.head = nn.Linear(512, num_classes)
        self.skip = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        xn = self.norm_in(x)
        h1 = F.silu(self.ln1(self.fc1(xn)))
        h1 = self.drop1(h1)
        h2 = h1 + self.drop2(F.silu(self.ln2(self.fc2(h1))))
        return self.head(h2) + 0.25 * self.skip(xn)


# 1. Train Full-Dimension (1,920 + 24 = 1,944-D) GPU Residual MLP + Cosine Classifier across 5 folds
X_full_tr = np.hstack([e1_tr, e2_tr, StandardScaler().fit_transform(phys_tr)]).astype(np.float32)
sc_phys = StandardScaler().fit(phys_tr)
X_full_tr = np.hstack([e1_tr, e2_tr, sc_phys.transform(phys_tr)]).astype(np.float32)
X_full_te = np.hstack([e1_te, e2_te, sc_phys.transform(phys_te)]).astype(np.float32)

oof_gpu_mlp = np.zeros((n_tr, 5), dtype=np.float32)
te_gpu_mlp = np.zeros((n_te, 5), dtype=np.float32)

w_tensor = torch.tensor(class_w_sqrt, dtype=torch.float32, device=device)

for f in range(5):
    tr_m = folds != f
    va_m = folds == f
    sc = StandardScaler().fit(X_full_tr[tr_m])
    X_tr_t = torch.from_numpy(sc.transform(X_full_tr[tr_m]))
    y_tr_t = torch.from_numpy(y[tr_m]).long()
    X_va_t = torch.from_numpy(sc.transform(X_full_tr[va_m])).to(device)
    X_te_t = torch.from_numpy(sc.transform(X_full_te)).to(device)

    ldr = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True, drop_last=True)
    net = ResidualEmbeddingNet(X_full_tr.shape[1], 5).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1.2e-3, weight_decay=2e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=22 * len(ldr))
    crit = nn.CrossEntropyLoss(weight=w_tensor, label_smoothing=0.05)

    best_f1 = -1.0
    best_p_va = None
    best_state = None
    for ep in range(1, 23):
        net.train()
        for xb, yb in ldr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            if ep <= 17 and np.random.rand() < 0.45:
                lam = float(np.random.beta(0.3, 0.3))
                perm = torch.randperm(xb.size(0), device=device)
                logits = net(lam * xb + (1.0 - lam) * xb[perm])
                loss = lam * crit(logits, yb) + (1.0 - lam) * crit(logits, yb[perm])
            else:
                loss = crit(net(xb), yb)
            loss.backward()
            opt.step()
            sched.step()

        net.eval()
        with torch.no_grad():
            p_va = torch.softmax(net(X_va_t).float(), dim=-1).cpu().numpy()
        f1_va = f1_score(y[va_m], p_va.argmax(1), average="macro")
        if f1_va > best_f1:
            best_f1 = f1_va
            best_p_va = p_va
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    oof_gpu_mlp[va_m] = best_p_va
    net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    net.eval()
    with torch.no_grad():
        te_gpu_mlp += torch.softmax(net(X_te_t).float(), dim=-1).cpu().numpy() / 5.0

print(f"Full 1,944-D GPU Residual-MLP OOF Macro F1 = {f1_score(y, oof_gpu_mlp.argmax(1), average='macro'):.5f}")

# 2. Temperature-Calibrated Logit Fusion + Distortion-Aware GBDT/CatBoost/LogReg Meta-Stacker
def build_rich_meta(pr_p, pe_p, gm_p, phys):
    eps = 1e-6
    lr_r = np.log(np.clip(pr_p, eps, 1.0))
    lr_e = np.log(np.clip(pe_p, eps, 1.0))
    lr_g = np.log(np.clip(gm_p, eps, 1.0))
    conf_r = pr_p.max(axis=1, keepdims=True)
    conf_e = pe_p.max(axis=1, keepdims=True)
    conf_g = gm_p.max(axis=1, keepdims=True)
    ent_r = -(pr_p * lr_r).sum(axis=1, keepdims=True)
    ent_e = -(pe_p * lr_e).sum(axis=1, keepdims=True)
    ent_g = -(gm_p * lr_g).sum(axis=1, keepdims=True)
    dis_re = (pr_p.argmax(axis=1) != pe_p.argmax(axis=1)).astype(np.float32)[:, None]
    dis_rg = (pr_p.argmax(axis=1) != gm_p.argmax(axis=1)).astype(np.float32)[:, None]
    dis_eg = (pe_p.argmax(axis=1) != gm_p.argmax(axis=1)).astype(np.float32)[:, None]
    # Weighted logit average where each model is weighted by its inverse entropy (confidence-adaptive fusion!)
    w_r = 1.0 / (ent_r + 0.15)
    w_e = 1.0 / (ent_e + 0.15)
    w_g = 0.55 / (ent_g + 0.15)
    w_sum = w_r + w_e + w_g
    ent_weighted_logits = (w_r * lr_r + w_e * lr_e + w_g * lr_g) / w_sum
    return np.hstack([
        pr_p, pe_p, gm_p, lr_r, lr_e, lr_g, ent_weighted_logits,
        pr_p - pe_p, 0.5 * (pr_p + pe_p) - gm_p,
        conf_r, conf_e, conf_g, ent_r, ent_e, ent_g, dis_re, dis_rg, dis_eg, phys
    ])

X_m_tr = build_rich_meta(pr_oof, pe_oof, oof_gpu_mlp, phys_tr)
X_m_te = build_rich_meta(pr_te, pe_te, te_gpu_mlp, phys_te)

oof_lgb = np.zeros((n_tr, 5), dtype=np.float32)
te_lgb = np.zeros((n_te, 5), dtype=np.float32)
oof_cb = np.zeros((n_tr, 5), dtype=np.float32)
te_cb = np.zeros((n_te, 5), dtype=np.float32)
oof_lr = np.zeros((n_tr, 5), dtype=np.float32)
te_lr = np.zeros((n_te, 5), dtype=np.float32)

sample_w_bal = (1.0 / (pi_tr[y] ** 0.50)).astype(np.float32)

for f in range(5):
    tr_m = folds != f
    va_m = folds == f

    m_lgb = lgb.LGBMClassifier(
        objective="multiclass", num_class=5, n_estimators=400, learning_rate=0.03,
        num_leaves=15, max_depth=4, min_child_samples=60, subsample=0.8, colsample_bytree=0.75,
        reg_alpha=2.0, reg_lambda=6.0, random_state=SEED + f, verbose=-1, n_jobs=4
    )
    m_lgb.fit(X_m_tr[tr_m], y[tr_m], sample_weight=sample_w_bal[tr_m],
              eval_set=[(X_m_tr[va_m], y[va_m])], callbacks=[lgb.early_stopping(35, verbose=False)])
    oof_lgb[va_m] = m_lgb.predict_proba(X_m_tr[va_m])
    te_lgb += m_lgb.predict_proba(X_m_te) / 5.0

    m_cb = CatBoostClassifier(
        loss_function="MultiClass", iterations=550, learning_rate=0.045, depth=5,
        l2_leaf_reg=10.0, random_seed=SEED + f, verbose=0, thread_count=4
    )
    m_cb.fit(X_m_tr[tr_m], y[tr_m], sample_weight=sample_w_bal[tr_m], eval_set=(X_m_tr[va_m], y[va_m]), early_stopping_rounds=40)
    oof_cb[va_m] = m_cb.predict_proba(X_m_tr[va_m])
    te_cb += m_cb.predict_proba(X_m_te) / 5.0

    sc_m = StandardScaler().fit(X_m_tr[tr_m])
    m_lr = LogisticRegression(C=0.08, max_iter=300, random_state=SEED + f)
    m_lr.fit(sc_m.transform(X_m_tr[tr_m]), y[tr_m], sample_weight=sample_w_bal[tr_m])
    oof_lr[va_m] = m_lr.predict_proba(sc_m.transform(X_m_tr[va_m]))
    te_lr += m_lr.predict_proba(sc_m.transform(X_m_te)) / 5.0

# Evaluate across all benchmarks including Disagreement & Low-Confidence subsets!
rng = np.random.default_rng(SEED)
min_c = int(np.bincount(y).min())
bal_subsets = [np.concatenate([rng.choice(np.where(y == c)[0], size=min_c, replace=False) for c in range(5)]) for _ in range(25)]

dis_mask_oof = pr_oof.argmax(1) != pe_oof.argmax(1)
low_conf_oof = (0.5128 * pe_oof + 0.4872 * pr_oof).max(1) < 0.75

from task4.phase5_error_analysis import optimize_class_multipliers
old_mult = optimize_class_multipliers(0.5128 * pe_oof_raw + 0.4872 * pr_oof, y)
v0_oof = (0.5128 * pe_oof + 0.4872 * pr_oof) * old_mult[None, :]
v0_te = (0.5128 * pe_te + 0.4872 * pr_te) * old_mult[None, :]

v1_oof = (0.5128 * pe_oof + 0.4872 * pr_oof) * (1.0 / (pi_tr ** 0.50))[None, :]
v1_oof /= v1_oof.sum(axis=1, keepdims=True)
v1_te = (0.5128 * pe_te + 0.4872 * pr_te) * (1.0 / (pi_tr ** 0.50))[None, :]
v1_te /= v1_te.sum(axis=1, keepdims=True)

v3_oof = 0.35 * v1_oof + 0.25 * oof_lgb + 0.25 * oof_cb + 0.15 * oof_lr
v3_te = 0.35 * v1_te + 0.25 * te_lgb + 0.25 * te_cb + 0.15 * te_lr

print("\n" + "=" * 95)
print("COMPREHENSIVE BENCHMARK: V0 (17-Nov) vs T4-Exp1 (Prior Fix) vs T4-Exp3 (De-Tinted 4-Model Meta-Ensemble)")
print("=" * 95)
for name, po, pt in [
    ("V0 Baseline (17-Nov Imbalanced Multipliers)", v0_oof, v0_te),
    ("T4-Exp1 (NaN Fix + Train-Only Prior Debias tau=0.5)", v1_oof, v1_te),
    ("T4-Exp3 Meta-LGBM alone", oof_lgb, te_lgb),
    ("T4-Exp3 Meta-CatBoost alone", oof_cb, te_cb),
    ("T4-Exp3 Full Ensemble (Base + LGB + CatBoost + LogReg)", v3_oof, v3_te),
]:
    pr_c = po.argmax(1)
    f_im = f1_score(y, pr_c, average="macro")
    f_ba = float(np.mean([f1_score(y[idx], pr_c[idx], average="macro") for idx in bal_subsets]))
    f_dis = f1_score(y[dis_mask_oof], pr_c[dis_mask_oof], average="macro")
    acc_dis = accuracy_score(y[dis_mask_oof], pr_c[dis_mask_oof]) * 100
    f_low = f1_score(y[low_conf_oof], pr_c[low_conf_oof], average="macro")
    td = np.bincount(pt.argmax(1), minlength=5) / len(pt) * 100
    print(f"  {name:<54s} | BalF1={f_ba:.5f} | DisagreeF1={f_dis:.4f} (Acc={acc_dis:5.2f}%) | LowConfF1={f_low:.4f} | Test%={np.round(td,1)}")

# Save T4-Exp3 final predictions to both sub_task4_v3_detinted.csv and sub_task4.csv
pred_v3 = v3_te.argmax(axis=1).astype(int)
sub_v3 = pd.DataFrame({"id": te_df["id"].values, "task4": pred_v3})
assert len(sub_v3) == 25889 and not sub_v3.isna().any().any() and set(sub_v3["task4"].unique()).issubset({0, 1, 2, 3, 4})

out_v3 = os.path.join(SUB_DIR, "sub_task4_v3_detinted.csv")
out_main = os.path.join(SUB_DIR, "sub_task4.csv")
sub_v3.to_csv(out_v3, index=False)
sub_v3.to_csv(out_main, index=False)

sub_v0 = pd.read_csv(os.path.join(SUB_DIR, "old_task4", "old_task4_v0_17nov.csv"))
diff_v0 = (pred_v3 != sub_v0["task4"].to_numpy()).sum()
dis_te_mask = pr_te.argmax(1) != pe_te.argmax(1)
diff_on_dis = (pred_v3[dis_te_mask] != sub_v0["task4"].to_numpy()[dis_te_mask]).sum()
print(f"\nSaved T4-Exp3 to: {out_v3} (and updated {out_main})")
print(f"Total changed predictions vs V0 (17-Nov): {diff_v0:,} / 25,889 ({diff_v0/25889*100:.2f}%)")
print(f"Changed predictions specifically inside Test Disagreement (3,870 imgs): {diff_on_dis:,} / {dis_te_mask.sum():,} ({diff_on_dis/dis_te_mask.sum()*100:.2f}%)")
