"""
Train Task 2: Player Segment Classification (4-class)
Upgraded Leak-Free Multi-Paradigm Pipeline (5-Fold Stratified CV):
  1. Denoising: Drop 16 pure-noise features identified via DGP & statistical analysis
  2. Leak-Free Fold Feature Engineering:
     - Standardized Z-score Composite Latent Axis Scores (Engagement Axis & Spending Axis)
     - Log1p transforms for skewed features + Interaction categorical (engagement_level x loyalty_tier)
     - Inner-OOF Binned Log-Likelihood / Naive Bayes meta-features (zero target leakage)
  3. 6 Diverse Level-1 Models (all trained from scratch):
     - LightGBM Classifier
     - XGBoost Classifier
     - CatBoost Classifier (native categoricals)
     - Decomposed 2-Axis Binary LightGBM (Engagement Axis x Spending Axis)
     - Spline GAM (SplineTransformer + MissingIndicator + OneHotEncoder + LogisticRegression)
     - MLP Neural Network (MLPClassifier)
  4. Level-2 Out-of-Fold Stacking (LogisticRegression on probabilities + log-probabilities)
  5. Coordinate Ascent + Multi-Start Nelder-Mead Class Multiplier Tuning for Macro F1
"""
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

SEED = 42
N_SPLITS = 5
NUM_CLASSES = 4
CLASS_NAMES = {0: "Casual", 1: "Grinder", 2: "Social", 3: "Whale"}

np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "task2")
SUB_DIR = os.path.join(BASE_DIR, "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

# 16 Pure-Noise Columns (zero correlation / identical distribution across all 4 segments)
NOISE_COLS = [
    "random_metric_1", "random_metric_2", "random_metric_3",
    "region", "platform", "device_type", "payment_method", "language",
    "account_status", "player_type_tag", "skill_tier",
    "account_age_days", "avg_match_length", "preferred_game_mode",
    "responds_to_discounts", "item_type_preference_social",
]

# Latent Axis A: Engagement / Activity (separates {1: Grinder, 3: Whale} vs {0: Casual, 2: Social})
ENG_COLS = [
    "days_since_last_login", "total_playtime_hours", "friend_count", "friend_invites_sent",
    "avg_session_duration", "gifts_sent_received", "tournament_entries", "speed_of_progression",
    "play_frequency", "peak_concurrent_hours", "login_streak", "chat_activity_score",
    "win_rate_ranked", "item_type_preference_performance", "team_play_percentage",
    "watches_esports", "competitive_rank", "ranked_participation_rate",
]

# Latent Axis B: Spending / Loyalty (separates {2: Social, 3: Whale} vs {0: Casual, 1: Grinder})
SPEND_COLS = [
    "vip_tier", "spending_frequency", "avg_monthly_spending", "rare_items_count",
    "total_spending_thb", "item_type_preference_cosmetic",
]

# Shared signal features correlating with both axes
BOTH_COLS = ["collection_progress", "achievement_completion_rate"]

NUM_SIGNAL = ENG_COLS + SPEND_COLS + BOTH_COLS
CAT_SIGNAL = ["engagement_level", "loyalty_tier"]
CAT_ALL = CAT_SIGNAL + ["eng_x_loyalty"]

SKEWED_COLS = [
    "vip_tier", "total_spending_thb", "avg_monthly_spending", "rare_items_count",
    "total_playtime_hours", "friend_count", "chat_activity_score",
    "gifts_sent_received", "tournament_entries",
]


def prepare_base_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only signal columns and build clean categorical interaction."""
    out = df[NUM_SIGNAL + CAT_SIGNAL].copy()
    for c in CAT_SIGNAL:
        out[c] = out[c].fillna("MISSING").astype(str)
    out["eng_x_loyalty"] = out["engagement_level"] + "_" + out["loyalty_tier"]
    return out


def compute_nb_features_fit_transform(X_tr_df: pd.DataFrame, y_tr: np.ndarray, X_others: list, n_bins: int = 35):
    """Fit quantile-binned Naive Bayes log-posteriors on X_tr_df (via inner OOF) and transform X_others."""
    def _fit_nb(df_fit: pd.DataFrame, y_fit: np.ndarray):
        priors = np.bincount(y_fit, minlength=NUM_CLASSES) / len(y_fit)
        cat_tables = {}
        for c in CAT_ALL:
            ct = pd.crosstab(df_fit[c], y_fit)
            ct = (ct + 1.0).div(ct.sum(axis=0) + len(ct), axis=1)
            cat_tables[c] = ct

        num_bins = {}
        num_tables = {}
        for c in NUM_SIGNAL:
            vals = df_fit[c].dropna()
            _, bins = pd.qcut(vals, q=n_bins, retbins=True, duplicates="drop")
            bins[0] = -1e12
            bins[-1] = 1e12
            b = pd.cut(df_fit[c], bins=bins).cat.codes
            ct = pd.crosstab(b, y_fit)
            ct = (ct + 2.0).div(ct.sum(axis=0) + 2.0 * len(ct), axis=1)
            num_bins[c] = bins
            num_tables[c] = ct
        return priors, cat_tables, num_bins, num_tables

    def _transform_nb(df_t: pd.DataFrame, params):
        priors, cat_tables, num_bins, num_tables = params
        log_post = np.tile(np.log(priors + 1e-12), (len(df_t), 1))
        for c, ct in cat_tables.items():
            s = df_t[c]
            for k in range(NUM_CLASSES):
                col_map = ct[k] if k in ct.columns else pd.Series(dtype=float)
                p = s.map(col_map).fillna(1.0 / max(1, len(ct))).values
                log_post[:, k] += np.log(np.clip(p, 1e-8, 1.0))
        for c in NUM_SIGNAL:
            bins = num_bins[c]
            ct = num_tables[c]
            b = pd.cut(df_t[c], bins=bins).cat.codes
            for k in range(NUM_CLASSES):
                col_map = ct[k] if k in ct.columns else pd.Series(dtype=float)
                p = b.map(col_map).fillna(1.0 / max(1, len(ct))).values
                log_post[:, k] += np.log(np.clip(p, 1e-8, 1.0))
        log_post -= log_post.max(axis=1, keepdims=True)
        probs = np.exp(log_post)
        probs /= probs.sum(axis=1, keepdims=True)
        return log_post, probs

    inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 100)
    tr_logp = np.zeros((len(X_tr_df), NUM_CLASSES))
    tr_prob = np.zeros((len(X_tr_df), NUM_CLASSES))
    for i_tr, i_va in inner_skf.split(X_tr_df, y_tr):
        params_i = _fit_nb(X_tr_df.iloc[i_tr], y_tr[i_tr])
        lp, pr = _transform_nb(X_tr_df.iloc[i_va], params_i)
        tr_logp[i_va] = lp
        tr_prob[i_va] = pr

    full_params = _fit_nb(X_tr_df, y_tr)
    other_results = [_transform_nb(df_o, full_params) for df_o in X_others]
    return (tr_logp, tr_prob), other_results


def build_fold_features(X_tr_raw: pd.DataFrame, y_tr: np.ndarray, X_others_raw: list):
    """Build strictly leak-free fold features: Z-score composites, log1p, and inner-OOF Naive Bayes features."""
    means = X_tr_raw[NUM_SIGNAL].mean()
    stds = X_tr_raw[NUM_SIGNAL].std().replace(0, 1.0)

    y_eng = np.isin(y_tr, [1, 3]).astype(float)
    y_spd = np.isin(y_tr, [2, 3]).astype(float)
    eng_weights = np.array([X_tr_raw[c].corr(pd.Series(y_eng, index=X_tr_raw.index)) for c in ENG_COLS])
    spd_weights = np.array([X_tr_raw[c].corr(pd.Series(y_spd, index=X_tr_raw.index)) for c in SPEND_COLS])
    eng_signs = np.array([-1.0 if c == "days_since_last_login" else 1.0 for c in ENG_COLS])

    (tr_nb_lp, tr_nb_pr), other_nb = compute_nb_features_fit_transform(X_tr_raw, y_tr, X_others_raw)

    def _apply(df_in: pd.DataFrame, nb_lp: np.ndarray, nb_pr: np.ndarray) -> pd.DataFrame:
        df = df_in.copy()
        for c in SKEWED_COLS:
            df[f"log1p_{c}"] = np.log1p(np.clip(df[c], 0, None))

        Z = (df[NUM_SIGNAL] - means) / stds
        Z_eng = Z[ENG_COLS].values * eng_signs
        Z_eng_w = Z[ENG_COLS].values * eng_weights
        Z_spd = Z[SPEND_COLS].values
        Z_spd_w = Z[SPEND_COLS].values * spd_weights
        Z_both = Z[BOTH_COLS].values

        df["eng_z_mean"] = np.nanmean(Z_eng, axis=1)
        df["eng_z_sum"] = np.nansum(Z_eng, axis=1)
        df["eng_z_std"] = np.nanstd(Z_eng, axis=1)
        df["eng_w_mean"] = np.nanmean(Z_eng_w, axis=1)
        df["eng_valid_cnt"] = np.sum(~np.isnan(Z_eng), axis=1)

        df["spend_z_mean"] = np.nanmean(Z_spd, axis=1)
        df["spend_z_sum"] = np.nansum(Z_spd, axis=1)
        df["spend_z_std"] = np.nanstd(Z_spd, axis=1)
        df["spend_w_mean"] = np.nanmean(Z_spd_w, axis=1)
        df["spend_valid_cnt"] = np.sum(~np.isnan(Z_spd), axis=1)

        df["both_z_mean"] = np.nanmean(Z_both, axis=1)
        df["eng_x_spend_z"] = df["eng_z_mean"] * df["spend_z_mean"]
        df["eng_minus_spend_z"] = df["eng_z_mean"] - df["spend_z_mean"]
        df["eng_plus_spend_z"] = df["eng_z_mean"] + df["spend_z_mean"]

        for k in range(NUM_CLASSES):
            df[f"nb_logp_{k}"] = nb_lp[:, k]
            df[f"nb_prob_{k}"] = nb_pr[:, k]
        df["nb_eng_prob"] = nb_pr[:, 1] + nb_pr[:, 3]
        df["nb_spd_prob"] = nb_pr[:, 2] + nb_pr[:, 3]
        return df

    df_tr_out = _apply(X_tr_raw, tr_nb_lp, tr_nb_pr)
    others_out = [_apply(df_o, lp, pr) for df_o, (lp, pr) in zip(X_others_raw, other_nb)]
    return df_tr_out, others_out


def build_stack_features(prob_list: list) -> np.ndarray:
    """Combine raw probabilities and clipped log-probabilities for Level-2 meta-learner."""
    raw = np.hstack(prob_list)
    log_p = np.log(np.clip(raw, 1e-6, 1.0))
    return np.hstack([raw, log_p])


def optimize_class_multipliers(probs: np.ndarray, y_true: np.ndarray):
    """Find class probability multipliers maximizing Macro F1 via Coordinate Ascent + Nelder-Mead."""
    best_weights = np.ones(NUM_CLASSES)
    best_score = f1_score(y_true, probs.argmax(axis=1), average="macro")

    # Coordinate Ascent passes
    w = np.ones(NUM_CLASSES)
    for _ in range(4):
        for k in range(NUM_CLASSES):
            grid = np.linspace(0.75, 1.30, 111)
            for val in grid:
                w_try = w.copy()
                w_try[k] = val
                s = f1_score(y_true, (probs * w_try).argmax(axis=1), average="macro")
                if s > best_score:
                    best_score = s
                    w[k] = val
                    best_weights = w_try.copy()

    # Multi-start Nelder-Mead refinement
    for seed_i in range(5):
        x0 = best_weights.copy() if seed_i == 0 else np.random.RandomState(SEED + seed_i).uniform(0.85, 1.15, size=NUM_CLASSES)
        res = minimize(
            lambda m: -f1_score(y_true, (probs * m).argmax(axis=1), average="macro"),
            x0=x0,
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-3, "fatol": 1e-5},
        )
        if -res.fun > best_score:
            best_score = -res.fun
            best_weights = res.x.copy()

    best_weights = best_weights / np.mean(best_weights)
    return best_weights, best_score


def main():
    t0 = time.time()
    print("=" * 78)
    print("🚀 Task 2 Upgraded Pipeline: Multi-Paradigm Leak-Free Ensemble + Stacking")
    print("=" * 78)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Loaded Train: {train_df.shape} | Test: {test_df.shape}")

    ID_COL = "id" if "id" in train_df.columns else train_df.columns[0]
    TARGET_COL = "segment"

    print("\n--- Class Distribution in Train ---")
    vc = train_df[TARGET_COL].value_counts().sort_index()
    for k, v in vc.items():
        print(f"  Class {int(k)} ({CLASS_NAMES[int(k)]}): {v:,} ({v / len(train_df) * 100:.2f}%)")

    X_base = prepare_base_df(train_df)
    X_test_base = prepare_base_df(test_df)
    y = train_df[TARGET_COL].astype(int).values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    model_names = ["LGB", "XGB", "CAT", "2AX", "GAM", "MLP"]
    oof_preds = {m: np.zeros((len(X_base), NUM_CLASSES)) for m in model_names}
    test_preds = {m: np.zeros((len(X_test_base), NUM_CLASSES)) for m in model_names}

    print(f"\n--- Training {len(model_names)} Diverse Models across {N_SPLITS} Stratified Folds ---")
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_base, y), 1):
        f_start = time.time()
        X_tr_raw = X_base.iloc[trn_idx].copy()
        X_va_raw = X_base.iloc[val_idx].copy()
        y_tr, y_va = y[trn_idx], y[val_idx]

        # Leak-free fold feature engineering
        X_tr_fe, [X_va_fe, X_te_fe] = build_fold_features(X_tr_raw, y_tr, [X_va_raw, X_test_base])

        # Train-only integer encoding for LightGBM / XGBoost
        X_tr_num = X_tr_fe.copy()
        X_va_num = X_va_fe.copy()
        X_te_num = X_te_fe.copy()
        for c in CAT_ALL:
            cats = sorted(X_tr_num[c].unique())
            mp = {v: i for i, v in enumerate(cats)}
            X_tr_num[c] = X_tr_num[c].map(mp).fillna(-1).astype(int)
            X_va_num[c] = X_va_num[c].map(mp).fillna(-1).astype(int)
            X_te_num[c] = X_te_num[c].map(mp).fillna(-1).astype(int)

        # 1. LightGBM
        m_lgb = lgb.LGBMClassifier(
            objective="multiclass", num_class=NUM_CLASSES,
            n_estimators=2000, learning_rate=0.025, num_leaves=31,
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
            random_state=SEED + fold, n_jobs=-1, verbose=-1,
        )
        m_lgb.fit(X_tr_num, y_tr, eval_set=[(X_va_num, y_va)], callbacks=[lgb.early_stopping(60, verbose=False)])
        oof_preds["LGB"][val_idx] = m_lgb.predict_proba(X_va_num)
        test_preds["LGB"] += m_lgb.predict_proba(X_te_num) / N_SPLITS

        # 2. XGBoost
        m_xgb = xgb.XGBClassifier(
            objective="multi:softprob", num_class=NUM_CLASSES, eval_metric="mlogloss",
            n_estimators=2000, learning_rate=0.025, max_depth=5, min_child_weight=20,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
            tree_method="hist", early_stopping_rounds=60,
            random_state=SEED + fold, n_jobs=-1,
        )
        m_xgb.fit(X_tr_num, y_tr, eval_set=[(X_va_num, y_va)], verbose=False)
        oof_preds["XGB"][val_idx] = m_xgb.predict_proba(X_va_num)
        test_preds["XGB"] += m_xgb.predict_proba(X_te_num) / N_SPLITS

        # 3. CatBoost
        m_cat = CatBoostClassifier(
            loss_function="MultiClass", classes_count=NUM_CLASSES,
            iterations=2000, learning_rate=0.04, depth=6, l2_leaf_reg=5.0,
            cat_features=CAT_ALL, random_seed=SEED + fold, thread_count=-1, verbose=False,
        )
        m_cat.fit(X_tr_fe, y_tr, eval_set=(X_va_fe, y_va), early_stopping_rounds=60, verbose=False)
        oof_preds["CAT"][val_idx] = m_cat.predict_proba(X_va_fe)
        test_preds["CAT"] += m_cat.predict_proba(X_te_fe) / N_SPLITS

        # 4. Decomposed 2-Axis Binary LightGBM
        y_tr_eng = np.isin(y_tr, [1, 3]).astype(int)
        y_va_eng = np.isin(y_va, [1, 3]).astype(int)
        y_tr_spd = np.isin(y_tr, [2, 3]).astype(int)
        y_va_spd = np.isin(y_va, [2, 3]).astype(int)

        m_eng = lgb.LGBMClassifier(
            n_estimators=1500, learning_rate=0.025, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
            random_state=SEED + fold, n_jobs=-1, verbose=-1,
        )
        m_eng.fit(X_tr_num, y_tr_eng, eval_set=[(X_va_num, y_va_eng)], callbacks=[lgb.early_stopping(50, verbose=False)])

        m_spd = lgb.LGBMClassifier(
            n_estimators=1500, learning_rate=0.025, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=2.0,
            random_state=SEED + fold, n_jobs=-1, verbose=-1,
        )
        m_spd.fit(X_tr_num, y_tr_spd, eval_set=[(X_va_num, y_va_spd)], callbacks=[lgb.early_stopping(50, verbose=False)])

        pe_va = m_eng.predict_proba(X_va_num)[:, 1]
        ps_va = m_spd.predict_proba(X_va_num)[:, 1]
        oof_preds["2AX"][val_idx] = np.column_stack([(1 - pe_va) * (1 - ps_va), pe_va * (1 - ps_va), (1 - pe_va) * ps_va, pe_va * ps_va])

        pe_te = m_eng.predict_proba(X_te_num)[:, 1]
        ps_te = m_spd.predict_proba(X_te_num)[:, 1]
        test_preds["2AX"] += np.column_stack([(1 - pe_te) * (1 - ps_te), pe_te * (1 - ps_te), (1 - pe_te) * ps_te, pe_te * ps_te]) / N_SPLITS

        # 5. Spline GAM (SplineTransformer + MissingIndicator + OneHotEncoder + LogisticRegression)
        pre_gam = ColumnTransformer([
            ("num_spl", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
                ("spl", SplineTransformer(n_knots=8, degree=3, include_bias=False)),
            ]), NUM_SIGNAL),
            ("num_miss", MissingIndicator(features="all"), NUM_SIGNAL),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_ALL),
        ])
        X_tr_gam = pre_gam.fit_transform(X_tr_raw)
        X_va_gam = pre_gam.transform(X_va_raw)
        X_te_gam = pre_gam.transform(X_test_base)

        m_gam = LogisticRegression(C=0.5, max_iter=500, solver="lbfgs")
        m_gam.fit(X_tr_gam, y_tr)
        oof_preds["GAM"][val_idx] = m_gam.predict_proba(X_va_gam)
        test_preds["GAM"] += m_gam.predict_proba(X_te_gam) / N_SPLITS

        # 6. Neural Network (MLPClassifier)
        eng_num_cols = [c for c in X_tr_fe.columns if c not in CAT_ALL]
        pre_mlp = ColumnTransformer([
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
            ]), eng_num_cols),
            ("miss", MissingIndicator(features="all"), NUM_SIGNAL),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_ALL),
        ])
        X_tr_mlp = pre_mlp.fit_transform(X_tr_fe)
        X_va_mlp = pre_mlp.transform(X_va_fe)
        X_te_mlp = pre_mlp.transform(X_te_fe)

        m_mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=0.5, batch_size=512,
            learning_rate_init=1e-3, early_stopping=True, n_iter_no_change=8,
            validation_fraction=0.1, max_iter=60, random_state=SEED + fold,
        )
        m_mlp.fit(X_tr_mlp, y_tr)
        oof_preds["MLP"][val_idx] = m_mlp.predict_proba(X_va_mlp)
        test_preds["MLP"] += m_mlp.predict_proba(X_te_mlp) / N_SPLITS

        fold_scores = " | ".join(
            f"{m}: {f1_score(y_va, oof_preds[m][val_idx].argmax(axis=1), average='macro'):.5f}"
            for m in model_names
        )
        print(f"Fold {fold}/5 ({time.time() - f_start:.1f}s) | {fold_scores}")

    print("\n--- Overall Level-1 OOF Macro F1 ---")
    for m in model_names:
        m_f1 = f1_score(y, oof_preds[m].argmax(axis=1), average="macro")
        w_f1 = f1_score(y, oof_preds[m].argmax(axis=1), average="weighted")
        print(f"  {m:4s} OOF Macro F1 : {m_f1:.5f} (Weighted F1: {w_f1:.5f})")

    # Level-2 Out-of-Fold Stacking Meta-Learner
    stack_X_oof = build_stack_features([oof_preds[m] for m in model_names])
    stack_X_test = build_stack_features([test_preds[m] for m in model_names])

    oof_stack = np.zeros((len(X_base), NUM_CLASSES))
    test_stack = np.zeros((len(X_test_base), NUM_CLASSES))

    for fold, (trn_idx, val_idx) in enumerate(skf.split(stack_X_oof, y), 1):
        meta = LogisticRegression(C=0.2, max_iter=500, solver="lbfgs")
        meta.fit(stack_X_oof[trn_idx], y[trn_idx])
        oof_stack[val_idx] = meta.predict_proba(stack_X_oof[val_idx])
        test_stack += meta.predict_proba(stack_X_test) / N_SPLITS

    raw_stack_f1 = f1_score(y, oof_stack.argmax(axis=1), average="macro")
    print(f"\nLevel-2 Stack OOF Macro F1 (Before Multiplier Tuning): {raw_stack_f1:.5f}")

    # Nested-OOF verification for Class Multiplier Tuning (prevents overfitting)
    nested_oof_preds = np.zeros(len(y), dtype=int)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(stack_X_oof, y), 1):
        mw, _ = optimize_class_multipliers(oof_stack[trn_idx], y[trn_idx])
        nested_oof_preds[val_idx] = (oof_stack[val_idx] * mw).argmax(axis=1)
    nested_f1 = f1_score(y, nested_oof_preds, average="macro")

    # Fit final multipliers on full OOF probabilities
    final_mult, full_tuned_f1 = optimize_class_multipliers(oof_stack, y)
    print(f"Class Multipliers (Coordinate Ascent + Nelder-Mead): {np.round(final_mult, 4)}")
    print(f"Nested-OOF Verified Macro F1 (After Tuning)        : {nested_f1:.5f}")
    print(f"Full OOF Macro F1 (After Tuning)                   : {full_tuned_f1:.5f}")
    print(f"Full OOF Weighted F1                               : {f1_score(y, (oof_stack * final_mult).argmax(axis=1), average='weighted'):.5f}")

    final_oof_preds = (oof_stack * final_mult).argmax(axis=1)
    final_test_preds = (test_stack * final_mult).argmax(axis=1).astype(int)

    print("\n--- Classification Report (OOF, Multiplier-Tuned) ---")
    print(classification_report(y, final_oof_preds, target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)], digits=4))

    sub_path = os.path.join(SUB_DIR, "sub_task2.csv")
    sub_df = pd.DataFrame({
        "id": test_df[ID_COL].values,
        "task2": final_test_preds,
    })
    assert not sub_df.isna().any().any(), "Found NaN in predictions!"
    assert set(sub_df["task2"].unique()).issubset({0, 1, 2, 3}), "Invalid class labels!"
    assert len(sub_df) == 25889, f"Expected 25889 rows, got {len(sub_df)}"
    sub_df.to_csv(sub_path, index=False)

    print(f"✅ Saved Task 2 predictions to: {sub_path} ({len(sub_df):,} rows)")
    print("Predicted Test Class Distribution:")
    for k, v in sub_df["task2"].value_counts().sort_index().items():
        print(f"  Class {int(k)} ({CLASS_NAMES[int(k)]}): {v:,} ({v / len(sub_df) * 100:.2f}%)")
    print(f"⏱️ Total Elapsed Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
