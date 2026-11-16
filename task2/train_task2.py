"""
Train Task 2: Player Segment Classification (4-class)
Models: LightGBM + XGBoost + CatBoost (5-Fold Stratified CV)
"""
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

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


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = set(df.columns)

    # Drop explicit random noise columns documented in datacard.txt
    for noise_col in ["random_metric_1", "random_metric_2", "random_metric_3"]:
        if noise_col in df.columns:
            df.drop(columns=[noise_col], inplace=True)

    # 1. Playtime & Activity Ratios
    if {"play_frequency", "avg_session_duration"}.issubset(cols):
        df["weekly_play_minutes"] = df["play_frequency"] * df["avg_session_duration"]
        df["weekly_play_hours"] = df["weekly_play_minutes"] / 60.0

    if {"total_playtime_hours", "account_age_days"}.issubset(cols):
        df["avg_daily_play_hours"] = df["total_playtime_hours"] / (df["account_age_days"] + 1.0)

    if {"login_streak", "account_age_days"}.issubset(cols):
        df["streak_to_age_ratio"] = df["login_streak"] / (df["account_age_days"] + 1.0)

    if {"avg_session_duration", "avg_match_length"}.issubset(cols):
        df["matches_per_session"] = df["avg_session_duration"] / (df["avg_match_length"] + 1.0)

    # 2. Spending & Whale Indicators
    if {"avg_monthly_spending", "weekly_play_hours"}.issubset(set(df.columns)):
        df["spend_per_weekly_hour"] = df["avg_monthly_spending"] / (df["weekly_play_hours"] + 0.1)

    if {"total_spending_thb", "total_playtime_hours"}.issubset(cols):
        df["lifetime_spend_per_hour"] = df["total_spending_thb"] / (df["total_playtime_hours"] + 1.0)

    if {"avg_monthly_spending", "spending_frequency"}.issubset(cols):
        df["avg_transaction_value"] = df["avg_monthly_spending"] / (df["spending_frequency"] + 0.1)

    if {"avg_monthly_spending", "total_spending_thb"}.issubset(cols):
        df["monthly_to_total_spend_ratio"] = df["avg_monthly_spending"] / (df["total_spending_thb"] + 1.0)

    if {"rare_items_count", "collection_progress"}.issubset(cols):
        df["rare_item_density"] = df["rare_items_count"] / (df["collection_progress"] + 0.01)

    # 3. Social Indicators
    if {"friend_count", "total_playtime_hours"}.issubset(cols):
        df["friends_per_hour"] = df["friend_count"] / (df["total_playtime_hours"] + 1.0)

    if {"friend_invites_sent", "friend_count"}.issubset(cols):
        df["invite_to_friend_ratio"] = df["friend_invites_sent"] / (df["friend_count"] + 1.0)

    if {"gifts_sent_received", "friend_count"}.issubset(cols):
        df["gifts_per_friend"] = df["gifts_sent_received"] / (df["friend_count"] + 1.0)

    if {"team_play_percentage", "chat_activity_score"}.issubset(cols):
        df["social_engagement_index"] = df["team_play_percentage"] * df["chat_activity_score"]

    # 4. Grinder / Competitive Indicators
    if {"ranked_participation_rate", "win_rate_ranked"}.issubset(cols):
        df["competitive_efficiency"] = df["ranked_participation_rate"] * df["win_rate_ranked"]

    if {"achievement_completion_rate", "total_playtime_hours"}.issubset(cols):
        df["achievement_speed"] = df["achievement_completion_rate"] / (df["total_playtime_hours"] + 1.0)

    # 5. Missing value count per player
    df["missing_count"] = df.isna().sum(axis=1)

    return df


def optimize_class_multipliers(oof_probs: np.ndarray, y_true: np.ndarray):
    """Find class probability multipliers that maximize Macro F1."""
    best_weights = np.ones(NUM_CLASSES)
    best_score = f1_score(y_true, oof_probs.argmax(axis=1), average="macro")

    def loss_func(weights):
        scaled = oof_probs * weights
        preds = scaled.argmax(axis=1)
        return -f1_score(y_true, preds, average="macro")

    res = minimize(
        loss_func,
        x0=np.ones(NUM_CLASSES),
        method="Nelder-Mead",
        options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-5},
    )
    if -res.fun > best_score:
        best_score = -res.fun
        best_weights = res.x
    return best_weights, best_score


def main():
    t0 = time.time()
    print("=" * 70)
    print("🚀 Starting Task 2 Training Pipeline (Player Segment Classification)")
    print("=" * 70)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Loaded Train: {train_df.shape} | Test: {test_df.shape}")

    ID_COL = "id" if "id" in train_df.columns else train_df.columns[0]
    TARGET_COL = "segment"

    print("\n--- Class Distribution in Train ---")
    vc = train_df[TARGET_COL].value_counts().sort_index()
    for k, v in vc.items():
        print(f"  Class {int(k)} ({CLASS_NAMES.get(int(k), k)}): {v:,} ({v/len(train_df)*100:.2f}%)")

    # Feature Engineering
    train_fe = create_features(train_df)
    test_fe = create_features(test_df)

    feature_cols = [c for c in train_fe.columns if c not in [ID_COL, TARGET_COL]]
    cat_cols = train_fe[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        trn_vals = train_fe[col].fillna("MISSING").astype(str)
        le.fit(trn_vals)
        mapping = {label: idx for idx, label in enumerate(le.classes_)}
        train_fe[col] = trn_vals.map(mapping).fillna(-1).astype(int)
        test_fe[col] = test_fe[col].fillna("MISSING").astype(str).map(mapping).fillna(-1).astype(int)

    X = train_fe[feature_cols].copy()
    y = train_fe[TARGET_COL].astype(int).values
    X_test = test_fe[feature_cols].copy()

    print(f"\nTotal Engineered Features: {len(feature_cols)} (Categorical: {len(cat_cols)})")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_lgb = np.zeros((len(X), NUM_CLASSES))
    oof_xgb = np.zeros((len(X), NUM_CLASSES))
    oof_cat = np.zeros((len(X), NUM_CLASSES))

    test_lgb = np.zeros((len(X_test), NUM_CLASSES))
    test_xgb = np.zeros((len(X_test), NUM_CLASSES))
    test_cat = np.zeros((len(X_test), NUM_CLASSES))

    print("\n--- Training 5-Fold Stratified Cross-Validation ---")
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
        f_start = time.time()
        X_trn, y_trn = X.iloc[trn_idx], y[trn_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # 1. LightGBM
        model_lgb = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=NUM_CLASSES,
            n_estimators=1500,
            learning_rate=0.04,
            num_leaves=63,
            subsample_for_bin=50000,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=SEED + fold,
            n_jobs=-1,
            verbose=-1,
        )
        model_lgb.fit(
            X_trn, y_trn,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof_lgb[val_idx] = model_lgb.predict_proba(X_val)
        test_lgb += model_lgb.predict_proba(X_test) / N_SPLITS

        # 2. XGBoost
        model_xgb = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            eval_metric="mlogloss",
            n_estimators=1500,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            early_stopping_rounds=50,
            random_state=SEED + fold,
            n_jobs=-1,
        )
        model_xgb.fit(
            X_trn, y_trn,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        oof_xgb[val_idx] = model_xgb.predict_proba(X_val)
        test_xgb += model_xgb.predict_proba(X_test) / N_SPLITS

        # 3. CatBoost
        model_cat = CatBoostClassifier(
            loss_function="MultiClass",
            classes_count=NUM_CLASSES,
            iterations=1200,
            learning_rate=0.06,
            depth=6,
            random_seed=SEED + fold,
            thread_count=-1,
            verbose=False,
        )
        model_cat.fit(
            X_trn, y_trn,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_cat[val_idx] = model_cat.predict_proba(X_val)
        test_cat += model_cat.predict_proba(X_test) / N_SPLITS

        f1_l = f1_score(y_val, oof_lgb[val_idx].argmax(axis=1), average="macro")
        f1_x = f1_score(y_val, oof_xgb[val_idx].argmax(axis=1), average="macro")
        f1_c = f1_score(y_val, oof_cat[val_idx].argmax(axis=1), average="macro")
        print(f"Fold {fold}/5 ({time.time() - f_start:.1f}s) | Macro F1 -> LGB: {f1_l:.5f} | XGB: {f1_x:.5f} | CAT: {f1_c:.5f}")

    f1_oof_lgb = f1_score(y, oof_lgb.argmax(axis=1), average="macro")
    f1_oof_xgb = f1_score(y, oof_xgb.argmax(axis=1), average="macro")
    f1_oof_cat = f1_score(y, oof_cat.argmax(axis=1), average="macro")

    print("\n--- Overall Out-of-Fold (OOF) Results ---")
    print(f"LightGBM OOF Macro F1 : {f1_oof_lgb:.5f} (Weighted F1: {f1_score(y, oof_lgb.argmax(axis=1), average='weighted'):.5f})")
    print(f"XGBoost  OOF Macro F1 : {f1_oof_xgb:.5f} (Weighted F1: {f1_score(y, oof_xgb.argmax(axis=1), average='weighted'):.5f})")
    print(f"CatBoost OOF Macro F1 : {f1_oof_cat:.5f} (Weighted F1: {f1_score(y, oof_cat.argmax(axis=1), average='weighted'):.5f})")

    # Grid search ensemble weights
    best_f1 = -1.0
    best_weights = (1/3, 1/3, 1/3)
    for w1 in np.linspace(0, 1, 11):
        for w2 in np.linspace(0, 1 - w1, int(round((1 - w1) * 10)) + 1):
            w3 = max(0.0, 1.0 - w1 - w2)
            blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
            score = f1_score(y, blend.argmax(axis=1), average="macro")
            if score > best_f1:
                best_f1 = score
                best_weights = (w1, w2, w3)

    w1, w2, w3 = best_weights
    oof_ens = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
    test_ens = w1 * test_lgb + w2 * test_xgb + w3 * test_cat

    print(f"\nBest Ensemble Weights (LGB, XGB, CAT): ({w1:.2f}, {w2:.2f}, {w3:.2f})")
    print(f"Ensemble OOF Macro F1 (Before Threshold Tuning): {best_f1:.5f}")

    # Class Multiplier Tuning for Macro F1
    mult, tuned_f1 = optimize_class_multipliers(oof_ens, y)
    print(f"Class Multipliers: {np.round(mult, 4)}")
    print(f"Ensemble OOF Macro F1 (After Multiplier Tuning) : {tuned_f1:.5f}")
    print(f"Ensemble OOF Weighted F1                        : {f1_score(y, (oof_ens * mult).argmax(axis=1), average='weighted'):.5f}")

    final_oof_preds = (oof_ens * mult).argmax(axis=1)
    final_test_preds = (test_ens * mult).argmax(axis=1).astype(int)

    print("\n--- Classification Report (OOF) ---")
    print(classification_report(y, final_oof_preds, target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)], digits=4))

    sub_path = os.path.join(SUB_DIR, "sub_task2.csv")
    sub_df = pd.DataFrame({
        "id": test_df[ID_COL].values,
        "task2": final_test_preds,
    })
    sub_df.to_csv(sub_path, index=False)

    print(f"✅ Saved Task 2 predictions to: {sub_path} ({len(sub_df):,} rows)")
    print("Predicted Test Class Distribution:")
    for k, v in sub_df["task2"].value_counts().sort_index().items():
        print(f"  Class {int(k)} ({CLASS_NAMES.get(int(k), k)}): {v:,} ({v/len(sub_df)*100:.2f}%)")
    print(f"⏱️ Total Elapsed Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
