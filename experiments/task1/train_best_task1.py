import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 TRAINING ULTIMATE TASK 1 PIPELINE (OOF F2 > 0.8371)")
print("="*60)

# 1. Load Data
train_df = pd.read_csv("data/task1/train.csv")
test_df = pd.read_csv("data/task1/test.csv")
sample_sub = pd.read_csv("data/sample_submission.csv")

labeled_df = train_df.dropna(subset=["is_cheater"]).copy()
unlabeled_df = train_df[train_df["is_cheater"].isna()].copy()
labeled_df["is_cheater"] = labeled_df["is_cheater"].astype(int)

print(f"Dataset summary:")
print(f" - Labeled Train:   {len(labeled_df):,} rows")
print(f" - Unlabeled Train: {len(unlabeled_df):,} rows (for Pseudo-Labeling)")
print(f" - Test Set:        {len(test_df):,} rows")

# 2. Advanced Feature Engineering
def advanced_feature_engineering(df):
    data = df.copy()

    # Base domain features
    data['aimbot_index'] = data['headshot_percentage'] * data['crosshair_placement']
    data['recoil_efficiency'] = data['spray_control_score'] * data['accuracy_score']
    data['aim_vs_smoothness'] = data['accuracy_score'] / (data['aiming_smoothness'] + 1e-4)
    data['reaction_speed_inv'] = 1000.0 / (data['reaction_time_ms'] + 1e-4)
    data['reaction_vs_accuracy'] = data['accuracy_score'] / (data['reaction_time_ms'] + 1e-4)
    
    # Inhuman mechanics & anomaly flags
    data['inhuman_reaction_hs'] = ((data['reaction_time_ms'] < 130) & (data['headshot_percentage'] > 50)).astype(float)
    data['inhuman_aim_score'] = (data['accuracy_score'] * data['spray_control_score'] * data['crosshair_placement'])
    data['reaction_aim_product'] = (data['accuracy_score'] * data['headshot_percentage']) / (data['reaction_time_ms'] + 1.0)

    # Mechanics vs Tactics Index
    data['mechanics_index'] = (data['accuracy_score'] + data['spray_control_score'] + data['crosshair_placement']) / 3.0
    data['tactics_index'] = (data['game_sense_score'] + data['team_play_score'] + data['buy_decision_score'] + data['map_knowledge']) / 4.0
    data['mech_vs_tactics'] = data['mechanics_index'] / (data['tactics_index'] + 1e-4)
    data['aim_vs_gamesense'] = data['accuracy_score'] / (data['game_sense_score'] + 1e-4)

    # Combat & Efficiency
    data['damage_per_survival'] = data['damage_per_round'] / (data['survival_time_avg'] + 1.0)
    data['combat_power'] = (data['kill_death_ratio'] * data['damage_per_round']) / (data['survival_time_avg'] + 1.0)
    data['clutch_vs_game_sense'] = data['clutch_success_rate'] / (data['game_sense_score'] + 1e-4)
    data['hs_vs_accuracy'] = data['headshot_percentage'] / (data['accuracy_score'] + 1e-4)
    data['damage_per_kdr'] = data['damage_per_round'] / (data['kill_death_ratio'] + 1e-4)
    data['kdr_vs_survival'] = data['kill_death_ratio'] / (data['survival_time_avg'] + 1.0)

    # Smurf & Account Suspicion
    data['suspicion_per_age'] = (data['kill_death_ratio'] * data['reports_received']) / (data['account_age_days'] + 1.0)
    data['kd_per_level'] = data['kill_death_ratio'] / (data['level'] + 1.0)
    data['progression_vs_age'] = data['level_progression_speed'] / (data['account_age_days'] + 1.0)
    data['reports_per_level'] = data['reports_received'] / (data['level'] + 1.0)
    data['reports_per_age'] = data['reports_received'] / (data['account_age_days'] + 1.0)
    data['friends_per_age'] = data['friend_network_size'] / (data['account_age_days'] + 1.0)
    data['friends_per_level'] = data['friend_network_size'] / (data['level'] + 1.0)
    data['device_churn_rate'] = data['device_changes_count'] / (data['account_age_days'] + 1.0)
    data['smurf_risk_composite'] = (data['reports_received'] * data['kill_death_ratio']) / (data['account_age_days'] * (data['friend_network_size'] + 1.0) + 10.0)

    # Missing indicators
    data['missing_feature_count'] = data.isnull().sum(axis=1)
    for col in ['reports_received', 'crosshair_placement', 'friend_network_size', 'account_age_days', 'reaction_time_ms', 'spray_control_score']:
        data[f'{col}_isna'] = data[col].isna().astype(float)

    # Level quantiles
    level_filled = data['level'].fillna(data['level'].median())
    data['level_bin'] = pd.qcut(level_filled, q=10, labels=False, duplicates='drop')

    cols_to_drop = ['player_id', 'is_cheater']
    return data.drop(columns=[c for c in cols_to_drop if c in data.columns])

print("\nExtracting 66 domain features...")
X_labeled = advanced_feature_engineering(labeled_df)
y_labeled = labeled_df['is_cheater'].values
X_unlabeled = advanced_feature_engineering(unlabeled_df)
X_test = advanced_feature_engineering(test_df)
print(f"Total features created: {X_labeled.shape[1]}")

def get_best_f2(y_true, probs, step=0.001):
    best_f2 = 0
    best_t = 0.5
    best_p = 0
    best_r = 0
    thresholds = np.arange(0.15, 0.45, step)
    for t in thresholds:
        preds = (probs >= t).astype(int)
        score = fbeta_score(y_true, preds, beta=2, zero_division=0)
        if score > best_f2:
            best_f2 = score
            best_t = t
            best_p = precision_score(y_true, preds, zero_division=0)
            best_r = recall_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, probs)
    return best_f2, best_t, best_p, best_r, auc

# 3. Stratified 5-Fold Training
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_labeled))
oof_xgb = np.zeros(len(X_labeled))
oof_cat = np.zeros(len(X_labeled))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

unlabeled_lgb = np.zeros(len(X_unlabeled))
unlabeled_xgb = np.zeros(len(X_unlabeled))
unlabeled_cat = np.zeros(len(X_unlabeled))

print("\nPhase 1: Training Initial 5-Fold Ensemble (scale_pos_weight=2.0)...")
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled)):
    X_trn, y_trn = X_labeled.iloc[trn_idx], y_labeled[trn_idx]
    X_val, y_val = X_labeled.iloc[val_idx], y_labeled[val_idx]

    # LightGBM
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=7,
        colsample_bytree=0.8,
        subsample=0.8,
        scale_pos_weight=2.0,
        random_state=42 + fold,
        verbose=-1,
        n_jobs=-1
    )
    clf_lgb.fit(X_trn, y_trn)
    oof_lgb[val_idx] = clf_lgb.predict_proba(X_val)[:, 1]
    test_lgb += clf_lgb.predict_proba(X_test)[:, 1] / 5.0
    unlabeled_lgb += clf_lgb.predict_proba(X_unlabeled)[:, 1] / 5.0

    # XGBoost
    clf_xgb = XGBClassifier(
        n_estimators=450,
        learning_rate=0.03,
        max_depth=6,
        colsample_bytree=0.8,
        subsample=0.8,
        scale_pos_weight=2.0,
        random_state=42 + fold,
        eval_metric='logloss',
        n_jobs=-1
    )
    clf_xgb.fit(X_trn, y_trn)
    oof_xgb[val_idx] = clf_xgb.predict_proba(X_val)[:, 1]
    test_xgb += clf_xgb.predict_proba(X_test)[:, 1] / 5.0
    unlabeled_xgb += clf_xgb.predict_proba(X_unlabeled)[:, 1] / 5.0

    # CatBoost
    clf_cat = CatBoostClassifier(
        iterations=450,
        learning_rate=0.04,
        depth=6,
        auto_class_weights='Balanced',
        random_seed=42 + fold,
        verbose=0,
        thread_count=-1
    )
    clf_cat.fit(X_trn, y_trn)
    oof_cat[val_idx] = clf_cat.predict_proba(X_val)[:, 1]
    test_cat += clf_cat.predict_proba(X_test)[:, 1] / 5.0
    unlabeled_cat += clf_cat.predict_proba(X_unlabeled)[:, 1] / 5.0

    print(f" Fold {fold+1}/5 done.")

f2_l, t_l, p_l, r_l, auc_l = get_best_f2(y_labeled, oof_lgb)
f2_x, t_x, p_x, r_x, auc_x = get_best_f2(y_labeled, oof_xgb)
f2_c, t_c, p_c, r_c, auc_c = get_best_f2(y_labeled, oof_cat)

print(f" Initial LightGBM: F2={f2_l:.5f} (AUC={auc_l:.5f})")
print(f" Initial XGBoost:  F2={f2_x:.5f} (AUC={auc_x:.5f})")
print(f" Initial CatBoost: F2={f2_c:.5f} (AUC={auc_c:.5f})")

# 4. Pseudo-labeling on unlabeled 13,184 rows
unlabeled_blend = 0.45 * unlabeled_lgb + 0.30 * unlabeled_xgb + 0.25 * unlabeled_cat
pos_pseudo = (unlabeled_blend >= 0.96)
neg_pseudo = (unlabeled_blend <= 0.04)
pseudo_mask = pos_pseudo | neg_pseudo

print(f"\nPhase 2: Pseudo-labeling on 13,184 unlabeled rows:")
print(f" - Confident Cheaters (p >= 0.96): {pos_pseudo.sum():,} samples")
print(f" - Confident Legit (p <= 0.04):    {neg_pseudo.sum():,} samples")
print(f" - Total Pseudo samples added:     {pseudo_mask.sum():,} samples")

X_pseudo = X_unlabeled[pseudo_mask].copy()
y_pseudo = (unlabeled_blend[pseudo_mask] >= 0.5).astype(int)

# 5. Retrain LightGBM with Pseudo-Labels
print("\nPhase 3: Retraining LightGBM with Pseudo-Labeled Data...")
oof_lgb_p = np.zeros(len(X_labeled))
test_lgb_p = np.zeros(len(X_test))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled)):
    X_trn_p = pd.concat([X_labeled.iloc[trn_idx], X_pseudo], ignore_index=True)
    y_trn_p = np.concatenate([y_labeled[trn_idx], y_pseudo])
    X_val = X_labeled.iloc[val_idx]

    clf_lgb_p = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=7,
        colsample_bytree=0.8,
        subsample=0.8,
        scale_pos_weight=2.0,
        random_state=42 + fold,
        verbose=-1,
        n_jobs=-1
    )
    clf_lgb_p.fit(X_trn_p, y_trn_p)
    oof_lgb_p[val_idx] = clf_lgb_p.predict_proba(X_val)[:, 1]
    test_lgb_p += clf_lgb_p.predict_proba(X_test)[:, 1] / 5.0
    print(f" Retrain Fold {fold+1}/5 done.")

f2_lp, t_lp, p_lp, r_lp, auc_lp = get_best_f2(y_labeled, oof_lgb_p)
print(f" LightGBM with Pseudo-Labeling: F2={f2_lp:.5f} (AUC={auc_lp:.5f})")

# 6. Optimized Weight Search on Final Ensemble
print("\nPhase 4: Weight Optimization via Nelder-Mead...")
def objective(weights):
    w = np.exp(weights) / np.sum(np.exp(weights))
    blend = w[0] * oof_lgb_p + w[1] * oof_xgb + w[2] * oof_cat
    f2, _, _, _, _ = get_best_f2(y_labeled, blend, step=0.005)
    return -f2

res = minimize(objective, [0.5, 0.3, 0.2], method='Nelder-Mead')
opt_w = np.exp(res.x) / np.sum(np.exp(res.x))
print(f"Optimal Ensemble Weights: LGB_Pseudo={opt_w[0]:.4f}, XGB={opt_w[1]:.4f}, CAT={opt_w[2]:.4f}")

oof_final = opt_w[0] * oof_lgb_p + opt_w[1] * oof_xgb + opt_w[2] * oof_cat
test_final = opt_w[0] * test_lgb_p + opt_w[1] * test_xgb + opt_w[2] * test_cat

# Fine-grained threshold search (step 0.001)
best_f2, best_t, best_p, best_r, auc_final = get_best_f2(y_labeled, oof_final, step=0.001)

print("\n" + "="*60)
print(f"🏆 ULTIMATE TASK 1 BENCHMARK RESULT 🏆")
print(f"5-Fold Out-Of-Fold F2 Score: {best_f2:.5f}")
print(f"ROC-AUC:                     {auc_final:.5f}")
print(f"Optimal Decision Threshold:  {best_t:.4f}")
print(f"Precision at Best Threshold: {best_p:.4f}")
print(f"Recall at Best Threshold:    {best_r:.4f}")
print("="*60)

# 7. Export Final Submission
test_pred_binary = (test_final >= best_t).astype(int)
sub_df = pd.DataFrame({
    "id": sample_sub["id"],
    "task1": test_pred_binary
})

assert len(sub_df) == 25889, "Row count error!"
assert set(sub_df["task1"].unique()).issubset({0, 1}), "Class value error!"
assert sub_df["task1"].isna().sum() == 0, "Missing values detected!"

os.makedirs("submissions", exist_ok=True)
sub_df.to_csv("submissions/sub_task1.csv", index=False)
sub_df.to_csv("submissions/sub_task1_ultimate.csv", index=False)

print(f"\nSaved submissions/sub_task1.csv successfully!")
print(f"Distribution in Test Set (25,889 rows):")
print(f" - Legitimate Players (0): {(test_pred_binary == 0).sum():,} ({(test_pred_binary == 0).mean()*100:.2f}%)")
print(f" - Cheaters (1):           {(test_pred_binary == 1).sum():,} ({(test_pred_binary == 1).mean()*100:.2f}%)")

# Save artifacts
np.save("task1/oof_ultimate_task1.npy", oof_final)
np.save("task1/test_ultimate_task1.npy", test_final)
with open("task1/best_threshold.txt", "w") as f:
    f.write(f"{best_t:.4f}\n")
    f.write(f"f2={best_f2:.5f}\n")
    f.write(f"auc={auc_final:.5f}\n")
    f.write(f"precision={best_p:.4f}\n")
    f.write(f"recall={best_r:.4f}\n")
print("Saved artifacts in task1/ folder successfully.")
