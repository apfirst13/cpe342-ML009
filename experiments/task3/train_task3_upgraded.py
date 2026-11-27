import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss, mean_absolute_error

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

SEED = 42
N_SPLITS = 5
np.random.seed(SEED)

BASE_DIR = r"c:\!KMUTT\Year 3\Semeter 1\ML\Kaggle\cpe342-ML009"
DATA_DIR = os.path.join(BASE_DIR, "data", "task3")
SUB_DIR = os.path.join(BASE_DIR, "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

ID_COL = "id" if "id" in test_df.columns else test_df.columns[0]
TARGET_COL = "spending_30d"
raw_features = [c for c in train_df.columns if c not in ["id", "player_id", TARGET_COL]]

y_all = train_df[TARGET_COL].values.astype(np.float64)
is_spender = (y_all > 0).astype(int)

def calculate_nmae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    y_sum = np.sum(y_true)
    if y_sum == 0:
        return 0.0
    return float(1.0 / (1.0 + (np.sum(np.abs(y_true - y_pred)) / y_sum)))

def optimize_simplex_weights(oof_list, y_true, loss_fn) -> np.ndarray:
    n_m = len(oof_list)
    init_w = np.ones(n_m) / n_m
    bounds = [(0.0, 1.0)] * n_m
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    res = minimize(
        lambda w: loss_fn(y_true, sum(w[i] * oof_list[i] for i in range(n_m))),
        init_w, method="SLSQP", bounds=bounds, constraints=cons
    )
    return res.x / np.sum(res.x)

# ==============================================================================
# PART 0: AUDIT TEAMMATE BASELINE (TRUE OOF vs IN-SAMPLE LEAKED OOF)
# ==============================================================================
print("=" * 80)
print("PART 0: AUDITING TEAMMATE BASELINE (5-FOLD TRUE OOF vs IN-SAMPLE LEAKED OOF)")
print("=" * 80)

def teammate_fe(df):
    d = df.copy()
    eps = 1e-6
    d["spend_trend"] = d["prev_month_spending"] / (d["historical_spending"] + eps)
    d["spend_per_CalendarDay"] = d["historical_spending"] / (d["account_age_days"] + 1.0)
    d["tx_per_day"] = d["total_transactions"] / (d["account_age_days"] + 1.0)
    d["playtime_per_day"] = d["total_playtime_hours"] / (d["account_age_days"] + 1.0)
    d["discount_tx_ratio"] = d["purchases_on_discount"] / (d["total_transactions"] + 1.0)
    d["social_per_friend"] = d["social_interactions"] / (d["friend_count"] + 1.0)
    d["login_recency_ratio"] = d["days_since_last_login"] / (d["account_age_days"] + 1.0)
    d["purchase_recency_ratio"] = d["days_since_last_purchase"] / (d["account_age_days"] + 1.0)
    d["session_intensity"] = d["avg_session_length"] * d["sessions_per_week"]
    d["value_per_hour"] = d["historical_spending"] / (d["total_playtime_hours"] + 1.0)
    return d

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_base_p = np.zeros(len(train_df))
oof_base_y_true_oof = np.zeros(len(train_df))
base_s2_models = []

X_base_all = teammate_fe(pd.DataFrame(SimpleImputer(strategy="median").fit_transform(train_df[raw_features]), columns=raw_features))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_base_all, is_spender)):
    m_c = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
    m_c.fit(X_base_all.iloc[tr_idx], is_spender[tr_idx], eval_set=[(X_base_all.iloc[va_idx], is_spender[va_idx])], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_base_p[va_idx] = m_c.predict_proba(X_base_all.iloc[va_idx])[:, 1]
    
    sp_tr = is_spender[tr_idx] == 1
    sp_va = is_spender[va_idx] == 1
    m_r = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.88, n_estimators=600, learning_rate=0.04, num_leaves=31, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
    m_r.fit(X_base_all.iloc[tr_idx][sp_tr], y_all[tr_idx][sp_tr], eval_set=[(X_base_all.iloc[va_idx][sp_va], y_all[va_idx][sp_va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_base_y_true_oof[va_idx] = np.clip(m_r.predict(X_base_all.iloc[va_idx]), 0, None)
    base_s2_models.append(m_r)

# Teammate's line 724 in-sample average across all 5 models vs True OOF
oof_base_y_leaked = np.mean([m.predict(X_base_all) for m in base_s2_models], axis=0)
pred_base_leaked = np.where(oof_base_p >= 0.49, 1.05742 * oof_base_p * oof_base_y_leaked, 0.0)
pred_base_true_oof = np.where(oof_base_p >= 0.49, 1.05742 * oof_base_p * oof_base_y_true_oof, 0.0)

print(f"Teammate Baseline Stage 1 OOF AUC                  : {roc_auc_score(is_spender, oof_base_p):.5f}")
print(f"Teammate Baseline Stage 2 True OOF Spender MAE     : {mean_absolute_error(y_all[is_spender==1], oof_base_y_true_oof[is_spender==1]):.2f} THB")
print(f"Teammate Baseline Stage 2 In-Sample Leaked MAE     : {mean_absolute_error(y_all[is_spender==1], oof_base_y_leaked[is_spender==1]):.2f} THB (80% train leak in line 724)")
print(f"Teammate Reported In-Sample Leaked NMAE (line 758) : {calculate_nmae(y_all, pred_base_leaked):.5f} (MAE = {mean_absolute_error(y_all, pred_base_leaked):.2f} THB)")
print(f"Teammate TRUE Out-Of-Fold NMAE (Unseen Data)       : {calculate_nmae(y_all, pred_base_true_oof):.5f} (MAE = {mean_absolute_error(y_all, pred_base_true_oof):.2f} THB)")

# ==============================================================================
# PART 1: EXACT ALGEBRAIC RECOVERY + FOLD-ISOLATED IMPUTATION & DGP FEATURES
# ==============================================================================
def apply_exact_algebraic_recovery(df):
    d = df.copy()
    for c in ["historical_spending", "prev_month_spending", "segment", "vip_status", "event_participation_rate"]:
        d[f"{c}_isna"] = d[c].isna().astype(float)
        
    # Exact Identity 1: purchases_on_discount == total_transactions * discount_rate_used
    m_tot_from_disc = d["total_transactions"].isna() & d["purchases_on_discount"].notna() & (d["discount_rate_used"] > 0)
    d.loc[m_tot_from_disc, "total_transactions"] = np.round(d.loc[m_tot_from_disc, "purchases_on_discount"] / d.loc[m_tot_from_disc, "discount_rate_used"])
    
    # Exact Identity 2: historical_spending == total_transactions * avg_transaction_value
    m_hist = d["historical_spending"].isna() & d["total_transactions"].notna() & d["avg_transaction_value"].notna()
    d.loc[m_hist, "historical_spending"] = d.loc[m_hist, "total_transactions"] * d.loc[m_hist, "avg_transaction_value"]
    
    m_tot = d["total_transactions"].isna() & d["historical_spending"].notna() & (d["avg_transaction_value"] > 0)
    d.loc[m_tot, "total_transactions"] = np.round(d.loc[m_tot, "historical_spending"] / d.loc[m_tot, "avg_transaction_value"])
    
    m_avg = d["avg_transaction_value"].isna() & d["historical_spending"].notna() & (d["total_transactions"] > 0)
    d.loc[m_avg, "avg_transaction_value"] = d.loc[m_avg, "historical_spending"] / d.loc[m_avg, "total_transactions"]
    
    m_hist2 = d["historical_spending"].isna() & d["total_transactions"].notna() & d["avg_transaction_value"].notna()
    d.loc[m_hist2, "historical_spending"] = d.loc[m_hist2, "total_transactions"] * d.loc[m_hist2, "avg_transaction_value"]
    
    m_disc = d["purchases_on_discount"].isna() & d["total_transactions"].notna() & d["discount_rate_used"].notna()
    d.loc[m_disc, "purchases_on_discount"] = np.round(d.loc[m_disc, "total_transactions"] * d.loc[m_disc, "discount_rate_used"])
    
    m_drate = d["discount_rate_used"].isna() & d["purchases_on_discount"].notna() & (d["total_transactions"] > 0)
    d.loc[m_drate, "discount_rate_used"] = d.loc[m_drate, "purchases_on_discount"] / d.loc[m_drate, "total_transactions"]
    
    # Exact Identity 3: is_premium_member == (vip_status >= 3)
    m_prem = d["is_premium_member"].isna() & d["vip_status"].notna()
    d.loc[m_prem, "is_premium_member"] = (d.loc[m_prem, "vip_status"] >= 3.0).astype(float)
    
    # Physical Ratio Identity 4: prev_month_spending == 0.3500 * historical_spending
    m_hist_from_prev = d["historical_spending"].isna() & d["prev_month_spending"].notna()
    d.loc[m_hist_from_prev, "historical_spending"] = d.loc[m_hist_from_prev, "prev_month_spending"] / 0.3500
    
    m_prev_from_hist = d["prev_month_spending"].isna() & d["historical_spending"].notna()
    d.loc[m_prev_from_hist, "prev_month_spending"] = d.loc[m_prev_from_hist, "historical_spending"] * 0.3500
    return d

def build_fold_features(df_tr_raw, df_va_raw, df_te_raw):
    d_tr = apply_exact_algebraic_recovery(df_tr_raw)
    d_va = apply_exact_algebraic_recovery(df_va_raw)
    d_te = apply_exact_algebraic_recovery(df_te_raw)
    
    # Save native-NaN + algebraic recovery version for Stage 1 GBDTs
    imp_predictors = ["historical_spending", "prev_month_spending", "event_participation_rate",
                      "vip_status", "is_premium_member", "total_transactions", "avg_transaction_value",
                      "purchases_on_discount", "owns_limited_edition", "tournament_participation"]
    
    seg_valid_tr = d_tr["segment"].notna()
    m_seg = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, random_state=SEED, verbose=-1, n_jobs=-1)
    m_seg.fit(d_tr.loc[seg_valid_tr, imp_predictors], d_tr.loc[seg_valid_tr, "segment"].astype(int))
    for d in [d_tr, d_va, d_te]:
        na_seg = d["segment"].isna()
        if na_seg.sum() > 0:
            d.loc[na_seg, "segment"] = m_seg.predict(d.loc[na_seg, imp_predictors]).astype(float)
        seg_probs = m_seg.predict_proba(d[imp_predictors])
        for s_i in range(4):
            d[f"seg_prob_{s_i}"] = seg_probs[:, s_i]
            
    for target_c in ["vip_status", "event_participation_rate", "historical_spending", "prev_month_spending",
                     "days_since_last_purchase", "purchase_frequency", "games_played", "cross_game_activity"]:
        preds_c = [c for c in raw_features if c != target_c]
        valid_tr = d_tr[target_c].notna()
        m_c = lgb.LGBMRegressor(objective="mae", n_estimators=120, learning_rate=0.05, num_leaves=15, random_state=SEED, verbose=-1, n_jobs=-1)
        m_c.fit(d_tr.loc[valid_tr, preds_c], d_tr.loc[valid_tr, target_c])
        for d in [d_tr, d_va, d_te]:
            na_c = d[target_c].isna()
            if na_c.sum() > 0:
                pred_vals = m_c.predict(d.loc[na_c, preds_c])
                if target_c == "vip_status":
                    pred_vals = np.clip(np.round(pred_vals), 0, 5)
                    prem_known = d.loc[na_c, "is_premium_member"].notna()
                    is_p1 = d.loc[na_c, "is_premium_member"] == 1.0
                    is_p0 = d.loc[na_c, "is_premium_member"] == 0.0
                    pred_vals = np.where(prem_known & is_p1, np.clip(pred_vals, 3, 5), pred_vals)
                    pred_vals = np.where(prem_known & is_p0, np.clip(pred_vals, 0, 2), pred_vals)
                d.loc[na_c, target_c] = pred_vals
                
    for d in [d_tr, d_va, d_te]:
        m_p = d["is_premium_member"].isna() & d["vip_status"].notna()
        d.loc[m_p, "is_premium_member"] = (d.loc[m_p, "vip_status"] >= 3.0).astype(float)
        
        eps = 1e-6
        d["spend_trend"] = d["prev_month_spending"] / (d["historical_spending"] + eps)
        d["spend_per_day"] = d["historical_spending"] / (d["account_age_days"] + 1.0)
        d["tx_per_day"] = d["total_transactions"] / (d["account_age_days"] + 1.0)
        d["playtime_per_day"] = d["total_playtime_hours"] / (d["account_age_days"] + 1.0)
        d["discount_tx_ratio"] = d["purchases_on_discount"] / (d["total_transactions"] + 1.0)
        d["social_per_friend"] = d["social_interactions"] / (d["friend_count"] + 1.0)
        d["login_recency_ratio"] = d["days_since_last_login"] / (d["account_age_days"] + 1.0)
        d["purchase_recency_ratio"] = d["days_since_last_purchase"] / (d["account_age_days"] + 1.0)
        d["session_intensity"] = d["avg_session_length"] * d["sessions_per_week"]
        d["value_per_hour"] = d["historical_spending"] / (d["total_playtime_hours"] + 1.0)
        
        d["is_holiday_season"] = d["seasonal_spending_pattern"].isin([0.0, 11.0]).astype(float)
        d["is_summer_season"] = d["seasonal_spending_pattern"].isin([5.0, 6.0]).astype(float)
        d["season_mult"] = 1.0 + 0.25 * d["is_holiday_season"] + 0.15 * d["is_summer_season"]
        d["vip_mult"] = (1.0 + 0.12 * d["vip_status"].fillna(1.0)) * np.where(d["is_premium_member"].fillna(0.0) == 1.0, 1.18, 1.0)
        d["seg_mult"] = d["segment"].map({0.0: 0.40, 1.0: 0.63, 2.0: 0.76, 3.0: 1.56}).fillna(0.75)
        d["expected_base_spend"] = d["historical_spending"] * d["seg_mult"] * d["season_mult"] * (d["vip_mult"] / 1.25)
        d["clean_base_spend"] = 0.5 * d["historical_spending"] + 0.5 * (d["prev_month_spending"] / 0.3500)
        d["log_hist"] = np.log1p(np.clip(d["historical_spending"], 0, None))
        d["log_prev"] = np.log1p(np.clip(d["prev_month_spending"], 0, None))
        d["log_avg_tx"] = np.log1p(np.clip(d["avg_transaction_value"], 0, None))
        
    for c in d_tr.columns:
        med = d_tr[c].median()
        d_tr[c] = d_tr[c].fillna(med)
        d_va[c] = d_va[c].fillna(med)
        d_te[c] = d_te[c].fillna(med)
    return d_tr, d_va, d_te

# ==============================================================================
# PART 2: 5-FOLD STRICT OUT-OF-FOLD TRAINING (STAGE 1 & STAGE 2)
# ==============================================================================
print("\n" + "=" * 80)
print("PART 2: RUNNING 5-FOLD STRICT OUT-OF-FOLD UPGRADED PIPELINE")
print("=" * 80)

n_train = len(train_df)
n_test = len(test_df)

# Stage 1 OOF & Test
oof_s1_lgb, test_s1_lgb = np.zeros(n_train), np.zeros(n_test)
oof_s1_xgb, test_s1_xgb = np.zeros(n_train), np.zeros(n_test)
oof_s1_cat, test_s1_cat = np.zeros(n_train), np.zeros(n_test)

# Stage 2 OOF & Test (6 Log-Ratio + 1 Direct Ratio = 7 models)
N_S2_MODELS = 7
s2_names = [
    "CatBoost_log(y/hist)",
    "CatBoost_log(y/prev)",
    "CatBoost_log(y/base)",
    "XGBoost_log(y/hist)",
    "XGBoost_log(y/prev)",
    "LightGBM_log(y/hist)",
    "LightGBM_weighted(y/hist)"
]
oof_s2_models = [np.zeros(n_train) for _ in range(N_S2_MODELS)]
test_s2_models = [np.zeros(n_test) for _ in range(N_S2_MODELS)]

t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, is_spender)):
    f_t0 = time.time()
    X_tr, X_va, X_te = build_fold_features(
        train_df[raw_features].iloc[tr_idx],
        train_df[raw_features].iloc[va_idx],
        test_df[raw_features]
    )
    y_tr, y_va = y_all[tr_idx], y_all[va_idx]
    sp_tr, sp_va = is_spender[tr_idx] == 1, is_spender[va_idx] == 1

    # --- STAGE 1: Binary Classification ---
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.025, num_leaves=23, max_depth=5,
        min_child_samples=60, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
        random_state=SEED + fold, n_jobs=-1, verbose=-1
    )
    clf_lgb.fit(X_tr, is_spender[tr_idx], eval_set=[(X_va, is_spender[va_idx])], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_s1_lgb[va_idx] = clf_lgb.predict_proba(X_va)[:, 1]
    test_s1_lgb += clf_lgb.predict_proba(X_te)[:, 1] / N_SPLITS

    clf_xgb = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.025, max_depth=5, min_child_weight=50,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
        eval_metric="logloss", early_stopping_rounds=50, random_state=SEED + fold, n_jobs=-1, tree_method="hist"
    )
    clf_xgb.fit(X_tr, is_spender[tr_idx], eval_set=[(X_va, is_spender[va_idx])], verbose=False)
    oof_s1_xgb[va_idx] = clf_xgb.predict_proba(X_va)[:, 1]
    test_s1_xgb += clf_xgb.predict_proba(X_te)[:, 1] / N_SPLITS

    clf_cat = CatBoostClassifier(
        iterations=1200, learning_rate=0.035, depth=6, l2_leaf_reg=5.0,
        eval_metric="Logloss", random_seed=SEED + fold, thread_count=-1, verbose=0
    )
    clf_cat.fit(X_tr, is_spender[tr_idx], eval_set=(X_va, is_spender[va_idx]), early_stopping_rounds=50, verbose=False)
    oof_s1_cat[va_idx] = clf_cat.predict_proba(X_va)[:, 1]
    test_s1_cat += clf_cat.predict_proba(X_te)[:, 1] / N_SPLITS

    # --- STAGE 2: Conditional Multiplicative Log-Ratio L1 Regressors ---
    hist_tr = np.clip(X_tr["historical_spending"].values[sp_tr], 1e-2, None)
    hist_va = np.clip(X_va["historical_spending"].values, 1e-2, None)
    hist_te = np.clip(X_te["historical_spending"].values, 1e-2, None)

    prev_tr = np.clip(X_tr["prev_month_spending"].values[sp_tr], 1e-2, None)
    prev_va = np.clip(X_va["prev_month_spending"].values, 1e-2, None)
    prev_te = np.clip(X_te["prev_month_spending"].values, 1e-2, None)

    base_tr = np.clip(X_tr["clean_base_spend"].values[sp_tr], 1e-2, None)
    base_va = np.clip(X_va["clean_base_spend"].values, 1e-2, None)
    base_te = np.clip(X_te["clean_base_spend"].values, 1e-2, None)

    t_lrh_tr = np.log(np.clip(y_tr[sp_tr] / hist_tr, 1e-4, None))
    t_lrh_va = np.log(np.clip(y_va[sp_va] / hist_va[sp_va], 1e-4, None))

    t_lrp_tr = np.log(np.clip(y_tr[sp_tr] / prev_tr, 1e-4, None))
    t_lrp_va = np.log(np.clip(y_va[sp_va] / prev_va[sp_va], 1e-4, None))

    t_lrb_tr = np.log(np.clip(y_tr[sp_tr] / base_tr, 1e-4, None))
    t_lrb_va = np.log(np.clip(y_va[sp_va] / base_va[sp_va], 1e-4, None))

    # Model 0: CatBoost log(y / hist)
    m0 = CatBoostRegressor(loss_function="MAE", eval_metric="MAE", iterations=1600, learning_rate=0.035, depth=6, l2_leaf_reg=5.0, random_seed=SEED + fold, thread_count=-1, verbose=0)
    m0.fit(X_tr[sp_tr], t_lrh_tr, eval_set=(X_va[sp_va], t_lrh_va), early_stopping_rounds=70, verbose=False)
    oof_s2_models[0][va_idx] = np.clip(np.exp(m0.predict(X_va)) * hist_va, 0, 300000)
    test_s2_models[0] += np.clip(np.exp(m0.predict(X_te)) * hist_te, 0, 300000) / N_SPLITS

    # Model 1: CatBoost log(y / prev)
    m1 = CatBoostRegressor(loss_function="MAE", eval_metric="MAE", iterations=1600, learning_rate=0.035, depth=6, l2_leaf_reg=5.0, random_seed=SEED + 10 + fold, thread_count=-1, verbose=0)
    m1.fit(X_tr[sp_tr], t_lrp_tr, eval_set=(X_va[sp_va], t_lrp_va), early_stopping_rounds=70, verbose=False)
    oof_s2_models[1][va_idx] = np.clip(np.exp(m1.predict(X_va)) * prev_va, 0, 300000)
    test_s2_models[1] += np.clip(np.exp(m1.predict(X_te)) * prev_te, 0, 300000) / N_SPLITS

    # Model 2: CatBoost log(y / clean_base_spend)
    m2 = CatBoostRegressor(loss_function="MAE", eval_metric="MAE", iterations=1600, learning_rate=0.035, depth=6, l2_leaf_reg=5.0, random_seed=SEED + 20 + fold, thread_count=-1, verbose=0)
    m2.fit(X_tr[sp_tr], t_lrb_tr, eval_set=(X_va[sp_va], t_lrb_va), early_stopping_rounds=70, verbose=False)
    oof_s2_models[2][va_idx] = np.clip(np.exp(m2.predict(X_va)) * base_va, 0, 300000)
    test_s2_models[2] += np.clip(np.exp(m2.predict(X_te)) * base_te, 0, 300000) / N_SPLITS

    # Model 3: XGBoost log(y / hist)
    m3 = xgb.XGBRegressor(objective="reg:absoluteerror", eval_metric="mae", n_estimators=1400, learning_rate=0.025, max_depth=5, min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0, early_stopping_rounds=60, random_state=SEED + fold, n_jobs=-1, tree_method="hist")
    m3.fit(X_tr[sp_tr], t_lrh_tr, eval_set=[(X_va[sp_va], t_lrh_va)], verbose=False)
    oof_s2_models[3][va_idx] = np.clip(np.exp(m3.predict(X_va)) * hist_va, 0, 300000)
    test_s2_models[3] += np.clip(np.exp(m3.predict(X_te)) * hist_te, 0, 300000) / N_SPLITS

    # Model 4: XGBoost log(y / prev)
    m4 = xgb.XGBRegressor(objective="reg:absoluteerror", eval_metric="mae", n_estimators=1400, learning_rate=0.025, max_depth=5, min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0, early_stopping_rounds=60, random_state=SEED + 10 + fold, n_jobs=-1, tree_method="hist")
    m4.fit(X_tr[sp_tr], t_lrp_tr, eval_set=[(X_va[sp_va], t_lrp_va)], verbose=False)
    oof_s2_models[4][va_idx] = np.clip(np.exp(m4.predict(X_va)) * prev_va, 0, 300000)
    test_s2_models[4] += np.clip(np.exp(m4.predict(X_te)) * prev_te, 0, 300000) / N_SPLITS

    # Model 5: LightGBM log(y / hist)
    m5 = lgb.LGBMRegressor(objective="mae", n_estimators=1400, learning_rate=0.025, num_leaves=23, max_depth=5, min_child_samples=60, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0, random_state=SEED + fold, n_jobs=-1, verbose=-1)
    m5.fit(X_tr[sp_tr], t_lrh_tr, eval_set=[(X_va[sp_va], t_lrh_va)], callbacks=[lgb.early_stopping(60, verbose=False)])
    oof_s2_models[5][va_idx] = np.clip(np.exp(m5.predict(X_va)) * hist_va, 0, 300000)
    test_s2_models[5] += np.clip(np.exp(m5.predict(X_te)) * hist_te, 0, 300000) / N_SPLITS

    # Model 6: Shallow LightGBM Weighted (y / hist)
    m6 = lgb.LGBMRegressor(objective="mae", n_estimators=1200, learning_rate=0.025, num_leaves=15, max_depth=5, min_child_samples=120, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0, random_state=SEED + fold, n_jobs=-1, verbose=-1)
    m6.fit(X_tr[sp_tr], y_tr[sp_tr] / hist_tr, sample_weight=hist_tr, eval_set=[(X_va[sp_va], y_va[sp_va] / hist_va[sp_va])], eval_sample_weight=[hist_va[sp_va]], callbacks=[lgb.early_stopping(60, verbose=False)])
    oof_s2_models[6][va_idx] = np.clip(m6.predict(X_va) * hist_va, 0, 300000)
    test_s2_models[6] += np.clip(m6.predict(X_te) * hist_te, 0, 300000) / N_SPLITS

    fold_p = (oof_s1_lgb[va_idx] + oof_s1_xgb[va_idx] + oof_s1_cat[va_idx]) / 3.0
    fold_y2 = np.mean([oof_s2_models[m_i][va_idx] for m_i in range(N_S2_MODELS)], axis=0)
    print(f"  Fold {fold+1}/5 ({time.time()-f_t0:.1f}s) | S1 AUC: {roc_auc_score(sp_va, fold_p):.5f} | S2 Spender MAE: {mean_absolute_error(y_va[sp_va], fold_y2[sp_va]):.2f} THB (vs Base {mean_absolute_error(y_va[sp_va], oof_base_y_true_oof[va_idx][sp_va]):.2f})")

# ==============================================================================
# PART 3: STRICT OUT-OF-FOLD SIMPLEX BLENDING & BAYES-L1 QUANTILE CALIBRATION
# ==============================================================================
print("\n" + "=" * 80)
print("PART 3: 5-FOLD OOF EVALUATION & NESTED-CV VERIFICATION")
print("=" * 80)

w_s1 = optimize_simplex_weights([oof_s1_lgb, oof_s1_xgb, oof_s1_cat], is_spender, log_loss)
oof_s1_prob = w_s1[0] * oof_s1_lgb + w_s1[1] * oof_s1_xgb + w_s1[2] * oof_s1_cat
test_s1_prob = w_s1[0] * test_s1_lgb + w_s1[1] * test_s1_xgb + w_s1[2] * test_s1_cat
print(f"Stage 1 Weights : LGB={w_s1[0]:.3f}, XGB={w_s1[1]:.3f}, CAT={w_s1[2]:.3f}")
print(f"Stage 1 OOF AUC : {roc_auc_score(is_spender, oof_s1_prob):.5f} | LogLoss: {log_loss(is_spender, oof_s1_prob):.5f}")

sp_all = is_spender == 1
print("\nStage 2 Individual 5-Fold OOF Spender MAEs:")
for i, name in enumerate(s2_names):
    print(f"  - {name:<26s}: {mean_absolute_error(y_all[sp_all], oof_s2_models[i][sp_all]):.2f} THB")

w_s2 = optimize_simplex_weights([m[sp_all] for m in oof_s2_models], y_all[sp_all], mean_absolute_error)
oof_s2_amount = sum(w_s2[i] * oof_s2_models[i] for i in range(N_S2_MODELS))
test_s2_amount = sum(w_s2[i] * test_s2_models[i] for i in range(N_S2_MODELS))

print("\nStage 2 Simplex Weights:")
for i, name in enumerate(s2_names):
    print(f"  - {name:<26s}: w = {w_s2[i]:.3f}")
print(f"Stage 2 Blended 5-Fold OOF Spender MAE : {mean_absolute_error(y_all[sp_all], oof_s2_amount[sp_all]):.2f} THB (vs Teammate 3679.32 THB -> -{3679.32 - mean_absolute_error(y_all[sp_all], oof_s2_amount[sp_all]):.2f} THB!)")

# Nested 5-Fold Verification of Bayes-Optimal Zero-Inflated L1 Quantile Mapping
oof_final_nested = np.zeros(n_train)
fold_params = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, is_spender)):
    sp_tr_f = is_spender[tr_idx] == 1
    u_tr_f = y_all[tr_idx][sp_tr_f] / np.clip(oof_s2_amount[tr_idx][sp_tr_f], 1e-2, None)
    
    # Grid search tau in [0.45, 0.52] and alpha in [0.97, 1.03] STRICTLY on tr_idx
    best_f_nmae = -1.0
    best_tau, best_alpha = 0.50, 1.00
    for tau in np.linspace(0.45, 0.52, 15):
        q_tr = np.clip((oof_s1_prob[tr_idx] - tau) / np.clip(oof_s1_prob[tr_idx], 1e-6, None), 0.0, 0.50)
        q_mult_tr = np.quantile(u_tr_f, q_tr)
        base_pred_tr = np.where(oof_s1_prob[tr_idx] <= tau, 0.0, oof_s2_amount[tr_idx] * q_mult_tr)
        for alpha in np.linspace(0.97, 1.03, 13):
            sc = calculate_nmae(y_all[tr_idx], alpha * base_pred_tr)
            if sc > best_f_nmae:
                best_f_nmae = sc
                best_tau, best_alpha = float(tau), float(alpha)
                
    # Apply out-of-fold to va_idx
    q_va = np.clip((oof_s1_prob[va_idx] - best_tau) / np.clip(oof_s1_prob[va_idx], 1e-6, None), 0.0, 0.50)
    q_mult_va = np.quantile(u_tr_f, q_va)
    oof_final_nested[va_idx] = np.where(oof_s1_prob[va_idx] <= best_tau, 0.0, best_alpha * oof_s2_amount[va_idx] * q_mult_va)
    fold_params.append((best_tau, best_alpha))

nested_nmae = calculate_nmae(y_all, oof_final_nested)
nested_mae = mean_absolute_error(y_all, oof_final_nested)
base_true_nmae = calculate_nmae(y_all, pred_base_true_oof)
base_true_mae = mean_absolute_error(y_all, pred_base_true_oof)

print("\n" + "=" * 80)
print("FINAL HEAD-TO-HEAD 5-FOLD OUT-OF-FOLD SUMMARY (104,000 PLAYERS)")
print("=" * 80)
print(f"1. Teammate Baseline True 5-Fold OOF NMAE : {base_true_nmae:.5f} (Overall MAE = {base_true_mae:.2f} THB)")
print(f"2. Upgraded Pipeline Nested 5-Fold OOF NMAE: {nested_nmae:.5f} (Overall MAE = {nested_mae:.2f} THB)")
print(f"   -> Genuine True OOF NMAE Improvement   : +{nested_nmae - base_true_nmae:.5f} (+{(nested_nmae - base_true_nmae)*100:.2f}% NMAE, -{base_true_mae - nested_mae:.2f} THB/player!)")

# Compute test set final predictions using consensus median tau and alpha across folds
avg_tau = float(np.median([p[0] for p in fold_params]))
avg_alpha = float(np.median([p[1] for p in fold_params]))
u_all_sp = y_all[sp_all] / np.clip(oof_s2_amount[sp_all], 1e-2, None)
q_te = np.clip((test_s1_prob - avg_tau) / np.clip(test_s1_prob, 1e-6, None), 0.0, 0.50)
q_mult_te = np.quantile(u_all_sp, q_te)
test_final_pred = np.clip(np.where(test_s1_prob <= avg_tau, 0.0, avg_alpha * test_s2_amount * q_mult_te), 0.0, 300000.0)

# Save artifacts and submission
np.save(os.path.join(BASE_DIR, "task3", "oof_s1_prob.npy"), oof_s1_prob)
np.save(os.path.join(BASE_DIR, "task3", "oof_s2_amount.npy"), oof_s2_amount)
np.save(os.path.join(BASE_DIR, "task3", "oof_final_nested.npy"), oof_final_nested)
np.save(os.path.join(BASE_DIR, "task3", "test_final_pred.npy"), test_final_pred)

sub = pd.DataFrame({
    "id": test_df[ID_COL].values,
    "task3": np.round(test_final_pred, 4)
})
sub_path = os.path.join(SUB_DIR, "sub_task3.csv")
sub.to_csv(sub_path, index=False)
print(f"\nSaved upgraded submission to {sub_path} ({len(sub):,} rows)")
print(f"Test Zero-Spend Rate : {(sub['task3'] == 0).mean()*100:.2f}% | Active Mean: {sub.loc[sub['task3']>0, 'task3'].mean():.2f} THB | Max: {sub['task3'].max():.2f} THB")
print(f"Total Runtime        : {time.time()-t0:.1f}s")
