"""
Task 5: Account Security Monitoring (Principled Multi-View Unsupervised Anomaly Ensemble)
Metric: F3-Score (Recall weighted 9x higher than Precision)

Methodology (Train from Scratch, Zero Magic Thresholds):
1. Multi-View Feature Construction from datacard.txt:
   - Profile View: 4-period mean and within-player std for all 29 temporal bases + static traits.
   - Temporal Delta View: Period-to-period deviations (p4-mean123, p3-mean12, p2-p1, p4-p1)
     standardized by each base's robust population MAD of baseline transitions (p2 - p1),
     plus Euclidean geolocation jumps (1->2, 2->3, 3->4, max).
   - Semantic Combat Subspace: Items 12-20 from datacard.txt (skill, KDA, accuracy, reaction, stability).
2. Diverse From-Scratch Unsupervised Model Families:
   - Out-of-Fold Monte Carlo Density-Ratio Estimation (log-odds of real X vs block-permuted X_perm
     across Profile, Mean+Static, Combat Subspace, and Temporal Deltas) to capture joint
     multi-feature anomaly clusters even inside normal marginal bounds.
   - Freedman-Diaconis Smoothed 1D KDE Surprise (max & top-3 per-feature -log p(x)) on continuous
     profile features to detect isolated low-density modes without dilution by irrelevant dimensions.
   - Isolation Forests on both Profile space and Temporal Delta space.
   - Robust Temporal Deviation Scores (L2 norm, L-inf max, and Mahalanobis-style covariance distance).
   - From-scratch Neural Autoencoder (MLPRegressor bottleneck) reconstruction MSE.
3. Normal-Bulk Robust Standardization & Soft-Max/Lp Aggregation:
   - Each detector is standardized via median/MAD, soft-compressed with log1p, and aggregated across
     specialized views using (Max + Top-2 Mean + L3 Norm) / 3 so specialist attack signals are preserved.
4. Principled Unsupervised Bayes-Optimal F3 Thresholding:
   - Fit a 2-component Gaussian Mixture Model (Normal bulk vs Anomaly tail) on the ensemble score
     to obtain posterior anomaly probabilities p_i, then select cutoff k* maximizing expected F3.
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler

SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "task5")
SUB_DIR = os.path.join(BASE_DIR, "submissions")
os.makedirs(SUB_DIR, exist_ok=True)

STATIC_COLS = [
    "account_age_days",
    "behavior_stability",
    "password_changed",
    "vpn_usage",
    "suspicious_login_time",
    "mass_item_sale",
]
COMBAT_BASES = [
    "skill_rating",
    "skill_trajectory",
    "kda_mean",
    "kda_std",
    "accuracy_mean",
    "accuracy_std",
    "reaction_time_mean",
    "reaction_time_std",
    "performance_stability",
]


def get_temporal_bases(df: pd.DataFrame) -> list[str]:
    return sorted({c[:-2] for c in df.columns[1:] if c.endswith(("_1", "_2", "_3", "_4"))})


def build_multiview_features(
    df: pd.DataFrame,
    bases: list[str],
    mad_ref: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Build Profile (mean/std + static) and Temporal Delta (MAD-normalized shifts + geo jumps) views."""
    prof: dict[str, pd.Series | np.ndarray] = {}
    delt: dict[str, pd.Series | np.ndarray] = {}
    new_mad: dict[str, float] = {}

    for b in bases:
        cols = [f"{b}_{i}" for i in range(1, 5)]
        vals = df[cols]
        prof[f"{b}_mean"] = vals.mean(axis=1)
        prof[f"{b}_std"] = vals.std(axis=1)

        if mad_ref is None:
            d21 = (df[f"{b}_2"] - df[f"{b}_1"]).dropna()
            mad21 = float(np.median(np.abs(d21 - np.median(d21))) * 1.4826)
            if mad21 < 1e-6:
                mad21 = float(d21.std() + 1e-6)
            new_mad[b] = mad21
        else:
            mad21 = mad_ref[b]

        delt[f"{b}_d4"] = (df[f"{b}_4"] - df[[f"{b}_1", f"{b}_2", f"{b}_3"]].mean(axis=1)) / mad21
        delt[f"{b}_d3"] = (df[f"{b}_3"] - df[[f"{b}_1", f"{b}_2"]].mean(axis=1)) / mad21
        delt[f"{b}_d2"] = (df[f"{b}_2"] - df[f"{b}_1"]) / mad21
        delt[f"{b}_p4m1"] = (df[f"{b}_4"] - df[f"{b}_1"]) / mad21

    g12 = np.sqrt((df["login_lat_2"] - df["login_lat_1"]) ** 2 + (df["login_lon_2"] - df["login_lon_1"]) ** 2)
    g23 = np.sqrt((df["login_lat_3"] - df["login_lat_2"]) ** 2 + (df["login_lon_3"] - df["login_lon_2"]) ** 2)
    g34 = np.sqrt((df["login_lat_4"] - df["login_lat_3"]) ** 2 + (df["login_lon_4"] - df["login_lon_3"]) ** 2)
    delt["geo_12"] = g12
    delt["geo_23"] = g23
    delt["geo_34"] = g34
    delt["geo_max"] = np.nanmax(np.vstack([g12, g23, g34]), axis=0)

    for c in STATIC_COLS:
        prof[c] = df[c]
    for c in ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]:
        delt[c] = df[c]

    return (
        pd.DataFrame(prof, index=df.index),
        pd.DataFrame(delt, index=df.index),
        new_mad if mad_ref is None else mad_ref,
    )


def generate_block_permuted_null(
    df: pd.DataFrame,
    bases: list[str],
    mad_ref: dict[str, float],
    n_repeats: int = 3,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate independent null copies by permuting each feature's 4-period block across rows."""
    rng = np.random.default_rng(seed)
    perm_profs, perm_delts = [], []
    for _ in range(n_repeats):
        pr = df.iloc[:, 1:].copy()
        for b in bases:
            pcols = [f"{b}_{i}" for i in range(1, 5)]
            idx = rng.permutation(len(pr))
            pr[pcols] = pr[pcols].iloc[idx].to_numpy()
        for c in STATIC_COLS:
            idx = rng.permutation(len(pr))
            pr[c] = pr[c].iloc[idx].to_numpy()
        pp, pdlt, _ = build_multiview_features(pr, bases, mad_ref=mad_ref)
        perm_profs.append(pp)
        perm_delts.append(pdlt)
    return pd.concat(perm_profs, axis=0, ignore_index=True), pd.concat(perm_delts, axis=0, ignore_index=True)


def fit_oof_density_ratio(
    X_real: pd.DataFrame,
    X_perm: pd.DataFrame,
    cols: list[str],
    max_iter: int = 250,
    max_depth: int = 6,
    n_splits: int = 5,
) -> np.ndarray:
    """Out-of-fold density-ratio log-odds log(p(x)/p_indep(x)) via from-scratch HistGBT."""
    Xr = X_real[cols].reset_index(drop=True)
    Xp = X_perm[cols].reset_index(drop=True)
    n = len(Xr)
    m_ratio = len(Xp) // n
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    log_odds = np.zeros(n)

    for tr, va in kf.split(Xr):
        tr_p = np.concatenate([tr + k * n for k in range(m_ratio)])
        X_tr = pd.concat([Xr.iloc[tr], Xp.iloc[tr_p]], axis=0, ignore_index=True)
        y_tr = np.concatenate([np.ones(len(tr), dtype=int), np.zeros(len(tr_p), dtype=int)])
        w_tr = np.concatenate([np.full(len(tr), float(m_ratio)), np.ones(len(tr_p))])
        clf = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=0.06,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=SEED,
        )
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        p = np.clip(clf.predict_proba(Xr.iloc[va])[:, 1], 1e-6, 1.0 - 1e-6)
        log_odds[va] = np.log(p / (1.0 - p))

    return log_odds


def compute_fd_kde_surprise(df_cont: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Freedman-Diaconis smoothed 1D KDE surprise (-log p(x) above feature median)."""
    surp_mat = np.zeros((len(df_cont), df_cont.shape[1]))
    for j, c in enumerate(df_cont.columns):
        v = df_cont[c].to_numpy()
        ok = ~np.isnan(v)
        if df_cont[c].nunique() <= 5:
            continue
        vals_ok = v[ok]
        q25, q75 = np.quantile(vals_ok, [0.25, 0.75])
        iqr = float(q75 - q25 + 1e-9)
        lo, hi = np.quantile(vals_ok, [0.001, 0.999])
        if hi <= lo:
            continue
        h_fd = 2.0 * iqr * (len(vals_ok) ** (-1.0 / 3.0))
        n_bins = int(np.clip(round((hi - lo) / max(h_fd, 1e-9)), 30, 120))
        edges = np.linspace(lo, hi, n_bins + 1)
        vc = np.clip(vals_ok, lo, hi)
        cnts, _ = np.histogram(vc, bins=edges)
        smooth_cnts = gaussian_filter1d(cnts.astype(float), sigma=1.0, mode="nearest")
        probs = (smooth_cnts + 0.5) / (ok.sum() + 0.5 * n_bins)
        b_idx = np.clip(np.searchsorted(edges[1:-1], vc, side="right"), 0, n_bins - 1)
        nlp = -np.log(probs[b_idx])
        surp_mat[ok, j] = np.maximum(0.0, nlp - np.median(nlp))

    kde_max = np.max(surp_mat, axis=1)
    kde_top3 = np.sort(surp_mat, axis=1)[:, -3:].sum(axis=1)
    return kde_max, kde_top3


def to_mad_z(scores: np.ndarray) -> np.ndarray:
    """One-sided robust Z-score relative to median and MAD."""
    med = np.median(scores)
    mad = np.median(np.abs(scores - med)) * 1.4826
    if mad < 1e-6:
        mad = np.std(scores) + 1e-6
    return np.maximum(0.0, (scores - med) / mad)


def bayes_optimal_f3_threshold(scores: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Fit 2-component GMM on ensemble scores and choose cutoff maximizing expected F3."""
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=SEED)
    gmm.fit(scores.reshape(-1, 1))
    anom_comp = int(np.argmax(gmm.means_.ravel()))
    post_prob = gmm.predict_proba(scores.reshape(-1, 1))[:, anom_comp]

    order = np.argsort(post_prob)[::-1]
    p_sorted = post_prob[order]
    p_expected_total = float(p_sorted.sum())
    tp_cum = np.cumsum(p_sorted)
    q_cum = np.arange(1, len(p_sorted) + 1)
    expected_f3 = 10.0 * tp_cum / (9.0 * p_expected_total + q_cum)

    best_k = int(np.argmax(expected_f3)) + 1
    preds = np.zeros(len(scores), dtype=int)
    preds[order[:best_k]] = 1
    return preds, best_k, p_expected_total


def main():
    t0 = time.time()
    print("=" * 72)
    print("Task 5: Principled Multi-View Unsupervised Anomaly Ensemble (F3-Opt)")
    print("=" * 72)

    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample_sub = pd.read_csv(os.path.join(BASE_DIR, "data", "sample_submission.csv"))
    bases = get_temporal_bases(test_df)
    print(f"Loaded test.csv: {test_df.shape[0]:,} rows x {test_df.shape[1]} cols ({len(bases)} temporal bases)")

    # 1. Multi-view feature engineering
    prof_df, delt_df, mad_ref = build_multiview_features(test_df, bases)
    prof_perm_3x, delt_perm_3x = generate_block_permuted_null(test_df, bases, mad_ref, n_repeats=3, seed=SEED)
    n = len(test_df)

    # 2. Out-of-fold Density-Ratio Estimators across semantic subspaces
    combat_mean_cols = [f"{b}_mean" for b in COMBAT_BASES]
    mean_static_cols = [f"{b}_mean" for b in bases] + STATIC_COLS
    prof_all_cols = list(prof_df.columns)
    delt_all_cols = list(delt_df.columns)

    print("Training OOF Monte Carlo Density-Ratio detectors...")
    dr_mean_stat = fit_oof_density_ratio(prof_df, prof_perm_3x.iloc[: 2 * n], mean_static_cols, max_iter=300, max_depth=6)
    dr_prof_all = fit_oof_density_ratio(prof_df, prof_perm_3x.iloc[: 2 * n], prof_all_cols, max_iter=300, max_depth=6)
    dr_combat_m = fit_oof_density_ratio(prof_df, prof_perm_3x, combat_mean_cols, max_iter=250, max_depth=6)
    dr_delt = fit_oof_density_ratio(delt_df, delt_perm_3x.iloc[: 2 * n], delt_all_cols, max_iter=200, max_depth=6)

    # 3. Freedman-Diaconis smoothed 1D KDE surprise
    cont_ms_cols = [f"{b}_mean" for b in bases] + ["account_age_days", "behavior_stability"]
    s_kde_max, s_kde_top3 = compute_fd_kde_surprise(prof_df[cont_ms_cols])

    # 4. Temporal Delta L2 / Max / Isolation Forest
    delt_cont_cols = [c for c in delt_all_cols if c not in ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]]
    X_dc = SimpleImputer(strategy="median").fit_transform(delt_df[delt_cont_cols])
    X_dc_rob = RobustScaler().fit_transform(X_dc)
    s_delt_max = np.max(np.abs(X_dc_rob), axis=1)
    s_delt_l2 = np.sqrt(np.mean(np.clip(X_dc_rob, -50.0, 50.0) ** 2, axis=1))
    iso_delt = IsolationForest(n_estimators=300, max_samples=512, contamination="auto", random_state=SEED, n_jobs=-1)
    iso_delt.fit(np.clip(X_dc_rob, -20.0, 20.0))
    s_iso_delt = -iso_delt.score_samples(np.clip(X_dc_rob, -20.0, 20.0))

    # 5. Profile Isolation Forest & Neural Autoencoder
    X_prof_cont = SimpleImputer(strategy="median").fit_transform(prof_df[cont_ms_cols])
    X_prof_rob = np.clip(RobustScaler().fit_transform(X_prof_cont), -15.0, 15.0)
    iso_prof = IsolationForest(n_estimators=300, max_samples=512, contamination="auto", random_state=SEED, n_jobs=-1)
    iso_prof.fit(X_prof_rob)
    s_iso_prof = -iso_prof.score_samples(X_prof_rob)

    X_comb = StandardScaler().fit_transform(np.hstack([X_prof_rob, np.clip(X_dc_rob, -15.0, 15.0)]))
    ae = MLPRegressor(hidden_layer_sizes=(64, 16, 64), max_iter=50, early_stopping=True, random_state=SEED)
    ae.fit(X_comb, X_comb)
    s_ae = np.mean((X_comb - ae.predict(X_comb)) ** 2, axis=1)

    # 6. Multi-view aggregation (preserves specialist view signals via log1p + Max/Top2/L3)
    views = np.vstack([
        to_mad_z(dr_mean_stat),
        to_mad_z(dr_prof_all),
        to_mad_z(dr_combat_m),
        0.5 * to_mad_z(s_kde_max) + 0.5 * to_mad_z(s_kde_top3),
        to_mad_z(s_iso_prof),
        (to_mad_z(s_delt_l2) + to_mad_z(s_delt_max) + to_mad_z(s_iso_delt) + to_mad_z(dr_delt)) / 4.0,
        to_mad_z(s_ae),
    ])
    v_log = np.log1p(views)
    s_max = np.max(v_log, axis=0)
    s_top2 = np.sort(v_log, axis=0)[-2:, :].mean(axis=0)
    s_lp3 = (np.mean(v_log ** 3, axis=0)) ** (1.0 / 3.0)
    final_score = (s_max + s_top2 + s_lp3) / 3.0

    # 7. Principled 2-GMM Bayes-Optimal F3 Thresholding
    task5_pred, best_k, p_exp_total = bayes_optimal_f3_threshold(final_score)
    print(f"2-GMM estimated anomaly mass: {p_exp_total:.1f} ({p_exp_total / n * 100:.2f}%)")
    print(f"Bayes-Optimal F3 cutoff: {best_k:,} accounts ({best_k / n * 100:.2f}%)")

    # 8. Unsupervised structural audit across discovered anomaly signatures
    m_act = test_df[[f"activity_level_{i}" for i in range(1, 5)]].mean(axis=1)
    m_comm = test_df[[f"communication_frequency_{i}" for i in range(1, 5)]].mean(axis=1)
    m_ip = test_df[[f"ip_hash_entropy_{i}" for i in range(1, 5)]].mean(axis=1)
    geo23 = np.sqrt((test_df["login_lat_3"] - test_df["login_lat_2"]) ** 2 + (test_df["login_lon_3"] - test_df["login_lon_2"]) ** 2)
    geo34 = np.sqrt((test_df["login_lat_4"] - test_df["login_lat_3"]) ** 2 + (test_df["login_lon_4"] - test_df["login_lon_3"]) ** 2)
    d_cur = test_df["currency_transfers_4"] - test_df[[f"currency_transfers_{i}" for i in range(1, 4)]].mean(axis=1)
    d_st = test_df["skill_trajectory_4"] - test_df[[f"skill_trajectory_{i}" for i in range(1, 4)]].mean(axis=1)
    acc_s = test_df[[f"accuracy_std_{i}" for i in range(1, 5)]].mean(axis=1)
    kda_s = test_df[[f"kda_std_{i}" for i in range(1, 5)]].mean(axis=1)
    perf = test_df[[f"performance_stability_{i}" for i in range(1, 5)]].mean(axis=1)

    print("\nDiagnostic Capture Rates by Structural Signature:")
    for name, mask in [
        ("Bot Farm (ip<0.25)", m_ip < 0.25),
        ("Low Activity (<0.35)", m_act < 0.35),
        ("No Comm (<0.15)", m_comm < 0.15),
        ("Low Stab (<0.70)", test_df["behavior_stability"] < 0.70),
        ("Theft (d_cur>6|geo34>4)", (d_cur > 6.0) | (geo34 > 4.0)),
        ("Boost (d_st>10)", d_st > 10.0),
        ("VPN (geo23>4)", geo23 > 4.0),
        ("Combat Var Corner", (acc_s > 9.0) & (kda_s > 0.70) & (perf > 0.87)),
    ]:
        m = mask.fillna(False).to_numpy()
        print(f"  {name:26s}: {task5_pred[m].mean() * 100:5.1f}% ({task5_pred[m].sum():,}/{m.sum():,})")

    out_path = os.path.join(SUB_DIR, "sub_task5.csv")
    out_df = pd.DataFrame({
        "id": sample_sub["id"].to_numpy(),
        "task5": task5_pred,
    })
    assert len(out_df) == 25889
    assert set(out_df["task5"].unique()).issubset({0, 1})
    assert not out_df.isna().any().any()
    out_df.to_csv(out_path, index=False)

    print(f"\nSaved Task 5 predictions to: {out_path}")
    print(f"Flagged Anomalies (1): {task5_pred.sum():,} ({task5_pred.mean() * 100:.2f}%)")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
