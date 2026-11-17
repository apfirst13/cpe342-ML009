"""
Task 5: Account Security Monitoring (Unsupervised Anomaly Detection - v4)
Metric: F3-Score (Recall weighted 9x higher than Precision)

Key Improvements in v4:
1. Focused ~40 high-signal features combining:
   - Static/mean levels (catches 24/7 bot farms & VPN users)
   - High-signal temporal drift/spike features (catches account thefts & account selling)
   - Security red flags (password_changed, mass_item_sale, vpn_usage, suspicious_login_time)
2. Diverse Unsupervised Ensemble (Isolation Forest + PCA Reconstruction + k-NN + Security/Tail Score)
   using Percentile Rank Averaging.
3. Matched 20% contamination rate for clean comparison against v1 (0.740).
"""
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "task5")
SUB_DIR = os.path.join(BASE_DIR, "submissions")
os.makedirs(SUB_DIR, exist_ok=True)


def build_focused_features(df: pd.DataFrame) -> pd.DataFrame:
    fe = pd.DataFrame(index=df.index)

    # 1. Geolocation jumps (consecutive & overall)
    jumps = []
    for a, b in [(1, 2), (2, 3), (3, 4), (1, 4)]:
        d = np.sqrt(
            (df[f"login_lat_{b}"] - df[f"login_lat_{a}"]) ** 2
            + (df[f"login_lon_{b}"] - df[f"login_lon_{a}"]) ** 2
        )
        jumps.append(d)
    jumps_df = pd.concat(jumps, axis=1)
    fe["geo_jump_max"] = jumps_df.max(axis=1)
    fe["geo_jump_sum"] = jumps_df.sum(axis=1)

    # 2. High-signal temporal bases: both MEAN (static/bot level) and RANGE/DRIFT (sudden theft/ring)
    high_signal_bases = [
        "login_count",
        "location_changes",
        "device_count",
        "ip_hash_entropy",
        "login_time_variance",
        "play_pattern_consistency",
        "activity_level",
        "skill_rating",
        "skill_trajectory",
        "kda_mean",
        "accuracy_mean",
        "reaction_time_mean",
        "performance_stability",
        "purchase_frequency",
        "purchase_amount_mean",
        "refund_count",
        "currency_transfers",
        "friend_removes",
        "guild_changes",
    ]
    for b in high_signal_bases:
        cols = [f"{b}_{i}" for i in [1, 2, 3, 4]]
        vals = df[cols]
        fe[f"{b}_mean"] = vals.mean(axis=1)
        fe[f"{b}_range"] = vals.max(axis=1) - vals.min(axis=1)
        fe[f"{b}_p4_m_p1"] = (df[f"{b}_4"] - df[f"{b}_1"]).abs()

    # 3. Static security & stability features
    fe["account_age_days"] = df["account_age_days"]
    fe["inv_behavior_stability"] = 1.0 - df["behavior_stability"]
    fe["password_changed"] = df["password_changed"].fillna(0)
    fe["vpn_usage"] = df["vpn_usage"].fillna(0)
    fe["suspicious_login_time"] = df["suspicious_login_time"].fillna(0)
    fe["mass_item_sale"] = df["mass_item_sale"].fillna(0)
    fe["missing_count"] = df.isna().sum(axis=1)

    return fe


def to_rank_prob(scores: np.ndarray) -> np.ndarray:
    return rankdata(scores, method="average") / len(scores)


def main():
    t0 = time.time()
    print("=" * 70)
    print("🚀 Task 5 (v4): Focused Feature + Security-Weighted Anomaly Ensemble")
    print("=" * 70)

    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample_sub = pd.read_csv(os.path.join(BASE_DIR, "data", "sample_submission.csv"))

    fe = build_focused_features(test_df)
    print(f"Focused features: {fe.shape[1]}")

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(fe)

    scaler = RobustScaler()
    X_scaled = np.clip(scaler.fit_transform(X_imp), -15.0, 15.0)

    # 1. Isolation Forest
    iso = IsolationForest(
        n_estimators=300,
        max_samples=512,
        contamination="auto",
        random_state=SEED,
        n_jobs=-1,
    )
    iso.fit(X_scaled)
    iso_score = -iso.score_samples(X_scaled)

    # 2. PCA Reconstruction Error
    pca = PCA(n_components=15, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    X_rec = pca.inverse_transform(X_pca)
    pca_score = np.mean((X_scaled - X_rec) ** 2, axis=1)

    # 3. k-NN Distance on PCA space
    nn = NearestNeighbors(n_neighbors=15, n_jobs=-1)
    nn.fit(X_pca)
    dists, _ = nn.kneighbors(X_pca)
    knn_score = dists[:, 1:].mean(axis=1)

    # 4. Security Red-Flag + Extreme Outlier Score
    # Weighted domain red flags + geo jump + behavior instability
    sec_score = (
        fe["password_changed"].values * 2.5
        + fe["mass_item_sale"].values * 2.5
        + fe["suspicious_login_time"].values * 1.5
        + fe["vpn_usage"].values * 1.5
        + np.clip(fe["geo_jump_max"].fillna(0.0).values / 5.0, 0.0, 3.0)
        + np.clip((0.72 - test_df["behavior_stability"].fillna(0.82).values) * 10.0, 0.0, 3.0)
    )

    # Rank-averaged soft voting
    r_iso = to_rank_prob(iso_score)
    r_pca = to_rank_prob(pca_score)
    r_knn = to_rank_prob(knn_score)
    r_sec = to_rank_prob(sec_score)

    ensemble_rank = 0.30 * r_iso + 0.25 * r_pca + 0.20 * r_knn + 0.25 * r_sec

    contamination_rate = 0.20
    cutoff = np.quantile(ensemble_rank, 1.0 - contamination_rate)
    task5_pred = (ensemble_rank >= cutoff).astype(int)

    # Diagnostic check
    for name, mask in [
        ("Geo > 5", fe["geo_jump_max"] > 5.0),
        ("BS < 0.70", test_df["behavior_stability"] < 0.70),
        ("PwdChg == 1", test_df["password_changed"] == 1),
        ("MassSale == 1", test_df["mass_item_sale"] == 1),
        ("VPN == 1", test_df["vpn_usage"] == 1),
        ("SuspLogin == 1", test_df["suspicious_login_time"] == 1),
    ]:
        print(f"  Capture {name:15s}: {task5_pred[mask].mean()*100:5.1f}% ({task5_pred[mask].sum():,}/{mask.sum():,})")

    out_path = os.path.join(SUB_DIR, "sub_task5_v4.csv")
    pd.DataFrame({
        "id": sample_sub["id"].values,
        "task5": task5_pred,
    }).to_csv(out_path, index=False)

    print(f"\n✅ Saved Task 5 v4 predictions to: {out_path}")
    print(f"Flagged Anomalies (1): {task5_pred.sum():,} ({task5_pred.mean()*100:.2f}%)")
    print(f"⏱️ Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
