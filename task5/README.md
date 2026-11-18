# Task 5: Account Security Monitoring (Unsupervised Anomaly Detection)

## Overview
- **Goal:** Detect anomalous/compromised player accounts (`0 = Normal`, `1 = Anomalous`) from `data/task5/test.csv` (`25,889 rows x 123 columns`) using strictly unsupervised methods trained from scratch (`scikit-learn`, `scipy`, `numpy`, `pandas`).
- **Metric:** $F_3 = \frac{10 \cdot P \cdot R}{9P + R}$ (weights Recall 9x higher than Precision).

## Key Discoveries from Statistical & Structural EDA
1. **8 Distinct Anomaly Mechanisms (~8.7% total mass):**
   - **Static/Profile Clusters:** 24/7 Bot Farms (`ip_hash_entropy`, `login_time_variance`, `session_duration_std`, `device_count_std=0`), Low-Activity/High-Skill/High-Spend accounts, Zero-Communication Friend/Guild Spammers, and Low `behavior_stability` compromised buyers.
   - **Temporal Jump Clusters (Period 3 & 4):** Account Theft (`currency_transfers`, `play_pattern_consistency`, `geo34`, `password_changed`, `mass_item_sale`), Boosting/Account Selling (`skill_trajectory`, `skill_rating`, `friend_removes`, `kda_mean`, `accuracy_mean`, `device_count_4==4`, `guild_changes_4==4`), and VPN Regional Exploit (`geo23`, `vpn_usage`, `purchase_frequency`, `purchase_amount_mean`).
   - **High-Dimensional Interaction Cluster:** Elevated joint `(accuracy_std, kda_std, performance_stability)` inside marginal bounds.
2. **Why Previous Approaches Underperformed:**
   - Hardcoded `5.69%` (`r2_sub_task5.csv`) missed static/interaction clusters (e.g. Bot Farm 2.4%, Low-Activity 5.7%, Zero-Comm 2.3%).
   - Rank averaging (`v4` at 20%, `draft3` at 35%) diluted specialist signals when an anomaly only appeared in one subspace, and heavily weighted noisy background flags (`password_changed ~ 5%`, `vpn_usage ~ 15%`), requiring 35% contamination just to recover recall.

## Upgraded Principled Pipeline (`train_task5.py`)
1. **Multi-View Feature Engineering (`datacard.txt`-derived):**
   - **Profile View:** 4-period means and within-player standard deviations for all 29 temporal bases + 6 static attributes.
   - **Temporal Delta View:** Period-to-period deviations (`p4 - mean(p1..3)`, `p3 - mean(p1..2)`, `p2 - p1`, `p4 - p1`) scaled by each base's robust population MAD of `p2 - p1`, plus Euclidean geolocation jumps (`geo_12`, `geo_23`, `geo_34`, `geo_max`).
2. **Diverse From-Scratch Unsupervised Detectors:**
   - **OOF Monte Carlo Density-Ratio Estimation (`HistGradientBoostingClassifier` on real vs. block-permuted independent null copies)** across Profile, Mean+Static, Semantic Combat Subspace, and Temporal Deltas.
   - **Freedman-Diaconis Smoothed 1D KDE Surprise** (`max` and `top-3` per-feature $-\log p(x)$) to detect isolated low-density clusters without high-dimensional dilution.
   - **Isolation Forests** on both Profile and Temporal Delta views.
   - **Robust Temporal Deviation Scores** ($L_2$ norm and $L_\infty$ max over MAD-scaled deltas).
   - **From-Scratch Neural Autoencoder** (`MLPRegressor` bottleneck) reconstruction MSE.
3. **Log-Compressed Soft-Max / $L_3$ Aggregation:**
   - Normalizes each detector via median/MAD, applies $\log(1 + z)$, and combines specialized views via $(\text{Max} + \text{Top2Mean} + L_3) / 3$.
4. **Unsupervised Bayes-Optimal $F_3$ Thresholding:**
   - Fits a 2-component Gaussian Mixture Model on the final ensemble score (estimated anomaly mass `~8.74%`) and selects the cutoff maximizing expected $F_3$ (`2,556` flagged accounts, `9.87%`).

## How to Run
```powershell
.\venv\Scripts\python.exe task5\train_task5.py
.\venv\Scripts\python.exe merge_submission.py
```
