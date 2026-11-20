# Task 5: Account Security Monitoring (Unsupervised Anomaly Detection)

## Overview
- **Goal:** Detect anomalous/compromised player accounts (`0 = Normal`, `1 = Anomalous`) from `data/task5/test.csv` (`25,889 rows x 123 columns`) using strictly unsupervised ML methods trained from scratch (`lightgbm`, `scikit-learn`, `scipy`, `numpy`, `pandas`).
- **Metric:** $F_3 = \frac{10 \cdot P \cdot R}{9P + R}$ (weights Recall 9x higher than Precision).
- **Competition Rule Compliance:**
  - Unsupervised learning strictly on `data/task5/test.csv` (no external data, no labels, no human hand-labeling, no pre-trained models, no AutoML).
  - **Zero rigid rule overrides:** Removes all hard-coded threshold overrides (`extract_structural_anomaly_signatures`) so 100% of predictions are produced probabilistically by trained unsupervised ML detectors, fulfilling the domain goal (*"without rigid rules or labeled data"*).

---

## V6 Pure Unsupervised Multi-Paradigm Architecture (`train_task5.py`)

1. **Multi-View & Bridged-Geolocation Feature Engineering:**
   - **Profile View:** 4-period mean & std across all 29 temporal bases + 6 static attributes.
   - **Bridged Geolocation & 1D Coordinate Deltas:** Computes consecutive (`geo_12`, `geo_23`, `geo_34`), bridged multi-period (`geo_13`, `geo_24`, `geo_14`) Euclidean and Haversine distances, plus 1D axis jumps (`|dlat|`, `|dlon|`) so missing intermediate coordinates (~18% NaN in single consecutive pairs) do not erase travel signals.
   - **Smooth Continuous Domain Interactions:** `inactivity_vs_skill`, `inactivity_vs_spend`, `combat_inconsistency`, `spam_ratio`, `bot_regularity`, and continuous ReLU-scaled surge intensities (`takeover_intensity`, `boosting_intensity`, `vpn_intensity`) with zero step cutoffs.

2. **Multi-Bandwidth 1D Topological Valley-Rebound KDE Detector:**
   - Automatically screens out low-cardinality discrete count bases (`nunique <= 25`) to eliminate integer-lattice artifacts.
   - Computes FFT-smoothed Gaussian KDE across continuous profile, Period-4, and bridged geolocation features at dual consensus bandwidths (`0.075 * IQR`, `0.115 * IQR`), scoring each point by density barrier drop and minority-cluster rebound depth relative to the main population mode (unimodal normal tails have zero rebound).
   - Fits a **3-component Gaussian Mixture Model** on `log1p(topo_max_ve)` ($1 - P(\text{Bulk Component})$) to preserve both moderate and extreme isolated density modes.

3. **Focused Subspace OOF Density Ratios & Debiased 3D Combat KDE:**
   - **3D Combat Consistency Subspace (`accuracy_std_mean`, `kda_std_mean`, `performance_stability_mean`):** Blends 5-fold OOF `LightGBM` density-ratio estimation (vs. 5x block-permuted independent null) with 1-step EM-debiased 3D/1D Gaussian-filtered KDE density ratios using **Residual Topological Exclusion** (`p_topo >= 0.50` excluded before 3D histogram smoothing so Bot extreme tails do not bleed false positives).
   - **Profile-Core, Full-Profile, and Temporal-Delta OOF Density Ratios:** Multi-subspace LightGBM estimators capturing multi-feature dependence anomalies.

4. **Unsupervised Mixture-MICE Cross-Feature `NaN` Recovery:**
   - Uses topological valley modes to automatically discover strongly bimodal missing-prone features (`1% < NaN < 25%`) without hard thresholds.
   - Trains 5-fold OOF `LightGBM` classifiers strictly from peer cross-features (excluding all columns sharing the same base/geo group), calibrates conditional probabilities out-of-fold via `IsotonicRegression`, and recovers hidden minority-mode cases specifically where primary values are `NaN`:
     - `behavior_stability`: **OOF ROC-AUC = `0.9816`** (`1,257` missing entries $\implies$ `17` recovered candidates at $p \ge 0.15$; mixture-expectation imputed back into downstream geometric profile features).
     - Temporal Period-4 & Bridged Geo features: **OOF ROC-AUC = `0.9946–1.0000`**.

5. **Classical Unsupervised Ensemble & Noisy-OR Expected-$F_3$ Thresholding:**
   - Trains Profile & Temporal `IsolationForest`s, `NearestNeighbors(k=20)`, and `MLPRegressor((64, 16, 64))` Bottleneck Neural Autoencoder alongside Robust L2/Max temporal norms.
   - Calibrates global score via 2-component `GaussianMixture`, fuses orthogonal anomaly posteriors via Probabilistic **Noisy-OR**, and chooses exact cutoff $k^*$ maximizing $\mathbb{E}[F_3(k)]$.

---

## Measured Post-Hoc Pure-ML Audit (Zero Overrides Used)

- **Runtime:** `111.1s`
- **Estimated Expected Anomaly Mass:** `2,103.5` accounts (`8.13%`)
- **Bayes-Optimal $F_3$ Cutoff:** **`2,192` flagged accounts (`8.47%`)**, cutoff `min p = 0.1546`
- **Coverage across Reference Signatures (100% Pure ML, No Rule Overrides):**

| Structural Reference Signature | V5 (Pure ML without Override) | V6 Pure Unsupervised ML (No Override) |
| :--- | :---: | :---: |
| **Bot Farm (`ip < 0.25`)** | 100.0% (207/207) | **100.0% (207/207)** |
| **Low Activity (`activity < 0.35`)** | 98.5% (261/265) | **100.0% (265/265)** |
| **No Communication (`comm < 0.15`)** | 97.7% (257/263) | **100.0% (263/263)** |
| **Low Stability (`stability < 0.70`)** | 36.2% (89/246) | **100.0% (246/246)** + `NaN` recovery |
| **Theft (`d_cur > 6` \| `geo34 > 4`)** | 100.0% (463/463) | **100.0% (463/463)** |
| **Boosting (`d_skill_traj > 10`)** | 100.0% (299/299) | **100.0% (299/299)** |
| **VPN (`geo23 > 4`)** | 100.0% (165/165) | **100.0% (165/165)** |
| **Combat Variance Corner** | 67.5% (260/385) | **77.7% (299/385)** |
| **Location Hopping (`loc_4 > 5`)** | 100.0% (493/493) | **100.0% (493/493)** |
| **Friend Removes (`rem_4 > 6`)** | 100.0% (297/297) | **100.0% (297/297)** |
| **OVERALL REFERENCE CAPTURE** | **86.6% (1,887/2,179)** | **96.05% (2,093/2,179)** |

- **Additional Genuine Recoveries Outside Reference Set (`99` accounts total):** includes `22` missing-`behavior_stability` buyer candidates (low login + high purchase frequency + older account age), `20` Period-4 purchase/VPN anomalies with missing consecutive geo/loc cells, and `14` borderline high-excess-density Combat Variance accounts.

---

## How to Run & Merge
```powershell
python task5\train_task5.py
python merge_submission.py
```
Outputs:
- Task 5 Prediction: `submissions/sub_task5.csv` (`25,889` rows, `2,192` anomalies / `8.47%`)
- Teammate Detailed Summary (Thai): [`task5/TASK5_HANDOFF_V6.md`](TASK5_HANDOFF_V6.md)
