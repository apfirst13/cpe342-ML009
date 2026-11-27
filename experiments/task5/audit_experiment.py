"""
Post-hoc evaluation and diff audit utility for Task 5 experiments.
Strictly for offline analysis/logging into task5/TASK5_RESEARCH_LOG.md;
never imported or used by task5/train_task5.py.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "task5")
DEFAULT_CAND = os.path.join(BASE_DIR, "submissions", "sub_task5_v7-b.csv")
DEFAULT_REF = os.path.join(BASE_DIR, "task5", "experiments", "sub_task5_v6.csv")


def compute_audit_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    m_act = df[[f"activity_level_{i}" for i in range(1, 5)]].mean(axis=1)
    m_comm = df[[f"communication_frequency_{i}" for i in range(1, 5)]].mean(axis=1)
    m_ip = df[[f"ip_hash_entropy_{i}" for i in range(1, 5)]].mean(axis=1)
    geo23 = np.sqrt((df["login_lat_3"] - df["login_lat_2"]) ** 2 + (df["login_lon_3"] - df["login_lon_2"]) ** 2)
    geo34 = np.sqrt((df["login_lat_4"] - df["login_lat_3"]) ** 2 + (df["login_lon_4"] - df["login_lon_3"]) ** 2)
    d_cur = df["currency_transfers_4"] - df[[f"currency_transfers_{i}" for i in range(1, 4)]].mean(axis=1)
    d_st = df["skill_trajectory_4"] - df[[f"skill_trajectory_{i}" for i in range(1, 4)]].mean(axis=1)
    acc_s = df[[f"accuracy_std_{i}" for i in range(1, 5)]].mean(axis=1)
    kda_s = df[[f"kda_std_{i}" for i in range(1, 5)]].mean(axis=1)
    perf = df[[f"performance_stability_{i}" for i in range(1, 5)]].mean(axis=1)

    ref9 = (
        (m_ip < 0.25)
        | (m_act < 0.35)
        | (m_comm < 0.15)
        | (df["behavior_stability"] < 0.70)
        | (d_cur > 6.0)
        | (geo34 > 4.0)
        | (d_st > 10.0)
        | (geo23 > 4.0)
        | (df["location_changes_4"] > 5.0)
        | (df["friend_removes_4"] > 6.0)
    ).fillna(False).to_numpy()
    combat_m = ((acc_s > 9.0) & (kda_s > 0.70) & (perf > 0.87)).fillna(False).to_numpy()
    ref10 = ref9 | combat_m
    near_combat = ((acc_s > 8.2) & (kda_s > 0.65) & (perf > 0.84)).fillna(False).to_numpy()
    bs_na = df["behavior_stability"].isna().to_numpy()
    geo_na = (geo23.isna() | geo34.isna()).to_numpy()
    cur_st_na = (d_cur.isna() | d_st.isna()).to_numpy()

    return {
        "Ref9_1794": ref9,
        "CombatVar_385": combat_m,
        "Ref10_2179": ref10,
        "NearCombat": near_combat & (~ref10),
        "bs_na": bs_na & (~ref10),
        "geo_or_d_na": (geo_na | cur_st_na) & (~ref10) & (~bs_na),
        "other_ex": (~ref10) & (~bs_na) & (~geo_na) & (~cur_st_na) & (~near_combat),
    }


def summarize_group(label: str, mask: np.ndarray, df: pd.DataFrame) -> None:
    cnt = int(mask.sum())
    if cnt == 0:
        print(f"  {label:<28s}: N=    0")
        return
    pwd = float(df.loc[mask, "password_changed"].mean() * 100.0)
    vpn = float(df.loc[mask, "vpn_usage"].mean() * 100.0)
    sale = float(df.loc[mask, "mass_item_sale"].mean() * 100.0)
    susp = float(df.loc[mask, "suspicious_login_time"].mean() * 100.0)
    print(f"  {label:<28s}: N={cnt:5d} | pwd={pwd:5.1f}% vpn={vpn:5.1f}% sale={sale:5.1f}% susp={susp:5.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cand", default=DEFAULT_CAND)
    parser.add_argument("--ref", default=DEFAULT_REF)
    args = parser.parse_args()

    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample_sub = pd.read_csv(os.path.join(BASE_DIR, "data", "sample_submission.csv"))
    cand_df = pd.read_csv(args.cand)
    ref_df = pd.read_csv(args.ref)

    assert len(cand_df) == 25889, f"Bad row count: {len(cand_df)}"
    assert list(cand_df.columns) == ["id", "task5"], f"Bad cols: {list(cand_df.columns)}"
    assert (cand_df["id"].values == sample_sub["id"].values).all(), "ID mismatch!"
    assert not cand_df.isna().any().any(), "NaNs found!"
    assert set(cand_df["task5"].unique()).issubset({0, 1}), "Non-binary values!"

    p_cand = cand_df["task5"].to_numpy() == 1
    p_ref = ref_df["task5"].to_numpy() == 1
    masks = compute_audit_masks(test_df)

    print("=" * 72)
    print(f"Candidate: {args.cand}")
    print(f"Reference: {args.ref}")
    print("=" * 72)
    print(f"Total Flagged: {p_cand.sum():,} / {len(p_cand):,} ({p_cand.mean() * 100:.2f}%)")
    print(f"Reference V6 : {p_ref.sum():,} / {len(p_ref):,} ({p_ref.mean() * 100:.2f}%)")
    print(f"Overlap      : {(p_cand & p_ref).sum():,} | Added (+): {(p_cand & ~p_ref).sum():,} | Removed (-): {(~p_cand & p_ref).sum():,}")

    print("\nSubgroup Capture Rates:")
    for nm, m in masks.items():
        if nm in ("Ref9_1794", "CombatVar_385", "Ref10_2179"):
            c_hit = int((p_cand & m).sum())
            r_hit = int((p_ref & m).sum())
            print(f"  {nm:<18s}: Cand={c_hit:4d}/{m.sum():4d} ({c_hit/m.sum()*100:5.1f}%) | Ref={r_hit:4d}/{m.sum():4d} ({r_hit/m.sum()*100:5.1f}%)")
        else:
            c_hit = int((p_cand & m).sum())
            r_hit = int((p_ref & m).sum())
            print(f"  {nm:<18s}: Cand={c_hit:4d} | Ref={r_hit:4d}")

    print("\nDiff Inspection (Red-Flag Sanity Check):")
    summarize_group("Added (+cand & ~ref)", p_cand & ~p_ref, test_df)
    summarize_group("Removed (~cand & ref)", ~p_cand & p_ref, test_df)
    summarize_group("Baseline Unflagged Norm", (~p_cand) & (~masks["Ref10_2179"]), test_df)


if __name__ == "__main__":
    main()
