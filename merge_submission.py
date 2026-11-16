"""
Script: merge_submission.py
Description: รวมผลลัพธ์การทำนายจากสมาชิกในทีมทั้ง 5 Tasks เข้าเป็นไฟล์เดียวสำหรับส่งบน Kaggle
พร้อมตรวจสอบความถูกต้องของฟอร์แมต ข้อมูลสูญหาย และขนาดของไฟล์
"""

import os
import sys
import glob
from datetime import datetime
import pandas as pd
import numpy as np

EXPECTED_ROWS = 25889
TASK4_VALID_LABELS = {"7-11K", "fiveN", "freefried", "pube", "roblock"}

def load_task_pred(task_idx, fallback_df):
    """
    ค้นหาไฟล์ผลลัพธ์ของ task ที่กำหนดในโฟลเดอร์ submissions/
    เช่น submissions/sub_task1.csv, submissions/task1.csv
    """
    col_name = f"task{task_idx}"
    patterns = [
        f"submissions/sub_task{task_idx}.csv",
        f"submissions/task{task_idx}.csv",
        f"submissions/*task{task_idx}*.csv"
    ]
    
    matched_files = []
    for pat in patterns:
        matched_files.extend(glob.glob(pat))
        
    if not matched_files:
        print(f"⚠️ [Task {task_idx}] ไม่พบไฟล์ผลลัพธ์ใน submissions/ -> จะใช้ค่า Default ชั่วคราว")
        if task_idx in [1, 5]:
            return pd.Series(0, index=fallback_df.index, name=col_name)
        elif task_idx == 2:
            return pd.Series(0, index=fallback_df.index, name=col_name)
        elif task_idx == 3:
            return pd.Series(0.0, index=fallback_df.index, name=col_name)
        elif task_idx == 4:
            return pd.Series("freefried", index=fallback_df.index, name=col_name)
            
    filepath = matched_files[0]
    print(f"✅ [Task {task_idx}] โหลดผลลัพธ์จาก: {filepath}")
    df = pd.read_csv(filepath)
    
    # ดึงคอลัมน์ prediction
    if col_name in df.columns:
        pred_series = df[col_name]
    elif "prediction" in df.columns:
        pred_series = df["prediction"]
    elif len(df.columns) == 2:
        pred_series = df.iloc[:, 1]
    elif len(df.columns) == 1:
        pred_series = df.iloc[:, 0]
    else:
        raise ValueError(f"ไม่สามารถระบุคอลัมน์คำตอบในไฟล์ {filepath} ได้ กรุณาตั้งชื่อคอลัมน์เป็น 'id' และ '{col_name}'")
        
    return pred_series

def main():
    sample_sub_path = "data/sample_submission.csv"
    
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
    else:
        print("ℹ️ ไม่พบ data/sample_submission.csv ในระบบ กำลังสร้าง ID เทมเพลต ANS00001 - ANS25889 ชั่วคราว...")
        ids = [f"ANS{str(i).zfill(5)}" for i in range(1, EXPECTED_ROWS + 1)]
        sample_sub = pd.DataFrame({"id": ids})
        
    out_df = pd.DataFrame()
    out_df["id"] = sample_sub["id"]
    
    # รวมผลการทำนายทั้ง 5 tasks
    for i in range(1, 6):
        out_df[f"task{i}"] = load_task_pred(i, sample_sub).values
        
    # --- Data Validation Check ---
    print("\n🔍 กำลังตรวจสอบความถูกต้องของไฟล์ Submission...")
    assert len(out_df) == EXPECTED_ROWS, f"❌ จำนวนแถวต้องมี {EXPECTED_ROWS} แต่ได้ {len(out_df)}"
    assert list(out_df.columns) == ["id", "task1", "task2", "task3", "task4", "task5"], "❌ คอลัมน์ไม่ถูกต้อง"
    
    # ตรวจสอบ NaN
    nan_counts = out_df.isna().sum()
    if nan_counts.any():
        print(f"❌ พบค่า NaN ในข้อมูล:\n{nan_counts}")
        sys.exit(1)
        
    # Task 1 & 5: Binary Check
    assert set(out_df["task1"].unique()).issubset({0, 1}), "❌ Task 1 ต้องมีเฉพาะค่า 0 หรือ 1"
    assert set(out_df["task5"].unique()).issubset({0, 1}), "❌ Task 5 ต้องมีเฉพาะค่า 0 หรือ 1"
    
    # Task 2: 4-Class Check
    assert set(out_df["task2"].unique()).issubset({0, 1, 2, 3}), "❌ Task 2 ต้องมีเฉพาะค่า 0, 1, 2, 3"
    
    # Task 3: Regression Non-negative Check
    if (out_df["task3"] < 0).any():
        print("⚠️ พบค่ายอดเงินติดลบใน Task 3 -> กำลังคลิปค่าต่ำสุดให้เป็น 0")
        out_df["task3"] = out_df["task3"].clip(lower=0)
        
    # Task 4: Label Check
    t4_labels = set(out_df["task4"].astype(str).unique())
    invalid_labels = t4_labels - TASK4_VALID_LABELS
    if invalid_labels:
        print(f"❌ Task 4 มี Label ที่ไม่ถูกต้อง: {invalid_labels}")
        print(f"Label ที่ถูกต้องคือ: {TASK4_VALID_LABELS}")
        sys.exit(1)
        
    # บันทึกไฟล์
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_latest = "final_submission.csv"
    out_history = f"submissions/submission_{timestamp}.csv"
    
    out_df.to_csv(out_latest, index=False)
    out_df.to_csv(out_history, index=False)
    
    print("\n🎉 รวมไฟล์สำเร็จเรียบร้อย!")
    print(f"📁 บันทึกไฟล์ที่: {out_latest} และ {out_history}")
    print(f"📊 ขนาดข้อมูล: {out_df.shape[0]} แถว x {out_df.shape[1]} คอลัมน์")
    print("\nตัวอย่างข้อมูล 5 แถวแรก:")
    print(out_df.head())

if __name__ == "__main__":
    main()
