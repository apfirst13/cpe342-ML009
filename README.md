# CPE 342 Machine Learning (1/2026) — Karena Player Intelligence System

Repository สำหรับโปรเจกต์รายวิชา **CPE 342 Machine Learning** (กลุ่ม **ML009** / ทีม Kaggle: **ML09**)  
การแข่งขัน: [Player Intelligence System - CPE342 ML](https://www.kaggle.com/competitions/player-intelligence-system-cpe-342-ml)

---

## สมาชิกในทีม (Team Members)

| GitHub | Task ที่รับผิดชอบ | ประเภทงาน / Metric | โฟลเดอร์ |
| :--- | :--- | :--- | :---: |
| `apfirst13` | Task 1: Anti-Cheat Pre-Filter<br>Task 5: Account Security Monitoring | Binary Classification ($F_2$ Score)<br>Unsupervised Anomaly ($F_3$ Score) | `task1/`<br>`task5/` |
| `meebotsompurin-stack` (`Purin`) | Task 2: Player Segment Classification | Multi-Class Classification (Macro $F_1$) | `task2/` |
| `poom2548` | Task 3: Player Monthly Spending Prediction | Regression (Normalized MAE - NMAE) | `task3/` |
| `Theerawat` | Task 4: Game Title Detection | Image Classification (Macro $F_1$) | `task4/` |

---

## โครงสร้างโปรเจกต์ (Repository Structure)

```text
cpe342-ML009/
├── task1/                  # โค้ดและโน้ตบุ๊กสำหรับ Task 1 (Anti-Cheat Pre-Filter)
├── task2/                  # โค้ดและโน้ตบุ๊กสำหรับ Task 2 (Player Segment Classification)
├── task3/                  # โค้ดและโน้ตบุ๊กสำหรับ Task 3 (Monthly Spending Prediction)
├── task4/                  # โค้ดและโน้ตบุ๊กสำหรับ Task 4 (Game Title Detection)
├── task5/                  # โค้ดและโน้ตบุ๊กสำหรับ Task 5 (Account Security Monitoring)
├── data/                   # โฟลเดอร์วางชุดข้อมูลจาก Kaggle (ไม่รวมใน Git)
├── submissions/            # ไฟล์คำตอบแยกตาม Task (sub_task1.csv - sub_task5.csv)
├── merge_submission.py     # สคริปต์รวมคำตอบทั้ง 5 Tasks และตรวจสอบความถูกต้อง
└── requirements.txt        # รายการไลบรารีที่ใช้ในโปรเจกต์
```

---

## วิธีติดตั้งและรันโค้ด (Setup & Usage)

### 1. ติดตั้ง Environment
```bash
git clone https://github.com/apfirst13/cpe342-ML009.git
cd cpe342-ML009

python -m venv venv
# Windows (PowerShell): .\venv\Scripts\Activate.ps1
# macOS / Linux:      source venv/bin/activate

pip install -r requirements.txt
```

### 2. เตรียมชุดข้อมูล
ดาวน์โหลดชุดข้อมูลจาก Kaggle แล้วแตกไฟล์ลงในโฟลเดอร์ `data/` (`data/task1/` ถึง `data/task5/` และ `data/sample_submission.csv`)

### 3. รันโมเดลของแต่ละ Task
```bash
# Task 1
python task1/train_task1_upgraded.py    # (หรือ task1/train_best_task1.py)

# Task 2
python task2/train_task2.py

# Task 3
python task3/train_task3.py

# Task 4 (รันเทรนบน Colab L4 จาก task4/task4_cloud_t4_v4.ipynb แล้วรวมผลด้วย)
python task4/run_phase3_cloud_meta_ensemble.py

# Task 5
python task5/train_task5.py
```

### 4. รวมไฟล์คำตอบสำหรับส่ง Kaggle
```bash
python merge_submission.py
```
สคริปต์จะดึง `submissions/sub_task1.csv` ถึง `sub_task5.csv` มารวมเป็น `final_submission.csv` (ตรวจสอบครบ 25,889 แถว และไม่มีค่าว่าง)

---

## สรุปแนวทางและผลลัพธ์แต่ละ Task

| Task | แนวทางหลัก (Methodology) | ผลประเมิน (Validation) |
| :--- | :--- | :--- |
| **Task 1** | 71-Feature Covariance-Imputed + Fold-Isolated Pseudo-Labeling + 7-Model Logit Blend | Nested OOF $F_2 = 0.83815$ (AUC $0.89336$) |
| **Task 2** | Denoise 16 Noise Cols + 2-Axis Latent Scores + 6-Model Level-2 Stacking | OOF Macro $F_1 = 0.76700$ |
| **Task 3** | Algebraic Recovery + Two-Stage Bayes-L1 + Mixture CDF + 50/50 Multi-CV Ensemble | Nested OOF NMAE $= 0.79630$ |
| **Task 4** | De-Tinted Restoration + Calibrated 3-Paradigm Ensemble (`ConvNeXt-S` + `Swin-T` + `5-Foundation MLP`) | Unseen-Scene $F_1 = 0.98036$ |
| **Task 5** | V6 Pure Unsupervised (Topo Valley KDE + 3D Combat DR + Mixture-MICE + Noisy-OR) | Pure-ML Reference Capture $96.05\%$ ($2,192$ accounts) |
