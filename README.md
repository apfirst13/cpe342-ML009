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
├── task1/task1.ipynb       # Final Notebook สำหรับ Task 1 (Anti-Cheat Pre-Filter)
├── task2/task2.ipynb       # Final Notebook สำหรับ Task 2 (Player Segment Classification)
├── task3/task3.ipynb       # Final Notebook สำหรับ Task 3 (Monthly Spending Prediction)
├── task4/task4.ipynb       # Final Notebook สำหรับ Task 4 (Game Title Detection)
├── task5/task5.ipynb       # Final Notebook สำหรับ Task 5 (Account Security Monitoring)
├── experiments/            # สคริปต์ทดลอง โน้ตบุ๊กเก่า และบันทึกการพัฒนา (task1-task5, scratch)
├── data/                   # โฟลเดอร์วางชุดข้อมูลจาก Kaggle (ไม่รวมใน Git)
├── submissions/            # ไฟล์คำตอบแยกตาม Task (sub_task1.csv - sub_task5.csv)
├── merge_submission.py     # สคริปต์รวมคำตอบทั้ง 5 Tasks และตรวจสอบความถูกต้อง
├── final_submission.csv    # ไฟล์คำตอบสุดท้ายพร้อมส่ง Kaggle
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

### 3. รัน Final Notebook ของแต่ละ Task
โน้ตบุ๊กสุดท้ายของทุก Task ถูกจัดให้อยู่ในไฟล์เดียวจบแบบ Standalone และบันทึกผลลัพธ์ไปยัง `submissions/sub_task1.csv` ถึง `sub_task5.csv` โดยอัตโนมัติ (สคริปต์ทดลองและบันทึกการพัฒนาก่อนหน้าเก็บแยกไว้ใน `experiments/`)

| Task | Final Notebook | สภาพแวดล้อมที่แนะนำ | ไฟล์คำตอบที่สร้าง |
| :--- | :--- | :--- | :--- |
| **Task 1** | `task1/task1.ipynb` | Local CPU/GPU (Jupyter / VS Code) | `submissions/sub_task1.csv` |
| **Task 2** | `task2/task2.ipynb` | Local CPU (Jupyter / VS Code) | `submissions/sub_task2.csv` |
| **Task 3** | `task3/task3.ipynb` | Local CPU (Jupyter / VS Code) | `submissions/sub_task3.csv` |
| **Task 4** | `task4/task4.ipynb` | Google Colab L4 GPU / Local GPU | `submissions/sub_task4.csv` |
| **Task 5** | `task5/task5.ipynb` | Local CPU (Jupyter / VS Code) | `submissions/sub_task5.csv` |

หรือรันผ่าน CLI ด้วย `jupyter nbconvert`:
```bash
jupyter nbconvert --to notebook --execute --inplace task1/task1.ipynb
jupyter nbconvert --to notebook --execute --inplace task2/task2.ipynb
jupyter nbconvert --to notebook --execute --inplace task3/task3.ipynb
# Task 4 แนะนำให้เปิดรัน E2E บน Google Colab L4 GPU (task4/task4.ipynb)
jupyter nbconvert --to notebook --execute --inplace task5/task5.ipynb
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
| **Task 1** | 71-Feature Covariance-Imputed + Fold-Isolated Pseudo-Labeling + 7-Model Logit Blend | Nested OOF $F_2 = 0.83841$ (AUC $0.89340$) / Global $0.83863$ |
| **Task 2** | Denoise 16 Noise Cols + 2-Axis Latent Scores + 6-Model Level-2 Stacking | Nested OOF Macro $F_1 = 0.75664$ (Full $0.75760$) |
| **Task 3** | Algebraic Recovery + Two-Stage Bayes-L1 + Mixture CDF + 50/50 Multi-CV Ensemble | Nested OOF NMAE $= 0.79630$ |
| **Task 4** | De-Tinted Restoration + Calibrated 3-Paradigm Ensemble (`ConvNeXt-S` + `Swin-T` + `5-Foundation MLP`) | Unseen-Scene $F_1 = 0.98038$ (Balanced $F_1 = 0.97910$) |
| **Task 5** | V7-A Pure Unsupervised (Unclipped 1D Topo KDE + Poisson $Z \ge 2.0$ + 1-Step EM 3D Combat DR + Mixture-MICE + Noisy-OR) | $2,163$ Flagged Accounts ($8.35\%$) |
