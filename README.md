# CPE 342 Machine Learning (1/2026) - Karena Player Intelligence System 

Kaggle Competition: [Player Intelligence System - CPE342 ML](https://www.kaggle.com/competitions/cpe342-karena)  
กำหนดส่ง Kaggle: **24 พ.ย. 2026** | กำหนดส่ง Report & Code (LEB2): **30 พ.ย. 2026**

---

##  การแบ่งหน้าที่ในทีม (Team Roles & Task Assignments)

| สมาชิก | Task ที่รับผิดชอบ | ชนิดของโมเดล / Metric | โฟลเดอร์ทำงาน |
| :---: | :--- | :--- | :---: |
| **คนที่ 1** | **Task 1:** Anti-Cheat Pre-Filter<br>**Task 5:** Account Security Monitoring | Binary Classification ($F_2$ Score)<br>Unsupervised Anomaly ($F_3$ Score) | `task1/`<br>`task5/` |
| **คนที่ 2** | **Task 4:** Game Title Detection | Image Classification (Macro $F_1$) | `task4/` |
| **คนที่ 3** | **Task 2:** Player Segment Classification | Multi-Class Classification (Macro $F_1$) | `task2/` |
| **คนที่ 4** | **Task 3:** Player Monthly Spending Prediction | Regression (Normalized MAE - NMAE) | `task3/` |

---

##  คู่มือเริ่มต้นใช้งานสำหรับเพื่อนร่วมทีม (Team Setup Guide)

ทำตามขั้นตอนด้านล่างนี้บนเครื่องของตัวเองเพียงครั้งเดียวเพื่อเริ่มงาน:

### 1. ดึงโค้ดลงเครื่อง (Git Clone)
เปิด Terminal หรือ PowerShell ในโฟลเดอร์ที่ต้องการ แล้วรัน:
```bash
git clone https://github.com/apfirst13/cpe342-ML009.git
cd kaggle
```

### 2. สร้าง Virtual Environment & ติดตั้ง Libraries
รันคำสั่งตามระบบปฏิบัติการของคุณ:

**สำหรับ Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
*(หากติดสิทธิ์ Restricted Script บน Windows ให้รัน: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` ก่อนสั่ง Activate)*

**สำหรับ macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

##  การจัดการชุดข้อมูล (Dataset Setup)

> ** ข้อสำคัญ:** โฟลเดอร์ `data/` จะ**ไม่ถูกอัปโหลดขึ้น GitHub** (ถูกกันไว้ด้วย `.gitignore` เนื่องจากขนาดไฟล์ใหญ่ 1GB+) ทุกคนต้องดาวน์โหลดมาไว้ในเครื่องตนเอง

1. ไปที่แท็บ Data บน Kaggle: [cpe342-karena/data](https://www.kaggle.com/competitions/cpe342-karena/data)
2. กดปุ่ม **"Download All"** จะได้ไฟล์ `cpe342-karena.zip`
3. แตกไฟล์ zip ทั้งหมดลงในโฟลเดอร์ `data/` ให้ได้โครงสร้างดังนี้:
```text
kaggle/
└── data/
    ├── task1/
    │   ├── train.csv
    │   ├── test.csv
    │   └── datacard.txt
    ├── task2/
    ├── task3/
    ├── task4/
    │   ├── train/            # โฟลเดอร์รูปภาพสำหรับเทรน
    │   ├── test/             # โฟลเดอร์รูปภาพสำหรับทดสอบ
    │   ├── train.csv
    │   └── test.csv
    ├── task5/
    │   ├── test.csv          # ไม่มี train.csv (Unsupervised)
    │   └── datacard.txt
    └── sample_submission.csv
```

---

##  ขั้นตอนการทำงานร่วมกันในทีม (Workflow)

1. **เขียนโค้ดในโฟลเดอร์ Task ของตัวเอง:**  
   เช่น คนที่ทำ Task 1 ให้เขียนโค้ดหรือ Notebook ไว้ในโฟลเดอร์ `task1/` เพื่อไม่ให้โค้ดชนกับเพื่อน
2. **ส่งออกผลลัพธ์การทำนาย (Prediction Export):**  
   เมื่อโมเดลทำนายผลชุดทดสอบเสร็จแล้ว ให้เซฟไฟล์ CSV ขนาด 25,889 แถว มี 2 คอลัมน์คือ `id` และคำตอบ มาวางไว้ที่โฟลเดอร์ `submissions/` โดยตั้งชื่อตาม Task ตัวเอง:
   * Task 1 $\rightarrow$ `submissions/sub_task1.csv` (คอลัมน์ `id`, `task1` ค่า 0 หรือ 1)
   * Task 2 $\rightarrow$ `submissions/sub_task2.csv` (คอลัมน์ `id`, `task2` ค่า 0, 1, 2, 3)
   * Task 3 $\rightarrow$ `submissions/sub_task3.csv` (คอลัมน์ `id`, `task3` ค่าเงินบาท THB $\ge 0$)
   * Task 4 $\rightarrow$ `submissions/sub_task4.csv` (คอลัมน์ `id`, `task4` ค่าตัวเลข `0–4`: `0=7-11 Knights, 1=FiveN, 2=Free Fried, 3=PUBE, 4=Roblock`)
   * Task 5 $\rightarrow$ `submissions/sub_task5.csv` (คอลัมน์ `id`, `task5` ค่า 0 หรือ 1)

3. **รวมไฟล์คำตอบอัตโนมัติ (Merge & Validate):**  
   รันสคริปต์รวมคะแนนใน Terminal:
   ```bash
   python merge_submission.py
   ```
   * สคริปต์จะดึงผลลัพธ์ของแต่ละ Task มารวมกัน
   * ตรวจสอบว่าครบ **25,889 แถว**, ไม่มีค่าว่าง (NaN), ค่าตัวเลข/คลาสถูกต้อง
   * สร้างไฟล์ **`final_submission.csv`** พร้อมนำไปกดส่งบน Kaggle ได้ทันที!
   *(หาก Task ไหนยังทำไม่เสร็จ สคริปต์จะมี Default ค่ากลางใส่ให้อัตโนมัติ เพื่อให้ส่งทดสอบระบบได้ก่อน)*

4. **การ Push โค้ดขึ้น GitHub:**
   ```bash
   git add taskX/
   git commit -m "feat(taskX): update baseline model"
   git pull origin main
   git push origin main
   ```

---

## 📝 บันทึกการอัปเดตโค้ดล่าสุด (Changelog)

### 🔹 Task 5: Account Security Monitoring (`task5/train_task5.py`, `task5/README.md`) — อัปเกรดคะแนนรวม Kaggle `0.76` $\rightarrow$ **`0.8224`**
* **Multi-View Feature Construction:** แยกข้อมูลเป็น **Profile View** (`mean`, `std`, static traits), **Temporal Delta View** (ความเปลี่ยนแปลงข้ามช่วงเวลา `p4 - mean(p1..3)`, `p3 - mean(p1..2)`, `p2 - p1`, `p4 - p1` ที่ปรับสเกลด้วยค่า MAD ของช่วงฐานปกติ `p2 - p1` และระยะกระโดดพิกัด `geo_12, geo_23, geo_34, geo_max`), และ **Semantic Combat Subspace**
* **Diverse From-Scratch Unsupervised Detectors:**
  * **OOF Monte Carlo Density-Ratio Estimation** (`HistGradientBoostingClassifier` แยกข้อมูลจริง vs ข้อมูลสุ่มสลับบล็อก) เพื่อจับกลุ่ม Anomaly หลายมิติที่ซ่อนอยู่
  * **Freedman-Diaconis Smoothed 1D KDE Surprise** (`max` และ `top-3` ของ $-\log p(x)$)
  * **Isolation Forests** บนทั้ง Profile space และ Temporal Delta space
  * **Robust Temporal Deviation Scores** ($L_2$, $L_\infty$) และ **From-Scratch Neural Autoencoder (`MLPRegressor`)** Reconstruction MSE
* **Log-Compressed Soft-Max / $L_3$ Aggregation & GMM $F_3$ Thresholding:** รวมสัญญาณจากทุกมุมมองด้วย `(Max + Top2Mean + L3) / 3` และใช้ **2-Component Gaussian Mixture Model (GMM)** หาจุดตัดที่ให้ค่าคาดหวัง $F_3$ สูงสุด (`2,556` บัญชี, `9.87%`)

### 🔹 Task 2: Player Segment Classification (`task2/train_task2.py`, `task2/task2.ipynb`) — อัปเกรดคะแนน OOF Macro F1 เป็น **`~0.767`**
* **Denoising:** ตัด 16 คอลัมน์ที่เป็น Pure Noise ออกจากการเทรน
* **Leak-Free Fold Feature Engineering:** สร้าง Composite Z-Scores แยกตาม 2 แกนหลัก (`Engagement Axis` & `Spending Axis`), `log1p`, และ Inner-OOF Binned Naive Bayes meta-features ภายในแต่ละ Fold (ไม่มี Data Leakage 100%)
* **6-Model Level-1 Ensemble + Level-2 Stacking:** เทรน **LightGBM, XGBoost, CatBoost, 2-Axis Binary LightGBM, Spline GAM, และ MLP Neural Network** แล้วทำ Level-2 LogisticRegression Stacking พร้อม Nested-OOF Class Multiplier Tuning

### 🔹 Submission Merger (`merge_submission.py`)
* **Task 4 Numeric Validation:** อัปเดต `merge_submission.py` ให้รองรับและตรวจสอบคอลัมน์ `task4` เป็นตัวเลข `0–4` (พร้อมระบบแปลงอัตโนมัติหากไฟล์ต้นทางเป็นชื่อเกม)

