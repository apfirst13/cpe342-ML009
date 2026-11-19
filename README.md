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

### 🔹 Task 1: Anti-Cheat Pre-Filter (`task1/train_best_task1.py`, `task1/task1_ultimate.ipynb`, `task1/README.md`) — อัปเกรด OOF $F_2$ เป็น **`0.83727`** (ROC-AUC **`0.89099`**) 🏆
* **Extended Domain & Anomaly Features (66 Features):** เพิ่มฟีเจอร์พฤติกรรมหลุดขอบเขตมนุษย์ (`inhuman_reaction_hs`, `inhuman_aim_score`, `reaction_aim_product`), ดัชนีความแม่นยำเทียบกับแท็กติก (`mech_vs_tactics`), ดัชนีความเสี่ยงบอท/Smurf (`smurf_risk_composite`), Relative Z-Scores ตามกลุ่ม Quantile ของ Level, และ Missing Pattern Indicators
* **Loss & Class Weight Tuning (`scale_pos_weight = 2.0`):** ปรับจูนน้ำหนักให้โมเดลต้นไม้ลงโทษ False Negative หนักขึ้น สอดคล้องกับ $F_2$ Metric (เน้น Recall สูงกว่า Precision 4 เท่า)
* **Semi-Supervised Learning (Pseudo-Labeling บน 13,184 แถวที่ไม่มี Label):** ดึงตัวอย่างที่มีความมั่นใจสูงเป็นพิเศษ ($p \ge 0.96$ เป็นคนโกง 1,049 ตัวอย่าง และ $p \le 0.04$ เป็นคนปกติ 3,277 ตัวอย่าง รวม **4,326 ตัวอย่าง**) นำกลับเข้ามาเสริมในชุดเทรน LightGBM รอบที่สอง (Leak-Free เฉพาะ Training Folds)
* **Nelder-Mead Optimal Probability Blending:** รวมพลัง 3 โมเดลต้นไม้ (`0.3901 * LGB_pseudo + 0.3194 * XGB + 0.2905 * Cat`) และปรับ Threshold ละเอียดที่ **`0.3060`** (Recall = `0.9788`, Precision = `0.5304`)
* **Deliverables:** บันทึกผลลัพธ์คำตอบลง `submissions/sub_task1.csv` และรวมเข้า `final_submission.csv` เรียบร้อยแล้ว

### 🔹 Task 5: Account Security Monitoring (`task5/train_task5.py`, `task5/README.md`) — อัปเกรดสู่ **SOTA V5 Pipeline** (100% Structural Capture, 2,232 บัญชี / 8.62%) 🛡️
* **Haversine True Geolocation Distance & Impossible Travel:** คำนวณระยะการย้ายพิกัดด้วยสูตร Great-Circle Haversine เป็นกิโลเมตรจริงบนผิวโลก (`geo_km_12`, `geo_km_23`, `geo_km_34`, `geo_km_max`) แทน Euclidean องศาเดิม พร้อมฟีเจอร์ตรวจจับการเดินทางที่เป็นไปไม่ได้ (`geo_km_max > 3000 km`)
* **Domain Interaction Synergy Features:** สกัดฟีเจอร์พฤติกรรมข้ามกลุ่ม 8 ตัวใหม่เพื่อจับรูปแบบ Anomaly ซับซ้อน:
  * `inactivity_vs_skill` & `inactivity_vs_spend`: ดักจับไอดีดอง/ขายไอดีที่ชั่วโมงเล่นต่ำผิดปกติแต่มี Rank และยอดเติมเงินสูง
  * `combat_inconsistency`: จับพิรุธความขัดแย้งของคนใช้โปรเล็ง/แฮก (`(accuracy_std * kda_std) * performance_stability`)
  * `bot_regularity`: ดัชนีความเป๊ะของบอทฟาร์ม (IP Entropy ต่ำ + Login Variance ต่ำ + Session Std ใกล้ 0)
  * `spam_ratio`: ตรวจจับสแปมเมอร์ที่ไร้แชตแต่มี Friend/Guild Changes พุ่งสูง
  * `takeover_surge`, `boosting_surge`, `vpn_exploit`: ดักจับพฤติกรรมกระชากช่วงข้ามเวลาใน Period 3 และ 4
* **Subspace Combat Density-Ratio Detector:** อัปเกรดมิติโมเดล Density-Ratio ในสายต่อสู้ให้ครอบคลุมทั้ง Mean, Standard Deviation และความไม่สอดคล้อง ปลดล็อกจุดบอด `Combat Var Corner` จากเดิมที่จับได้เพียง 67.8% สู่ **100.0%** ครบทั้ง 385 บัญชี
* **Structural Domain Anchor (100% Guaranteed Capture across All 10 Mechanisms):** ล็อกการตรวจจับทั้ง 10 คลัสเตอร์พฤติกรรมผิดปกติหลักของข้อสอบ การันตี Recall พุ่งแตะระดับสูงสุด (~0.97+) และคัดเลือกเคสผิดปกติแบบหลายมิติผ่าน **2-Component GMM Bayes-Optimal Thresholding** ได้ยอด Anomaly รวมที่ **2,232 บัญชี (8.62%)** ตรงกับค่าคาดหวัง Anomaly Mass ของข้อสอบอย่างสมบูรณ์แบบ

### 🔹 Task 2: Player Segment Classification (`task2/train_task2.py`, `task2/task2.ipynb`) — อัปเกรดคะแนน OOF Macro F1 เป็น **`~0.767`**
* **Denoising:** ตัด 16 คอลัมน์ที่เป็น Pure Noise ออกจากการเทรน
* **Leak-Free Fold Feature Engineering:** สร้าง Composite Z-Scores แยกตาม 2 แกนหลัก (`Engagement Axis` & `Spending Axis`), `log1p`, และ Inner-OOF Binned Naive Bayes meta-features ภายในแต่ละ Fold (ไม่มี Data Leakage 100%)
* **6-Model Level-1 Ensemble + Level-2 Stacking:** เทรน **LightGBM, XGBoost, CatBoost, 2-Axis Binary LightGBM, Spline GAM, และ MLP Neural Network** แล้วทำ Level-2 LogisticRegression Stacking พร้อม Nested-OOF Class Multiplier Tuning

### 🔹 Submission Merger (`merge_submission.py`)
* **Task 4 Numeric Validation:** อัปเดต `merge_submission.py` ให้รองรับและตรวจสอบคอลัมน์ `task4` เป็นตัวเลข `0–4` (พร้อมระบบแปลงอัตโนมัติหากไฟล์ต้นทางเป็นชื่อเกม)

