# Task 1: Anti-Cheat Detection (Binary Classification)

## ข้อมูลทั่วไป
* **ผู้รับผิดชอบ (Author):** apfirst13 (คนที่ 1: Task 1 และ Task 5)
* **เวอร์ชันล่าสุด:** v3 Ultimate Pipeline (19 พฤศจิกายน 2026)
* **โจทย์:** จำแนกผู้เล่นที่เป็นคนโกง (Cheater: 1) ออกจากผู้เล่นปกติ (Legitimate Player: 0) ในระบบ Karena Player Intelligence System
* **เกณฑ์การวัดผล (Metric):** F2 Score (กำหนดค่าน้ำหนักให้ Recall สำคัญกว่า Precision 4 เท่า, $\beta = 2$)
  $$F_2 = \frac{(1 + 2^2) \cdot P \cdot R}{2^2 \cdot P + R} = \frac{5 \cdot P \cdot R}{4P + R}$$

---

## ปัญหาที่ตรวจพบในชุดข้อมูล (Problems Identified)

1. **ปัญหาข้อมูลขาด Label ในชุดฝึก (Missing Labels in Training Data):**
   * ในไฟล์ `train.csv` มีข้อมูลทั้งหมด 104,000 แถว แต่พบค่าว่าง (NaN) ในคอลัมน์เป้าหมาย `is_cheater` สูงถึง 13,184 แถว หากตัดทิ้งไปทั้งหมดจะเสียข้อมูลพฤติกรรมที่มีประโยชน์ไป 12.7%
2. **ปัญหาการใช้เกณฑ์การตัดสินใจแบบเริ่มต้น (Default Threshold Inefficiency):**
   * การใช้เกณฑ์ตัดสินใจมาตรฐานที่ Threshold = 0.50 ส่งผลให้ได้ค่า F2 Score เพียง 0.6121 เนื่องจากโมเดลปล่อยให้คนโกงหลุดรอดสายตาไปเป็นจำนวนมาก (Recall ต่ำ)
3. **ปัญหาความไม่สมดุลของคลาสและความสำคัญของเมทริกซ์ (Asymmetric Metric Optimization):**
   * เป้าหมายของ $F_2$ คือการจับคนโกงให้ได้ครบถ้วน (Recall สำคัญกว่า 4 เท่า) ดังนั้นการเทรนด้วย Cross-Entropy ธรรมดาแบบไม่ปรับ Class Weights จะทำให้โมเดลไม่ให้ความสำคัญกับคลาส 1 ในบริเวณ Boundary
4. **ความซับซ้อนของพฤติกรรมการโกงและการขาดหายของข้อมูล (Subtle Cheating Behaviors & Missing Patterns):**
   * คนโกงยุคใหม่ไม่ได้โกงแบบโจ่งแจ้ง แต่จะเกิดจากความผิดปกติสัมพันธ์กันระหว่างความแม่นยำ (Mechanics) กับความเข้าใจเกม (Tactics) รวมถึงเวลาตอบสนองที่หลุดขอบเขตมนุษย์

---

## สถาปัตยกรรมและการปรับปรุงขั้นสูงสุด (Ultimate Solutions Implemented)

### 1. Extended Domain & Behavioral Feature Engineering (รวม 66 ฟีเจอร์)
* **Inhuman Mechanics & Physical Impossibility Indicators:**
  * `inhuman_reaction_hs`: แฟล็กตรวจสอบเวลาตอบสนองต่ำผิดปกติ (`reaction_time_ms < 130ms`) ร่วมกับยิงเข้าหัวสูง (`headshot_percentage > 50%`)
  * `inhuman_aim_score`: ผลคูณระหว่างความแม่นยำ การคุมแรงดีด และการวางเป้า (`accuracy * spray_control * crosshair_placement`)
  * `reaction_aim_product`: สัดส่วนความแม่นยำและเฮดช็อตเทียบกับเวลาตอบสนอง
* **Mechanics vs Tactics Index:**
  * `mechanics_index`: ค่าเฉลี่ยความสามารถเชิงกลไกการยิง (Accuracy + Spray Control + Crosshair Placement)
  * `tactics_index`: ค่าเฉลี่ยความเข้าใจเกมและแท็กติก (Game Sense + Team Play + Buy Decision + Map Knowledge)
  * `mech_vs_tactics`: สัดส่วนระหว่าง Mechanics ต่อ Tactics (คนโกงมักมีค่านี้สูงโด่ง)
* **Account Suspicion & Smurf Detection:**
  * `smurf_risk_composite`: อัตราส่วนความเสี่ยงบัญชีผี/บอทคำนวณจากรีพอร์ตและ KDR เทียบกับอายุบัญชีและจำนวนเพื่อน
  * `suspicion_per_age`, `reports_per_level`, `device_churn_rate`
* **Relative Statistics by Level Quantiles (Z-scores):**
  * แบ่งผู้เล่นออกเป็น 10 Quantiles ตาม `level` เพื่อเปรียบเทียบพฤติกรรมในระดับเลเวลเดียวกัน
* **Missing Value Indicators:**
  * สร้างแฟล็ก `_isna` สำหรับฟีเจอร์สำคัญ เพื่อให้โมเดลจับแพทเทิร์นการหลีกเลี่ยงการบันทึกข้อมูลของโปรแกรมโกง

### 2. Loss & Class Weight Tuning (`scale_pos_weight = 2.0`)
* ปรับแต่งน้ำหนักของคลาสเป้าหมายให้โมเดลต้นไม้ลงโทษ False Negative หนักขึ้น สอดคล้องกับ $F_2$ Metric ทำให้โมเดลขยายพื้นที่การกวาดจับคนโกงได้คมขึ้น

### 3. Semi-Supervised Learning (Pseudo-Labeling บนข้อมูล 13,184 แถว)
* ใช้โมเดล Ensemble ทำนาย Probability บน 13,184 แถวที่ไม่มี Label ใน `train.csv`
* คัดเลือกเฉพาะแถวที่มีความมั่นใจสูงเป็นพิเศษ ($p \ge 0.96$ เป็นคนโกง 1,049 ตัวอย่าง และ $p \le 0.04$ เป็นคนปกติ 3,277 ตัวอย่าง รวม **4,326 ตัวอย่าง**)
* นำข้อมูล Pseudo-labels เข้าไปเสริมในชุดฝึกสอนเฉพาะ Training Folds (ไม่รั่วไหลเข้า Validation) และเทรน LightGBM ซ้ำ ส่งผลให้ $F_2$ ขยับขึ้นทันที

### 4. Multi-Model 5-Fold Stratified Blending (Nelder-Mead Optimized)
* ผสมผสาน 3 สถาปัตยกรรมโมเดลชั้นนำ:
  1. **LightGBM (Pseudo-labeled)**
  2. **XGBoost (Depth-wise Trees)**
  3. **CatBoost (Symmetric Trees)**
* ค้นหาน้ำหนักที่ดีที่สุดด้วยอัลกอริทึม Nelder-Mead บน Out-Of-Fold Predictions:
  $$\text{Final Pred} = 0.3901 \cdot \text{LGB}_{\text{pseudo}} + 0.3194 \cdot \text{XGB} + 0.2905 \cdot \text{Cat}$$
* ค้นหาจุดตัด Threshold ละเอียดระดับ 0.001 จนพบจุด Optimum ที่ **Threshold = 0.3060**

---

## ตารางเปรียบเทียบผลลัพธ์การทดลอง (Experimentation Roadmap)

| โมเดล / เวอร์ชัน | จำนวนฟีเจอร์ | Hyperparameters / กลยุทธ์ | Optimal Threshold | Recall | Precision | OOF $F_2$ Score |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **1. Baseline LightGBM** | 31 | ค่าเริ่มต้น (Default Threshold 0.50) | 0.5000 | 0.4412 | 0.7231 | 0.6121 |
| **2. Single LightGBM + Feature Eng** | 42 | ปรับ Threshold 0.245 บนฟีเจอร์โดเมน | 0.2450 | 0.9785 | 0.5284 | 0.8365 |
| **3. Multi-Model Ensemble (v2)** | 53 | Blending คงที่ (0.50 LGB + 0.30 XGB + 0.20 Cat) | 0.2550 | 0.9732 | 0.5365 | 0.8370 |
| **4. Ultimate Pipeline (v3)** 🏆 | **66** | **Pseudo-Labeling (4,326 ตัวอย่าง) + Nelder-Mead Blend** | **0.3060** | **0.9788** | **0.5304** | **0.83727** |

*หมายเหตุ: ROC-AUC ของโมเดลสุดท้ายอยู่ที่ **0.89099***

---

## การนำไปใช้งานและไฟล์ผลลัพธ์ (Deliverables & Code Usage)

1. **สคริปต์เทรนอัตโนมัติ:**
   * สคริปต์หลัก: `python task1/train_best_task1.py` รัน End-to-End ครบทุกขั้นตอน
2. **ไฟล์ Submission ที่ได้:**
   * `submissions/sub_task1.csv` (ขนาด 25,889 แถว)
   * อัตราส่วนการทำนายในชุดทดสอบ: คนโกง (1) = **15,784 คน (60.97%)** | ผู้เล่นปกติ (0) = **10,105 คน (39.03%)**
3. **การรวมเข้า Submission รวม:**
   * รัน `python merge_submission.py` เพื่ออัปเดตไฟล์ `final_submission.csv` พร้อมส่ง Kaggle ทันที
