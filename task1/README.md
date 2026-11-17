# Task 1: Anti-Cheat Detection (Binary Classification)

## ข้อมูลทั่วไป
* **ผู้รับผิดชอบ (Author):** apfirst13 (คนที่ 1: Task 1 และ Task 5)
* **วันและเวลาที่บันทึก/ปรับปรุง:** 17 พฤศจิกายน 2026 เวลา 15:35 น. (UTC+7)
* **โจทย์:** จำแนกผู้เล่นที่เป็นคนโกง (Cheater: 1) ออกจากผู้เล่นปกติ (Legitimate Player: 0) ในระบบ Karena Player Intelligence System
* **เกณฑ์การวัดผล (Metric):** F2 Score (กำหนดค่าน้ำหนักให้ Recall สำคัญกว่า Precision 4 เท่า, beta = 2)
  $$F_2 = \frac{(1 + 2^2) \cdot P \cdot R}{2^2 \cdot P + R} = \frac{5 \cdot P \cdot R}{4P + R}$$

---

## ปัญหาที่ตรวจพบในชุดข้อมูล (Problems Identified)

1. **ปัญหาข้อมูลขาด Label ในชุดฝึก (Missing Labels in Training Data):**
   * ในไฟล์ train.csv มีข้อมูลทั้งหมด 60,407 แถว แต่พบค่าว่าง (NaN) ในคอลัมน์เป้าหมาย is_cheater สูงถึง 13,184 แถว หากนำไปเทรนโดยตรงจะทำให้โมเดลเกิด Error หรือเรียนรู้แบบผิดพลาด
2. **ปัญหาการใช้เกณฑ์การตัดสินใจแบบเริ่มต้น (Default Threshold Inefficiency):**
   * การใช้ค่าเกณฑ์ตัดสินใจมาตรฐานที่ Threshold = 0.50 ทำให้โมเดลเน้นความสมดุลระหว่าง Precision และ Recall ส่งผลให้ได้ค่า F2 Score เพียง 0.6121 เนื่องจากโมเดลปล่อยให้คนโกงหลุดรอดสายตาไปเป็นจำนวนมาก (Recall ต่ำ)
3. **ปัญหาความซับซ้อนของพฤติกรรมการโกง (Subtle Cheating Behaviors):**
   * พฤติกรรมของคนโกงยุคใหม่ไม่ได้แสดงความผิดปกติจากฟีเจอร์เดี่ยว เช่น ไม่ได้มีค่า KDR สูงผิดปกติเพียงอย่างเดียว แต่เกิดจากความสัมพันธ์ร่วมกันระหว่างเวลาตอบสนอง (Reaction Time), ความแม่นยำในการเล็ง (Crosshair Placement), และการคุมแรงดีด (Spray Control)

---

## แนวทางการแก้ไขและการปรับปรุง (Solutions Implemented)

1. **การคัดกรองข้อมูลฝึกสอน (Data Filtering):**
   * กรองแถวที่ไม่มี Label ออก (dropna subset=['is_cheater']) เพื่อให้ชุดฝึกสอนมีความถูกต้องและปราศจาก Noise เหลือข้อมูลสมบูรณ์ 47,223 แถว
2. **วิศวกรรมฟีเจอร์เชิงลึกสำหรับระบบตรวจจับการโกง (Domain-Specific Feature Engineering):**
   * **Aimbot & Shooting Indicators:**
     * `aimbot_index`: คำนวณผลคูณระหว่าง `headshot_percentage` และ `crosshair_placement`
     * `recoil_efficiency`: คำนวณผลคูณระหว่าง `spray_control_score` และ `accuracy_score`
     * `aim_vs_smoothness`: อัตราส่วนระหว่าง `accuracy_score` ต่อ `aiming_smoothness`
   * **Reaction Speed Anomalies:**
     * `reaction_speed_inv`: ส่วนกลับของเวลาตอบสนอง (1000 / reaction_time_ms)
     * `reaction_vs_accuracy`: อัตราส่วนความแม่นยำต่อเวลาตอบสนอง
   * **Smurf & Account Suspicion:**
     * `suspicion_per_age`: ดัชนีความน่าสงสัยเฉลี่ยต่ออายุบัญชี ((kdr * reports_received) / (account_age_days + 1))
     * `kd_per_level`: สัดส่วน KDR ต่อเลเวลผู้เล่น
     * `progression_vs_age`: ความเร็วในการอัปเลเวลเทียบกับอายุบัญชี
   * **Combat Efficiency & Bot Patterns:**
     * `damage_per_survival`: ปริมาณความเสียหายเฉลี่ยต่อเวลาที่รอดชีวิต
     * `clutch_vs_game_sense`: สัดส่วนการพลิกเกมเทียบกับความเข้าใจเกม
     * `missing_feature_count`: จำนวนข้อมูลที่สูญหายในแถว (แพทเทิร์นเฉพาะของโปรแกรมบอท)
3. **การสร้างโมเดลด้วย LightGBM และ Stratified 5-Fold Cross Validation:**
   * ใช้ `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` เพื่อรักษาการกระจายตัวของคลาสในทุก Fold
   * ปรับจูน Hyperparameters ของ `LGBMClassifier`:
     * n_estimators = 500
     * learning_rate = 0.03
     * num_leaves = 63
     * colsample_bytree = 0.8
     * subsample = 0.8
4. **การปรับจูนเกณฑ์การตัดสินใจแบบละเอียดเพื่อ F2 Score (Threshold Tuning):**
   * ค้นหาค่าเกณฑ์ที่เหมาะสมที่สุดในช่วง 0.10 ถึง 0.70 บน Out-of-Fold Predictions
   * พบจุด Optimum ที่ Threshold = 0.245 ซึ่งให้น้ำหนักกวาดจับคนโกงได้ครบถ้วนโดยไม่ปล่อยให้หลุดรอด

---

## ตารางเปรียบเทียบผลลัพธ์ (Performance Comparison)

| รายการ | โมเดลเริ่มต้น (Baseline) | โมเดลหลังทำ Feature Engineering + จูน Threshold |
| :--- | :--- | :--- |
| **จำนวนฟีเจอร์ที่ใช้** | 31 ฟีเจอร์ดิบ | 42 ฟีเจอร์ (เพิ่ม 11 Domain Features) |
| **Validation Strategy** | Stratified 5-Fold | Stratified 5-Fold |
| **Decision Threshold** | 0.50 (Default) | 0.245 (F2-Calibrated) |
| **5-Fold Out-of-Fold F2 Score** | **0.6121** | **0.8365 (+0.2244)** |
| **จำนวน Cheaters ที่ทำนาย (1)** | ประมาณ 8,200 บัญชี | 15,491 บัญชี |
| **จำนวน Legitimate ที่ทำนาย (0)** | ประมาณ 17,689 บัญชี | 10,398 บัญชี |

---

## ขั้นตอนการทำงานของโค้ดใน Notebook (task1_model.ipynb)

* **ส่วนที่ 1: Load Data & Preprocessing:** โหลดชุดข้อมูล train.csv, test.csv และกรองค่าว่างใน target
* **ส่วนที่ 2: Exploratory Data Analysis (EDA):** พล็อตเปรียบเทียบฟีเจอร์สำคัญระหว่างคนปกติกับคนโกง
* **ส่วนที่ 3: Baseline 5-Fold Training:** รันโมเดล LightGBM พื้นฐาน ได้ F2 เริ่มต้นที่ 0.6121
* **ส่วนที่ 4: Initial Threshold Search:** สแกนหา Threshold ที่ดีที่สุดรอบแรก (พบที่ 0.20 ได้ F2 = 0.8359)
* **ส่วนที่ 5: Advanced Feature Engineering:** สกัดฟีเจอร์ใหม่ 11 ตัวเพื่อจับพฤติกรรม Aimbot, Smurf และความเร็วตอบสนอง
* **ส่วนที่ 6: Retraining & Fine-Grained Threshold Tuning:** เทรนซ้ำบนฟีเจอร์ใหม่และปรับ Threshold ละเอียดที่ 0.245 ได้คะแนน F2 สูงสุดที่ 0.8365
* **ส่วนที่ 7: Export Predictions:** ส่งออกผลลัพธ์ลงในไฟล์ submissions/sub_task1.csv ขนาด 25,889 แถว
