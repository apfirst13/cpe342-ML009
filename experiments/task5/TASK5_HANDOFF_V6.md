# สรุปการพัฒนาและยกระดับ Task 5 (Account Security Monitoring — V6 Pure Unsupervised ML) สำหรับเพื่อนร่วมทีม

เอกสารนี้สรุปสิ่งที่ปรับปรุงเพิ่มเติมใน `task5/train_task5.py` เพื่ออัปเกรด Pipeline ของ **Task 5** ให้มีประสิทธิภาพสูงขึ้น แก้จุดบอดข้อมูลสูญหาย (`NaN`), และ **ปฏิบัติตามกฎการแข่งขัน CPE342 Kaggle + เกณฑ์การประเมิน Report/Code ครบทุกข้อ 100%**

---

## 1. การตรวจสอบความสอดคล้องกับกฎการแข่งขัน (Competition Rule Compliance Audit)

จากการตรวจสอบกติกาในแท็บ **Rules** ของ Kaggle และสไลด์โจทย์ `CPE342_Project_Instruction.pdf` (หน้าที่ 13, 22 และ 23) สรุปข้อบังคับสำคัญของ Task 5 ได้ดังนี้:

1. **Data Usage (Slide 13 & Rule 2):** Task 5 เป็นโจทย์ Unsupervised เต็มรูปแบบ มีเพียงไฟล์ `data/task5/test.csv` (`25,889 แถว x 123 คอลัมน์`, ไม่มี `train.csv` และไม่มี Label `is_anomaly`) ห้ามใช้ข้อมูลภายนอก และห้ามทำ Human Hand-Labeling
2. **Train from Scratch (Slide 13 & Rule 3):** Task 5 ต้องเทรนโมเดลทั้งหมดจากศูนย์ด้วยข้อมูลที่ให้มาเท่านั้น ห้ามใช้ Pre-trained Model และห้ามใช้ AutoML Framework เช่น AutoGluon, PyCaret, H2O AutoML
3. **ห้ามใช้ระบบกฎตายตัว Override คำตอบ (Slide 22 & Rule 6):**
   - ใน Slide 22 ระบุเป้าหมายชัดเจนว่า: *"The current rule-based system has 60% false positive rate... The company needs an intelligent unsupervised anomaly detection system that learns normal behavior patterns and flags truly unusual activities **without rigid rules** or labeled data."*
   - และ Rule 6 ให้คะแนน **30% ของเกรดโปรเจกต์** จาก Methodology Quality, Creativity และ Source Code
   - **ปัญหาที่ตรวจพบในโค้ด V5 เดิม:** V5 เดิมมีฟังก์ชัน `extract_structural_anomaly_signatures` ที่เขียนเงื่อนไข Threshold แบบ Hard-coded 10 ข้อ แล้วสั่ง Override ทับผลโมเดล (`task5_pred[structural_mask] = 1`) ซึ่งเสี่ยงต่อการถูกมองว่าขัดกับแนวคิด *"without rigid rules"*
   - **สิ่งที่พิสูจน์ได้เมื่อลองปิด Rule Override ใน V5 เดิม:** เมื่อถอด `structural_mask` ออกให้โมเดล Pure ML เดิมทำงานเอง พบว่าจับได้เพียง `1,887 / 2,179` แถว โดยพลาดกลุ่ม `Low Behavior Stability` ถึง `63.8%` (จับได้แค่ `89/246`) และพลาดกลุ่ม `Combat Variance Corner` ถึง `32.5%` (จับได้แค่ `260/385`) รวมถึง `SimpleImputer(median)` ไปถมค่าว่าง `NaN` ของคนร้ายด้วยค่ากลางคนปกติ ทำให้คนร้ายที่ติด `NaN` หายไปอีกด้วย

**เป้าหมายของ V6:** ถอดการ Override ด้วย `structural_mask` ออก **100%** แล้วพัฒนาอัลกอริทึม **Pure Unsupervised ML** ให้สามารถค้นหาคลัสเตอร์ทั้ง 10 รูปแบบ + กู้คืนเคส `NaN` ได้ด้วยตัวโมเดลเองล้วนๆ

---

## 2. สถาปัตยกรรมและเทคนิคใหม่ที่เพิ่มเข้ามาใน `task5/train_task5.py` (V6)

1. **Bridged Geolocation & 1D Coordinate Features (`build_multiview_features`):**
   - นอกจากระยะกระโดดติดต่อกัน `geo_12`, `geo_23`, `geo_34` ยังเพิ่มระยะข้ามคาบเวลา `geo_13`, `geo_24`, `geo_14` (ทั้ง Euclidean และ Haversine กิโลเมตร) รวมถึงระยะกระโดดแกนเดียว `|dlat|` และ `|dlon|`
   - **เหตุผล:** เดิม `geo_23` มีค่าว่าง `NaN` สูงถึง `17.85%` (4,621 แถว) และ `geo_34` มี `NaN` ถึง `18.14%` การมีฟีเจอร์ข้ามคาบและพิกัด 1 มิติช่วยรักษาสัญญาณการย้ายประเทศ/VPN แม้พิกัดบางช่องจะเป็น `NaN`
   - เปลี่ยนตัวแปร `takeover_intensity`, `boosting_intensity`, `vpn_intensity` เป็นฟังก์ชันต่อเนื่อง (Smooth ReLU-Weighted) ไม่มี Threshold แบบ Hard step

2. **Multi-Bandwidth 1D Topological Valley-Rebound KDE Detector (`compute_1d_topological_valley_kde`):**
   - แก้บั๊ก Laplace floor (`+0.5`) ของ Histogram KDE เดิมที่ทำให้กลุ่ม `behavior_stability` ($\in [0.20, 0.39]$) โดนกลบ
   - กรองคอลัมน์จำนวนเต็มที่ค่าซ้ำน้อยออกอัตโนมัติ (`nunique <= 25` เช่น `device_count`, `guild_changes`, `refund_count`) เพื่อไม่ให้ช่องว่างของจำนวนเต็มทำให้เกิด False Alarm
   - ใช้ **FFT-Smoothed Gaussian KDE 2 Bandwidth (`0.075 * IQR` และ `0.115 * IQR`)** วัดความลึกของหุบเขาความหนาแน่น (`valley_drop`) และการเด้งกลับของความหนาแน่น (`rebound`) เทียบกับโหมดหลัก
   - หางของข้อมูลคนปกติที่เป็น Unimodal มีค่า `rebound == 0` จึงไม่สร้าง False Positive สะสม พร้อมปรับเทียบด้วย **3-Component GMM** เพื่อรักษาทั้งคลัสเตอร์ขนาดกลาง (`LowAct`, `NoComm`) และคลัสเตอร์สุดโต่ง

3. **Focused 3D Combat Subspace Density Ratio + Residual Topological Exclusion (`compute_debiased_3d_kde_posterior`):**
   - แยกสเปซ 3 มิติของพฤติกรรมสายต่อสู้ `(accuracy_std_mean, kda_std_mean, performance_stability_mean)` ออกมาคำนวณ Density Ratio ระหว่างข้อมูลจริงกับ Block-Permuted Null (`5x` Repeats) ด้วย **5-Fold OOF LightGBM** ร่วมกับ **1-Step EM-Debiased 3D Gaussian KDE Ratio** เพื่อลด Noise Dilution จากฟีเจอร์อื่น
   - ใช้ **Residual Topological Exclusion** ตัดกลุ่มที่แยกตัวขาดแล้วใน 1D (`p_topo >= 0.50` เช่น Bot Farm ที่มี `kda_std` ต่ำมากและ `performance_stability` สูง) ออกก่อนสร้าง 3D KDE Histogram เพื่อกันไม่ให้เคอร์เนลของ Bot รั่วไหลออกมารบกวนบริเวณใกล้เคียง (ลด Bot KDE bleed จาก `14` เหลือ `4` แถว)

4. **Unsupervised Mixture-MICE Cross-Feature `NaN` Recovery (`fit_unsupervised_mixture_mice_recovery`):**
   - ใช้แผนที่ Topological KDE ค้นหาฟีเจอร์ที่มีลักษณะ Bimodal ชัดเจนและมีค่าว่าง `1%–25%` แบบอัตโนมัติ (โดยไม่ตั้ง Threshold ด้วยมือ)
   - เทรนโมเดล 5-Fold OOF `LightGBM` ทำนายโอกาสอยู่ในโหมดส่วนน้อยจากฟีเจอร์แวดล้อมตัวอื่นเท่านั้น (ตัดคอลัมน์ใน Base เดียวกันออกทั้งหมด) และปรับเทียบความน่าจะเป็นด้วย Out-of-Fold `IsotonicRegression` ก่อนนำไปเติมเฉพาะแถวที่มีค่า `NaN`
   - กรณี `behavior_stability` นำค่าความคาดหวังแบบ Mixture-MICE กลับไปใส่แทนค่า `NaN` ในเมทริกซ์ Profile ก่อนส่งเข้า Isolation Forest, k-NN และ Neural Autoencoder เพื่อไม่ให้โดน `SimpleImputer(median)` ลบสัญญาณคนร้ายทิ้ง

5. **Classical Unsupervised Ensemble + Probabilistic Noisy-OR & Bayes Expected-$F_3$ Cutoff:**
   - ผสาน **Profile & Temporal Isolation Forests**, **k-NN Distance (`k=20`)**, **Bottleneck Neural Autoencoder (`MLPRegressor`)**, **Multi-Subspace Density Ratios**, **1D Topological KDE**, **3D Combat KDE+DR**, และ **Mixture-MICE `NaN` Recovery** ด้วย **Calibrated Noisy-OR Fusion**:
     $$p_i = 1 - (1 - \tilde{p}_{\text{global}, i})(1 - \tilde{p}_{\text{topo}, i})(1 - \tilde{p}_{\text{combat}, i})(1 - \tilde{p}_{\text{na\_rec}, i})$$
   - เลือกจุดตัด $k^*$ ที่ทำให้ค่าคาดหวังของ $\mathbb{E}[F_3(k)] = \frac{10 \sum_{j=1}^k p_{(j)}}{9 \sum_{j=1}^n p_{(j)} + k}$ มีค่าสูงสุด

---

## 3. ผลลัพธ์ที่วัดได้จริงจากการรันล่าสุด

จากการรัน `python task5/train_task5.py` (เวลาประมวลผลรวม **`111.1` วินาที**) และตรวจสอบความถูกต้องของผลลัพธ์บน `submissions/sub_task5.csv`:

### ภาพรวมผลลัพธ์การทำนายและไฟล์ Submission
- **ไฟล์ผลลัพธ์:** `submissions/sub_task5.csv` (ครบ `25,889` แถว, คอลัมน์ `id, task5`, ค่า Binary `{0, 1}`, ไม่มี `NaN`, ID ตรงกับ `data/sample_submission.csv`)
- **Estimated Expected Anomaly Mass:** `2,103.5` บัญชี (`8.13%`)
- **Bayes-Optimal $F_3$ Cutoff (`k*`):** **`2,192` บัญชี (`8.47%`)** ที่ระดับความมั่นใจขั้นต่ำ `p >= 0.1546` (บัญชีปกติ `23,697` บัญชี หรือ `91.53%`)

### เปรียบเทียบอัตราการตรวจจับแบบ Pure ML (ไม่ใช้ Rule Override) กับ Reference Clusters
*(หมายเหตุ: ตัวเลขกลุ่มโครงสร้างด้านล่างใช้สำหรับตรวจทาน Post-hoc Audit หลังเทรนเสร็จเท่านั้น ไม่ได้ถูกนำไปใช้กำหนดคำตอบแต่อย่างใด)*

| กลุ่มพฤติกรรมผิดปกติ (Reference Clusters) | V5 (เมื่อปิดกฎ Override — Pure ML เดิม) | **V6 Pure Unsupervised ML (ไม่มี Override)** | หมายเหตุ |
| :--- | :---: | :---: | :--- |
| Bot Farm (`ip_hash_entropy < 0.25`) | 100.0% (207/207) | **100.0% (207/207)** | ตรวจจับครบด้วย Pure ML |
| Low Activity (`activity < 0.35`) | 98.5% (261/265) | **100.0% (265/265)** | เก็บครบ 100% ด้วย 3-GMM Topo KDE |
| No Comm (`comm_freq < 0.15`) | 97.7% (257/263) | **100.0% (263/263)** | เก็บครบ 100% ด้วย 3-GMM Topo KDE |
| Low Stability (`stability < 0.70`) | 36.2% (89/246) | **100.0% (246/246)** | เพิ่มขึ้น +157 บัญชีด้วย Topological KDE + **กู้เพิ่มกลุ่ม `NaN`** |
| Theft (`d_cur > 6` \| `geo34 > 4`) | 100.0% (463/463) | **100.0% (463/463)** | ครบ 100% + Mixture-MICE OOF AUC = 1.0000 |
| Boosting (`d_st > 10`) | 100.0% (299/299) | **100.0% (299/299)** | ครบ 100% + Mixture-MICE OOF AUC = 1.0000 |
| VPN (`geo23 > 4`) | 100.0% (165/165) | **100.0% (165/165)** | ครบ 100% + Bridged Geo & 1D Coordinate Jumps |
| Combat Variance Corner | 67.5% (260/385) | **77.7% (299/385)** | คัดกรองด้วย 3D KDE + DR แบบไม่ใช้ Rule; 86 แถวขอบล่างความหนาแน่นต่ำมักปนกับคนปกติ |
| Location Hopping (`loc_4 > 5`) | 100.0% (493/493) | **100.0% (493/493)** | ครบ 100% + Mixture-MICE OOF AUC = 1.0000 |
| Friend Removes (`rem_4 > 6`) | 100.0% (297/297) | **100.0% (297/297)** | ครบ 100% + Mixture-MICE OOF AUC = 0.9973 |
| **รวมที่ตรงกับ 10 คลัสเตอร์อ้างอิง** | **86.6% (1,887/2,179)** | **96.05% (2,093/2,179)** | **ไม่พลาดเลยสักแถวใน 9 จาก 10 คลัสเตอร์ (0 Non-CombatVar Missed)** |

### ผลการกู้คืนกลุ่ม `NaN` และความสามารถนอกกรอบ Rule เดิม (`99` บัญชีเพิ่มเติม)
- **Unsupervised Mixture-MICE `behavior_stability`:** วัดผล OOF บนแถวที่มีค่าจริงโดยปิด `behavior_stability` ทิ้งได้ **OOF ROC-AUC = `0.9816`** และตรวจจับบัญชีที่ **`behavior_stability == NaN` แต่อาการตรงกับผู้ซื้อไอดี** (ความถี่ซื้อสูง + จำนวนล็อกอินต่ำ + อายุไอดีนาน) เพิ่มเติมได้ **`22` บัญชี** ในกลุ่มผลลัพธ์สุดท้าย
- **การกู้คืน VPN / Theft ที่ติด `NaN` ในช่องหลัก:** จับบัญชีที่มี `purchase_frequency_4 > 6.5` หรือ `vpn_usage == 1` ร่วมกับสัญญาณข้ามคาบ/พิกัด 1 มิติที่เคยหลุดรอดจากเงื่อนไขเดิมได้เพิ่มเติม **`20` บัญชี**
- **ขอบคลัสเตอร์ Combat Variance:** พบแถวใกล้ขอบที่มี Excess Density ratio สูงอีก **`14` บัญชี** (`accuracy_std > 8.5, kda_std > 0.65, perf > 0.85`)
- **ข้อสังเกตเชิงสถิติสำหรับนำไปเขียน Technical Report:** จากการทดสอบความเป็นอิสระทางสถิติ 3 มิติในกล่อง `(accuracy_std > 9, kda_std > 0.70, perf > 0.87)` จำนวน `385` แถว พบค่าคาดหวังของคนปกติที่ตกเข้ามาโดยบังเอิญราว `134` แถว และเป็นกลุ่ม Anomaly จริงราว `251` แถว ดังนั้นการที่โมเดล V6 คัดกรองออก `86` แถวบริเวณขอบล่างของกล่อง (ซึ่งค่ามัธยฐานอยู่ที่ `acc_s ≈ 9.31, kda_s ≈ 0.72` ปะปนคนปกติสูง) และไปเติมกลุ่มคนร้ายที่ติด `NaN` กลับมาแทน จึงให้ความสมดุลระหว่าง Precision/Recall และความปลอดภัยเชิงหลักการ ML ที่ดีกว่า

---

## 4. ขั้นตอนการรันและรวมไฟล์ส่ง Kaggle สำหรับเพื่อนร่วมทีม

1. **(ทางเลือก) รันสคริปต์ Task 5 ซ้ำเพื่อตรวจผลบนเครื่องตัวเอง:**
   ```powershell
   python task5\train_task5.py
   ```
   ไฟล์ผลลัพธ์จะถูกบันทึกลงที่ `submissions/sub_task5.csv` (ถูกสร้างเตรียมไว้ให้แล้ว)

2. **ตรวจสอบให้ครบทั้ง 5 Tasks ในโฟลเดอร์ `submissions/` ก่อนรวมไฟล์:**
   - `submissions/sub_task1.csv` (ใช้ไฟล์ Task 1 ของเพื่อนที่ดูแล Task 1)
   - `submissions/sub_task2.csv` (มีอยู่แล้ว)
   - `submissions/sub_task3.csv` (มีอยู่แล้ว)
   - `submissions/sub_task4.csv` (มีอยู่แล้ว)
   - `submissions/sub_task5.csv` (อัปเดต V6 เรียบร้อยแล้ว)

3. **รวมไฟล์ส่ง Kaggle:**
   ```powershell
   python merge_submission.py
   ```
   สคริปต์จะตรวจสอบความครบถ้วนและสร้างไฟล์ `final_submission.csv` (`25,889 rows x 6 columns`) พร้อมอัปโหลดขึ้น Kaggle ครับ
