# Task 4 Research Log — Game Title Detection (Image Classification, Macro F1)

บันทึกการวิจัยและอัปเกรดโมเดลสำหรับ **Task 4: Game Title Detection** (`31,546` รูป Train, `25,889` รูป Test, ขนาด `230x120` RGB, 5 คลาสเกม) โดยยึดกติกาการแข่งขัน CPE342 Kaggle และ `task4/TASK4_BLUEPRINT.md` อย่างเคร่งครัด 100%

- **กติกาหลักที่ยึดถือในทุกการทดลอง:**
  1. อนุญาตให้ใช้ Pre-trained Vision Backbones จาก `timm` ได้เต็มที่
  2. ห้ามใช้ AutoML Framework
  3. ห้ามทำ Pseudo-labeling หรือใช้ข้อมูลจาก `test.csv` มาเทรนหรือจูนพารามิเตอร์ใดๆ (ทุกตัวคูณและพารามิเตอร์คำนวณจาก `train.csv` เท่านั้น)
  4. คำตอบในไฟล์ `.csv` ต้องเป็นตัวเลขจำนวนเต็ม `{0, 1, 2, 3, 4}` ครบ `25,889` แถว ไม่มีค่า `NaN`

---

## 1. การวินิจฉัยปัญหาของ Baseline เดิม (`sub_task4.csv` วันที่ 17 พ.ย.)

| รายการตรวจวัด | บนชุด Train OOF (`31,546` รูป) | บนชุด Test (`25,889` รูป) | ข้อสรุปเชิงวิศวกรรม |
| :--- | :---: | :---: | :--- |
| **ResNet34d vs EffNet-B0 Disagreement** | `3.48%` | **`14.95%` (`3,870` รูป)** | พุ่งขึ้น `4.3` เท่าจาก Domain Shift บนฉากใหม่และภาพย้อมสี |
| **Mean Max-Prob Confidence** | `0.9099` | **`0.7615`** | ความมั่นใจเฉลี่ยร่วงลง `-0.1484` |
| **สัดส่วนภาพความมั่นใจต่ำ (`< 0.70`)** | `6.84%` | **`35.05%` (`9,073` รูป)** | กว่า 1 ใน 3 ของชุด Test มีความไม่แน่นอนสูง |
| **Nearest-Neighbor Scene Similarity (`>= 0.85`)** | `13.8%` (Mean `0.7194`) | **`7.2%` (Mean `0.6702`)** | Test มาจากฉากเกม/วิดีโอชุดใหม่ที่ไม่ซ้ำกับ Train |
| **Heavy RGB Channel Tint (`chan_cast` P90 / P95)** | `82.82` / — | **`109.12` / `137.42`** | Test มีย้อมสี RGB รุนแรงกว่า Train (`cast > 55`: `27.76%` vs `20.40%`) |
| **Class Distribution (`Class 0..4 %`)** | `[39.4, 14.0, 17.7, 18.8, 10.1]` | `[21.4, 20.1, 14.2, 26.1, 18.3]` | เกิด Label Shift รุนแรง (`Class 0` ลดครึ่ง, `Class 4` เพิ่มเกือบเท่าตัว) |
| **บั๊กในโค้ดเดิม (`17 พ.ย.`)** | `EffNet-B0` มี `148` แถวเป็น `NaN` | `Phase 5` คูณ `[1.1588, ..., 0.8584]` + `0.15 * VFlip` TTA | ตัวคูณเก่าดัน `Class 0` ขึ้นและกด `Class 4` ลง สวนทางกับ Label Shift |

---

## 2. ตารางสรุปผลการทดลอง Task 4

| รอบ | โมเดล / แนวทาง | **Balanced-Prior OOF F1** | **Disagree Subset F1 (Acc)** | **Low-Conf (`<0.75`) F1** | Test Class Dist (`0..4 %`) | Diff เทียบ V0 | ไฟล์ส่งเพื่อน | Kaggle Total LB |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **V0** (`17 พ.ย.`) | `ResNet34d + EffNetB0` + Imbalanced Train Multipliers (`[1.1588..0.8584]`) | `0.97131` | `0.6403` (`65.27%`) | `0.7510` | `[22.6, 19.1, 14.3, 26.6, 17.4]` | Base | `submissions/old_task4/old_task4_v0_17nov.csv` | `0.83611` |
| **T4-Exp1** | แก้ `NaN` 148 แถว + ถอดตัวคูณผิดทาง + Train-Only Prior Debias (`tau=0.50`) | `0.97239` (`+0.00108`) | `0.5795` (`58.59%`) | `0.7228` | `[18.7, 21.4, 14.4, 25.4, 20.0]` | `1,643` รูป (`6.35%`) | `submissions/sub_task4_v1_prior.csv` | สำรองเทียบ |
| **T4-Exp2** | De-Tinted 3-View (`MobileNetV3-Large + ConvNeXt-Nano-IN12k`) PCA-MLP + 24 Distortion Feats + LGBM/LR Stacker | `0.97326` (`+0.00195`) | — | — | `[19.9, 21.0, 14.4, 26.2, 18.6]` | — | `submissions/sub_task4_v2_ensemble.csv` | พัฒนาต่อเป็น Exp3 |
| **T4-Exp3** | **Full 1,944-D GPU Residual-MLP (`0.9142` เดี่ยว) + 24 Distortion Feats + `CatBoost + LGBM + LogReg` Meta-Ensemble** | **`0.97355` (`+0.00224`)** | **`0.6492` (`65.45%`)** | **`0.7522`** | **`[20.0, 20.9, 14.5, 26.6, 18.1]`** | **`1,323` รูป (`5.11%`, `1,090` รูปในจุดที่ 2 โมเดลเดิมขัดกัน)** | **`submissions/sub_task4_v3_detinted.csv`** (และ `submissions/sub_task4.csv`) | **`0.83963` (`+0.00352`)** |
| **T4-Exp4 – Exp6** | **Colab L4 + Local Calibrated 3-Paradigm Ensemble (`ConvNeXt-Small` + `Swin-Tiny` End-to-End 5-Fold บน Unseen-Scene CV + `5-Foundation DINOv2/SigLIP2/EVA02-CLIP/ConvNeXt-T/EffNetV2-S` MLP)** | **Unseen-Scene `0.98036` (`Bal 0.97909`, `+0.1062` จากฐาน V3 `0.87290`)** | **Test Disagree `ConvNeXtS != SwinT` ลดเหลือ `9.23%`** | **`Dark<45: 0.9403` (`+0.2044`), `Rot: 0.9695` (`+0.1220`), `Noise: 0.9779`** | **`[16.2, 23.2, 15.4, 25.9, 19.3]`** | **`3,481` รูปเทียบ V3 (`13.45%`)** | **`submissions/sub_task4_v5_cloud_ensemble.csv`** (และอัปเดต `submissions/sub_task4.csv`) | **`0.85209` (`+0.01246` จาก V3)** |

*(หมายเหตุ: ผลคะแนนรวม Kaggle LB `0.83611 -> 0.83963` หรือเพิ่มขึ้น `+0.00352` เทียบเท่ากับการดันคะแนนเฉพาะ Task 4 ขึ้นประมาณ **`+0.0176` หรือ `+1.76%`** จากน้ำหนัก 20%)*

---

## 3. รายละเอียดเชิงวิศวกรรมของ T4-Exp2 และ T4-Exp3

### 3.1 การแก้สีย้อมและภาพมืดจัดด้วยสมการ Per-Image Algebraic De-Tinting (`gpu_detint_and_normalize`)
- เนื่องจากภาพถูกย้อมสีเชิงเส้น $I'_c = \alpha_c I_c + \beta_c$ จึงคำนวณค่าเฉลี่ย $\mu_c$ และส่วนเบี่ยงเบนมาตรฐาน $\sigma_c$ เฉพาะพิกเซลเนื้อภาพที่ไม่ใช่ขอบดำจากการหมุน ($\max(R,G,B) \ge 12/255$) แล้วแปลงกลับสู่การกระจายตัวมาตรฐาน ($\mu = 0.45, \sigma = 0.23$) ทำให้ตัวคูณย้อมสี $\alpha_c$ และความมืด $\beta_c$ หักล้างกันทางพีชคณิตโดยสมบูรณ์
- สกัดฟีเจอร์ 3 มุมมอง (Raw + De-Tinted + De-Tinted HFlip) จาก 2 สถาปัตยกรรมคนละตระกูลกับโมเดลเดิม:
  1. `mobilenetv3_large_100.ra_in1k` (`1,280` มิติ)
  2. `convnext_nano.in12k_ft_in1k` (`640` มิติ)
  รวมกับฟีเจอร์กายภาพของความเพี้ยนภาพอีก `24` มิติ (`chan_cast`, `chroma`, `black_border`, `lap_std`, ความต่างแสง/สีครึ่งบน-ครึ่งล่างของ UI เกม) รวมเป็น `1,944` มิติ

### 3.2 Full-Dimension GPU Residual-MLP & Confidence-Adaptive Meta-Stacker (`task4/run_t4_exp3_gpu_meta_ensemble.py`)
- เทรน `ResidualEmbeddingNet` (LayerNorm + SiLU + Residual Bottleneck + Skip Connection + MixUp + Balanced Weighting) บนฟีเจอร์แก้สีย้อม `1,944` มิติ แบบ 5-Fold Out-of-Fold ได้คะแนนเดี่ยว **`OOF Macro F1 = 0.91416`** โดยไม่ต้อง Fine-tune Backbone
- นำความน่าจะเป็นของโมเดลแก้สีย้อมไปผสานกับ `ResNet34d` และ `EfficientNet-B0` (ที่แก้บั๊ก `148` NaN แล้ว) ด้วย **Inverse-Entropy Confidence-Adaptive Logit Fusion** ร่วมกับ **5-Fold `CatBoost + LightGBM + LogisticRegression` Meta-Stacker** ที่เรียนรู้ว่าภายใต้สภาวะย้อมสีหนัก มืดจัด หรือมีขอบหมุน โมเดลตัวใดแม่นยำกว่ากัน
- **ผลลัพธ์:**
  - ชนะ V0 เดิมครบทุกเกณฑ์วัดผล: **Balanced-Prior OOF Macro F1 เพิ่มเป็น `0.97355` (`+0.00224`)**, **Disagreement Subset Macro F1 เพิ่มจาก `0.6403` เป็น `0.6492` (`+0.0089`)**, และ **Low-Confidence Subset Macro F1 เพิ่มเป็น `0.7522`**
  - บนชุด Test (`25,889` รูป) เปลี่ยนคำตอบไปรวม **`1,323` รูป (`5.11%`)** โดย **`1,090` รูป (`82.4%` ของที่เปลี่ยนทั้งหมด)** อยู่ในกลุ่ม `3,871` รูปที่ `ResNet34d` และ `EfficientNet-B0` เคยทายขัดแย้งกันโดยตรง!
  - **ผลคะแนน Kaggle LB จริง:** ดันคะแนนรวมพุ่งจาก **`0.83611` เป็น `0.83963` (`+0.00352` คะแนนรวม / `+0.0176` โดยประมาณบน Task 4)** ขึ้นมาจ่อติด `0.8400` แล้ว เหลือเพียง `+0.00037`!

---

## 4. รอบ T4-Exp4 ถึง T4-Exp6 — Cloud L4 Multi-Foundation + Domain-Hardened E2E Ensemble

### 4.1 Phase 0: Golden Unseen-Scene 5-Fold Benchmark (`task4/phase0_unseen_scene_benchmark.py`)
- สร้างชุดแบ่งกลุ่มฉากภายใน `train.csv` เพียงอย่างเดียว (`418` กลุ่มฉาก เฉลี่ย `~75` รูป/กลุ่ม) เพื่อให้ Validation Fold เจอเฉพาะฉากใหม่ที่ใกล้เคียงชุด Test (`Val->Train NN >= 0.85` ลดจาก `10.95%` เหลือ `5.03%`) โดยยังคงสัดส่วนคลาสสมดุลทุก Fold (`[6292, 6314, 6311, 6312, 6317]`)
- วัดผลชุดฟีเจอร์ V3 เดิม (`1,944-D`) บน Golden Unseen-Scene 5-Fold ได้ `BalF1 = 0.87290` และพบจุดอ่อนหลักที่ `Dark < 45` ได้เพียง `0.7359`, `Border > 0.15` ได้ `0.8475`, และ `Noise > 90` ได้ `0.8457`

### 4.2 Phase 1 + Phase 2 บน Google Colab L4 (`task4/task4_cloud_t4_v4.ipynb`)
- พัฒนาฟังก์ชันกู้ภาพ 4 มุมมองแบบรายรูปอิสระ (`gpu_denoise_restore_base`) ได้แก่ `Median 3x3` ลดเกรนนอยส์ก่อนดึงแสง, ปรับ Gamma สำหรับภาพมืดจัดแบบ Noise-Safe ไม่ขยายนอยส์สีรุ้ง, De-Tinting เฉพาะพิกเซลเนื้อภาพ, `86%` Center-Crop ตัดมุมสามเหลี่ยมดำจากการหมุน, และมุมมองโครงสร้างแสงเงาขาวดำไร้เฉดสี (`Hue-Free Luminance Structure`)
- **T4-Exp4 (5 Vision Foundation Backbones):** `DINOv2-Small`, `SigLIP2-Base`, `EVA02-CLIP-Base`, `ConvNeXt-Tiny-IN22k`, และ `EffNetV2-S-IN21k` บน `folds_scene` ดัน Unseen-Scene All/Bal F1 ขึ้นมาที่ **`0.94045 / 0.93903`** (`Dark < 45` พุ่งเป็น `0.8408`)
- **T4-Exp5 (End-to-End Domain-Invariant Fine-Tuning บน `folds_scene`):** เพิ่ม GPU Affine Rotation `±35°` ขอบดำ, Channel Permutation, Extreme Dark Gamma, Shot Noise, Cutout, MixUp, Softmax แบบ FP32, และ 4-View TTA โดยตัด Vertical Flip กลับหัวทิ้ง
  - `mobilenetv3_large_100.ra_in1k`: Unseen All/Bal F1 = **`0.94789 / 0.94708`** (`D45 = 0.8677`)
  - `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k`: Unseen All/Bal F1 = **`0.97451 / 0.97313`** (`D45 = 0.9332`, Test Conf `0.8205`)
  - `convnext_small.fb_in22k_ft_in1k`: Unseen All/Bal F1 = **`0.97765 / 0.97672`** (`D45 = 0.9421`, `Rot = 0.9658`, `Noise = 0.9745`, Test Conf `0.8371`)

### 4.3 Phase 3: PyTorch Temperature Calibration & 3-Paradigm Ensemble (`task4/run_phase3_cloud_meta_ensemble.py`)
- ปรับเทียบอุณหภูมิรายโมเดลด้วย PyTorch LBFGS (`BalLogLoss` ลดลงทุกตัว เช่น `ConvNeXt-Small: 0.1068 -> 0.0804`, `Swin-Tiny: 0.1202 -> 0.0840`) และวัดเทียบทั้ง **Nested Calibrated 7-Model Logit Blend (`All = 0.98016, Bal = 0.97866`)**, **Meta-CatBoost (`0.97846 / 0.97852`)**, **Meta-LightGBM (`0.97851 / 0.97774`)**, และ **3-Family Calibrated Logit (`0.50 ConvNeXt-Small + 0.35 Swin-Tiny + 0.15 Cloud-5-Foundation MLP`)**
- ตัวที่ชนะทุกเกณฑ์ Unseen-Scene สูงสุดคือ **3-Family Calibrated Logit (`AllF1 = 0.98036`, `BalF1 = 0.97909`, `Dark<45 = 0.9403`, `Border>0.15 = 0.9695`, `Noise>90 = 0.9779`)**
- ลด Test Disagreement ระหว่างสถาปัตยกรรม (`ConvNeXt-Small != Swin-Tiny`) ลงเหลือเพียง **`9.23%`** (เทียบกับ `ResNet34d != EffNet-B0` เดิม `14.95%`), เปลี่ยนคำตอบจาก V3 รวม **`3,481` รูป (`13.45%`)** โดยตรวจสอบภาพที่เปลี่ยนด้วยตาเปล่าแล้วยืนยันว่ากู้ภาพ `FiveN`, `Roblock`, `Free Fried`, และ `PUBE` ที่เคยตกเป็น `7-11 Knights` กลับคืนได้ถูกต้องชัดเจน
- บันทึกผลลัพธ์ตรวจสอบครบ `25,889` แถว ไร้ `NaN` ลงไฟล์ **`submissions/sub_task4_v5_cloud_ensemble.csv`** และอัปเดต `submissions/sub_task4.csv` เรียบร้อยแล้ว


