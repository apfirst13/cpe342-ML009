# Task 5 Research Log — Account Security Monitoring

บันทึกการทดลองวิจัยแบบควบคุมตัวแปรสำหรับ **Task 5** บนชุดข้อมูล `data/task5/test.csv` จำนวน `25,889` บัญชี (`123` คอลัมน์) โดยอิงกติกาการแข่งขัน CPE342 Kaggle และเกณฑ์ประเมินโปรเจกต์ `CPE342_Project_Instruction.pdf` อย่างเคร่งครัด 100%

- **กำหนดส่ง Kaggle:** `24 พ.ย. 2026`
- **กำหนดส่ง LEB2 Report (20%) & Code (10%):** `30 พ.ย. 2026`
- **กติกาหลักที่ห้ามละเมิดในทุก Run:**
  1. เทรนจากศูนย์ด้วยข้อมูลใน `data/task5/test.csv` เท่านั้น (Pure Unsupervised)
  2. ห้ามใช้ข้อมูลภายนอก ห้ามทำ Hand-Labeling และห้ามใช้ AutoML Framework
  3. ห้ามเขียน Hard-Coded Rule Overrides ทับผลทำนายของโมเดล

---

## 1. ประวัติอ้างอิงก่อนเริ่มวิจัย V7

| เวอร์ชัน | แนวทางหลัก | สถิติอ้างอิงภายใน | จำนวน Flag | คะแนนรวม Kaggle LB | สถานะ |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **V4** | Pure ML ยุคแรก (IF + LGBM DR + Geo) | Ref10 `1,999/2,179` (`91.7%`), CombatVar `67.8%`, LowAct `82.3%`, NoComm `96.6%` | `~2,100+` | — | แทนที่แล้ว |
| **V5** | Pure ML + `Structural Domain Anchor` Override | Ref10 `2,179/2,179` (`100.0%`), เพิ่มเติม `53` แถว (ถ้าปิด Override เหลือ `1,887/2,179`) | `2,232` (`8.62%`) | `0.8224` (ช่วงนั้น ก่อน Task 3 ดันเป็น `0.83453`) | ถอด Rule ออกแล้วใน V6 |
| **V6** (`29b471c`) | Pure Unsupervised Multi-Paradigm (1D Topo KDE + 3D Combat KDE/DR + Mixture-MICE NaN Recovery + Noisy-OR + Bayes Expected F3) | Ref9 `1,794/1,794` (`100.0%`), CombatVar `299/385` (`77.7%`), Extra `99` แถว (รวม `other_ex = 43` แถว), Expected Mass `2,103.5` (`p >= 0.1546`), Mixture-MICE `bs` AUC `0.9816` | `2,192` (`8.47%`) | `0.83583` (ฐานหลังรวม Task 1 อัปเกรด) | **Baseline V6 (`old_task5.csv`)** |

### ข้อเท็จจริงสำคัญจาก Forensic Audit ของ V6 (`25,889` แถว)
- แถวที่ V6 ไม่ได้ flag จำนวน `23,697` แถว มี `Real-vs-Null OOF AUC = 0.48324`, `0` คู่ฟีเจอร์ระดับ Profile ที่มี `|r| > 0.04`, และการทำนาย Red Flags (`password_changed`, `mass_item_sale`, `vpn_usage`, `suspicious_login_time`) ได้ AUC `0.4990` ถึง `0.5110`
- กลุ่ม `other_ex` (`43` แถว) ที่ V6 flag เพิ่มเข้ามาพบ `41/43` มี `max |z| < 4.5` และ `password_changed = 0.0%`:
  - `20/43` แถว เกิดจาก `p_topo = 1.0` บน `1–4` sample tail-edge ripples และ `1-sigma` shoulder jitter จากการทำ quantile clipping Pile-up ใน `compute_1d_topological_valley_kde`
  - `18/43` แถว เกิดจาก `p_combat >= 0.16` ของ tree variance noise (`p_ex_c3 = 0.22–0.38`) และ boundary bleed ของ 3D KDE (`bw = 0.24`)
- กลุ่ม `CombatVar_missed` (`86` แถว) มีส่วนเกินทางสถิติในขอบล่างราว `~12` แถว โดย V6 ปัด `p_combat = 0.076` ทิ้งเป็น `0.000` ด้วย `clamp_floor(p_combat, 0.16)`

---

## 2. ตารางสรุปผลการทดลอง V7

| รอบ | โมเดล / สมมติฐาน | Flag (`%`) | Ref9 (`1,794`) | CombatVar (`385`) | `other_ex` | Diff เทียบ V6 (`+เพิ่ม / -ลด`) | Kaggle Total LB | ทิศทางผล |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **V6** | Baseline Archive (`submissions/old_task5/old_task5_v6.csv`) | `2,192` (`8.47%`) | `1,794` (`100.0%`) | `299` (`77.7%`) | `43` | Base | `0.83583` | Base |
| **V7-A** | Precision Cleanup 1D Topological KDE (Unclipped + Poisson Island `Z >= 2.0`) | `2,163` (`8.35%`) | `1,794` (`100.0%`) | `299` (`77.7%`) | `23` | `+1 / -30` | **`0.83611`** | **แชมป์ปัจจุบัน (`+0.00028`)** |
| **V7-B** | 2-Step EM Dual-BW 3D Combat KDE + 3D Poisson Gate (`z_3d >= 1.25`) + Multi-Paradigm Geometric Consensus | `2,165` (`8.36%`) | `1,794` (`100.0%`) | `322` (`83.6%`) | `0` | `+36 / -63` (`+36 / -34` เทียบ V7-A) | `0.83609` | **ลด `-0.00002` (ไม่ใช้ต่อ, ยึดฐาน V7-A)** |
| **V7-C** | ย้อนฐานกลับ V7-A (`0.83611`) + Surgical Tree-Noise Cleanup ไม่ดึงขอบต่ำ + 4-Period Peer Mixture-MICE (`bs` AUC `0.9816 -> 0.9852`) | *รอทดลองถัดไป* | — | — | — | — | — | รอดำเนินการ |

---

## 3. บันทึกรายละเอียดแต่ละ Experiment

### Exp 1 / V7-A — Precision Cleanup บน `compute_1d_topological_valley_kde`
- **วันที่:** `2026-11-21`
- **ไฟล์โค้ด & Submission:** [`task5/train_task5.py`](train_task5.py) / ไฟล์ส่งเพื่อน `submissions/sub_task5.csv` (สำรอง `task5/experiments/sub_task5_v7a.csv`) / ฐานเดิม `submissions/old_task5/old_task5.csv`
- **สมมติฐาน:** การแก้จุดอ่อนใน `compute_1d_topological_valley_kde` ด้วย 2 กลไกทางสถิติล้วนคือ การทำ Histogram เฉพาะในช่วงข้อมูล (`np.histogram(vo, bins=edges)`) เพื่อกันไม่ให้จุดหางสุดขีดไปกองที่ขอบบ่อ และการใช้ Poisson Statistical Island Excess Significance (`z_isl >= 2.0`) จะตัด False Positive จากคลื่นรบกวนที่หางกราฟออก โดยยังคงความครอบคลุมของกลุ่ม `Ref9 = 100.0%` และไม่ลดคุณภาพการกู้คืน Missing ใน `Mixture-MICE`
- **สิ่งที่แก้ในโค้ด:**
  - ปรับ `compute_1d_topological_valley_kde` บรรทัด `212–286` ใน `task5/train_task5.py` เพิ่มพารามิเตอร์ `min_island_z: float = 2.0`, ใช้ `np.histogram(vo, bins=edges)` แบบไม่ Clip, และกรอง Rebound Island ต่อเนื่องที่ `(isl_n - exp_val) / sqrt(isl_n + exp_val + 1) >= 2.0`
- **Rule Check:** Pure Unsupervised ML on `data/task5/test.csv`, Train from scratch (`91.3s`), Zero rule overrides, Zero external data/labels/AutoML
- **ผลรันและ Diff Audit เทียบ V6:**
  - `Estimated Expected Anomaly Mass`: `2,073.6` (`8.01%`), Cutoff `k* = 2,163` (`8.35%`), `min p = 0.1552`
  - `Ref9 (9 กลุ่มอ้างอิงหลัก)`: `100.0%` (`Bot 207/207`, `LowAct 265/265`, `NoComm 263/263`, `LowStab 246/246`, `Theft 463/463`, `Boost 299/299`, `VPN 165/165`, `LocHop 493/493`, `Rem4 297/297`)
  - `CombatVar Corner`: `299 / 385` (`77.7%`, เท่า V6)
  - `Mixture-MICE`: `behavior_stability OOF AUC = 0.9816` (`17` recovered), `geo_23 = 0.9990` (`42`), `geo_34 = 0.9996` (`102`), `d_cur = 0.9995` (`17`), `d_st = 0.9997` (`11`)
  - `other_ex`: ลดจาก `43` เหลือ **`23` แถว** (ตัด `20/20` แถวจาก `p_topo` ได้หมดสิ้น ที่เหลือมาจาก tree noise `p_combat` และ `p_global` สำหรับ Exp 2)
  - `Diff vs V6`: ซ้อนทับ `2,162` แถว / **เพิ่ม `+1` แถว** (`P005705`, `password_changed=1.0`) / **ตัดออก `-30` แถว** (`pwd = 3.4%`, `sale = 3.4%`, `susp = 3.4%`, ทั้งหมดเป็น Normal Tail Ripples ที่ค่าพฤติกรรมปกติทุกมิติ)
- **ผลคะแนน Kaggle LB:** **`0.83583` -> `0.83611`** (**ดันคะแนนรวมขึ้น `+0.00028`** หรือ `+0.028%`, เทียบเท่าประมาณ `+0.0014` บน Task 5 ที่น้ำหนัก 20%)
- **ข้อสรุปจาก Exp 1:** ยืนยันสมมติฐานสำเร็จ 100% — การตัด False Positive จากคลื่นรบกวนที่หางกราฟออก `-30` แถว พร้อมเก็บ `+1` แถวจริง (`2,192 -> 2,163`) ช่วยดันคะแนนรวม Kaggle ขึ้นจริง เก็บ `V7-A` เป็นฐานใหม่ (`0.83611`) แล้วเดินหน้า **Exp 2 / V7-B** ต่อ

---

### Exp 2 / V7-B — 2-Step EM Dual-Bandwidth 3D Combat KDE + 3D Poisson Excess Gate + Multi-Paradigm Geometric Consensus
- **วันที่:** `2026-11-21`
- **ไฟล์โค้ด & Submission:** [`task5/train_task5.py`](train_task5.py) / ไฟล์ส่งเพื่อน `submissions/sub_task5_v7-b.csv` (สำรอง `task5/experiments/sub_task5_v7b.csv`) / ฐาน V7-A ก่อนหน้า `submissions/old_task5/old_task5.csv` (`old_task5_v7a.csv`) / ฐาน V6 เดิม `submissions/old_task5/old_task5_v6.csv`
- **สมมติฐาน:**
  1. ในกลุ่ม `other_ex` (`23` แถวที่เหลือใน V7-A) พบว่าทุกแถวมีค่า 3D Kernel-Volume Poisson Excess Z-score `z_3d <= 0.96` (`< 1.0` ทั้ง `23/23` แถว) โดย `18` แถวเกิดจาก tree-split variance ของ `p_ex_c3` ในสเปซ 3 มิติที่เบาบาง และอีก `5` แถวเกิดจากใบต้นไม้เดี่ยวใน `p_global` โดยที่ตัวตรวจจับเชิงเรขาคณิตทั้ง 4 ตัว (`IsolationForest`, `k-NN`, `Temporal Delta`, `Autoencoder`) มีค่าเฉลี่ย `Z <= 1.02`
  2. ในทางกลับกัน กลุ่ม `CombatVar_missed` (`86` แถวขอบล่าง) มีส่วนเกินจริงที่มี `z_3d` สูงถึง `2.05` แต่ใน V6/V7-A โดนปัดทิ้งด้วย `clamp_floor(p_combat, 0.16)` และการทำ EM Marginal Debiasing แค่ 1 รอบยังไม่พอแยกความหนาแน่นขอบล่าง
  3. การใช้ **2-Step Iterative EM Marginal Debiasing** ร่วมกับ **Dual-Bandwidth (`0.18, 0.24`) 3D KDE** ที่กรองด้วย **3D Poisson Excess Gate (`z_3d >= 1.25`)** และ **Multi-Paradigm Geometric Consensus (`geom_supp >= 1.15`)** พร้อมลดจุดตัด `clamp_floor(p_combat, 0.12)` ให้สอดคล้องกับทฤษฎี Bayes F3 จะสามารถ **ตัด `other_ex` ลงเหลือ `0` แถว** พร้อม **ดึง True Positive ขอบ `CombatVar` ที่ถ่วงน้ำหนัก `9x` เพิ่มขึ้นมากกว่า `+20` แถว** ในรอบเดียว
- **สิ่งที่แก้ในโค้ด:**
  - อัปเกรด `compute_debiased_3d_kde_posterior` บรรทัด `333–393` ใน `task5/train_task5.py` ให้คำนวณ Dual-Bandwidth `(0.18, 0.24)`, ทำ Iterative EM Marginal Debiasing `2` รอบ, และคำนวณมวลตัวอย่างในปริมาตรเคอร์เนล 3 มิติ `obs_mass`, `exp_mass` เพื่อสร้างตัวกรองความมั่นใจ `z_3d = (obs_mass - exp_mass) / sqrt(obs_mass + exp_mass + 1)` ที่ `z_thresh = 1.25`
  - ปรับการรวม `p_combat` บรรทัด `535–539` เป็น `np.maximum(p_kde_c3, (0.5 * p_ex_c3 + 0.5 * p_kde_c3) * conf_c3)` และลด floor ใน Noisy-OR บรรทัด `624` จาก `0.16` เป็น `0.12`
  - เพิ่ม Multi-Paradigm Geometric Consensus บรรทัด `611–615` ถ่วงน้ำหนัก `p_global` ด้วย `geom_supp = np.maximum(views_global[2:6].mean(axis=0), z_3d_c3)` เพื่อไม่ให้ใบต้นไม้เดี่ยวที่ไม่มีการยืนยันจากโมเดลเรขาคณิตหลุดเป็น False Positive
- **Rule Check:** Pure Unsupervised ML on `data/task5/test.csv`, Train from scratch (`108.4s`), Zero rule overrides, Zero external data/labels/AutoML
- **ผลรันและ Diff Audit เทียบ V7-A และ V6:**
  - `Estimated Expected Anomaly Mass`: `2,081.8` (`8.04%`), Cutoff `k* = 2,165` (`8.36%`), `min p = 0.1207`
  - `Ref9 (9 กลุ่มอ้างอิงหลัก)`: **`100.0%` ครบทุกกลุ่ม** (`Bot 207/207`, `LowAct 265/265`, `NoComm 263/263`, `LowStab 246/246`, `Theft 463/463`, `Boost 299/299`, `VPN 165/165`, `LocHop 493/493`, `Rem4 297/297`, ไม่หลุดแม้แต่แถวเดียว `+0 / -0`)
  - `CombatVar Corner`: พุ่งจาก `299 / 385` (`77.7%`) ขึ้นเป็น **`322 / 385` (`83.6%`, เก็บเพิ่มสุทธิ `+23` บัญชี: `+24 / -1`)**
  - `Ref10_2179`: เพิ่มจาก `2,093 / 2,179` (`96.1%`) ขึ้นเป็น **`2,116 / 2,179` (`97.1%`)**
  - `other_ex (Pure-Noise FPs)`: ลดจาก `43` (V6) -> `23` (V7-A) -> **`0` แถว** (กำจัดเกลี้ยง `100%`)
  - `Diff vs V7-A`: ซ้อนทับ `2,129` แถว / **ตัดออก `-34` แถว** (`other_ex -23`, ค่า `mass_item_sale = 0.0%`, `password_changed = 3.1%` ต่ำกว่าค่าเฉลี่ยคนปกติ) / **เพิ่มเข้า `+36` แถว** (`CombatVar +24`, `NearCombat +11`, ค่า `mass_item_sale = 5.9%` สูงเกือบ 3 เท่าของค่าพื้นหลังปกติ `2.1%` และ `suspicious_login_time = 8.8%`)
  - `Diff vs V6`: ซ้อนทับ `2,129` แถว / **เพิ่มเข้า `+36` แถว** / **ตัดออก `-63` แถว** (`pwd = 1.7%`, `sale = 1.7%`)
- **ผลคะแนน Kaggle LB:** **`0.83611` -> `0.83609`** (**ลดลง `-0.00002`** เทียบกับ V7-A หรือ `-0.002%`, ยังสูงกว่าฐาน V6 เดิม `0.83583` อยู่ `+0.00026`)
- **การวิเคราะห์สาเหตุ Root Cause ที่ลดลง `-0.00002`:**
  1. **ดึงแถวขอบล่างที่ปะปนคนปกติสูงเกินไป (`+36` แถว):** ค่า `z_thresh = 1.25` ร่วมกับ `clamp_floor = 0.12` ดึงแถวขอบ `Combat` และ `NearCombat` ที่ `perf` ลงไปถึง `0.836` เข้ามาถึง `+36` แถว แต่มีอัตรา `password_changed = 3.2%` เท่านั้น บ่งชี้ว่าส่วนใหญ่ยังเป็นพื้นหลังปกติที่บังเอิญตกเข้ามาในกล่อง `Combat`
  2. **เผลอตัดแถวที่มีสัญญาณจริงออก `11` แถวนอก `other_ex`:** จากทั้งหมด `-34` แถวที่หายไปเทียบกับ V7-A แม้ `23` แถวจะเป็น `other_ex` แต่ `11` แถวที่เหลือมีเคสสำคัญหลุดไปด้วย เช่น `P005705` (`password_changed = 1.0`, `geo34_na`, ตัว `+1` ที่ช่วย V7-A ชนะ V6), `P015605` (`vpn_usage = 1.0`, `suspicious_login_time = 1.0`, `geo23/34_na`), `P012608` (`CombatVar_385 = True`) และกลุ่ม `behavior_stability NaN` (`3` แถว) เพราะ `conf_glob` และเพดาน `min p = 0.1207` ที่เปลี่ยนทำให้แถวขอบเหล่านี้โดนเบียดตกอันดับ
- **ข้อสรุปและการตัดสินใจ:** **ไม่เก็บ V7-B เป็นฐาน** ให้ **ยึด `V7-A` (`0.83611`) เป็นฐานแชมป์เดิม** แล้วออกแบบ **Exp 3 / V7-C** บนฐาน `V7-A` โดยตัด `other_ex` เฉพาะจุดที่ไม่แตะเคส `NaN`/`CombatVar` และอัปเกรด `Mixture-MICE` ด้วยฟีเจอร์ 4 คาบเต็ม (`behavior_stability` OOF AUC `0.9816 -> 0.9852`) แทนครับ
