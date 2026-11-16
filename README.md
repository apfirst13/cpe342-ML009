# CPE 342 Machine Learning (1/2026) - Karena Player Intelligence System 🎮

Kaggle Competition: [Player Intelligence System - CPE342 ML](https://www.kaggle.com/competitions/cpe342-karena)  
กำหนดส่ง Kaggle: **24 พ.ย. 2026** | กำหนดส่ง Report & Code (LEB2): **30 พ.ย. 2026**

---

## 👥 การแบ่งหน้าที่ในทีม (Team Roles & Task Assignments)

| สมาชิก | Task ที่รับผิดชอบ | ชนิดของโมเดล / Metric | โฟลเดอร์ทำงาน |
| :---: | :--- | :--- | :---: |
| **คนที่ 1** | **Task 1:** Anti-Cheat Pre-Filter<br>**Task 5:** Account Security Monitoring | Binary Classification ($F_2$ Score)<br>Unsupervised Anomaly ($F_3$ Score) | `task1/`<br>`task5/` |
| **คนที่ 2** | **Task 4:** Game Title Detection | Image Classification (Macro $F_1$) | `task4/` |
| **คนที่ 3** | **Task 2:** Player Segment Classification | Multi-Class Classification (Macro $F_1$) | `task2/` |
| **คนที่ 4** | **Task 3:** Player Monthly Spending Prediction | Regression (Normalized MAE - NMAE) | `task3/` |

---

## 💻 คู่มือเริ่มต้นใช้งานสำหรับเพื่อนร่วมทีม (Team Setup Guide)

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

## 📂 การจัดการชุดข้อมูล (Dataset Setup)

> **⚠️ ข้อสำคัญ:** โฟลเดอร์ `data/` จะ**ไม่ถูกอัปโหลดขึ้น GitHub** (ถูกกันไว้ด้วย `.gitignore` เนื่องจากขนาดไฟล์ใหญ่ 1GB+) ทุกคนต้องดาวน์โหลดมาไว้ในเครื่องตนเอง

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

## 🛠️ ขั้นตอนการทำงานร่วมกันในทีม (Workflow)

1. **เขียนโค้ดในโฟลเดอร์ Task ของตัวเอง:**  
   เช่น คนที่ทำ Task 1 ให้เขียนโค้ดหรือ Notebook ไว้ในโฟลเดอร์ `task1/` เพื่อไม่ให้โค้ดชนกับเพื่อน
2. **ส่งออกผลลัพธ์การทำนาย (Prediction Export):**  
   เมื่อโมเดลทำนายผลชุดทดสอบเสร็จแล้ว ให้เซฟไฟล์ CSV ขนาด 25,889 แถว มี 2 คอลัมน์คือ `id` และคำตอบ มาวางไว้ที่โฟลเดอร์ `submissions/` โดยตั้งชื่อตาม Task ตัวเอง:
   * Task 1 $\rightarrow$ `submissions/sub_task1.csv` (คอลัมน์ `id`, `task1` ค่า 0 หรือ 1)
   * Task 2 $\rightarrow$ `submissions/sub_task2.csv` (คอลัมน์ `id`, `task2` ค่า 0, 1, 2, 3)
   * Task 3 $\rightarrow$ `submissions/sub_task3.csv` (คอลัมน์ `id`, `task3` ค่าเงินบาท THB $\ge 0$)
   * Task 4 $\rightarrow$ `submissions/sub_task4.csv` (คอลัมน์ `id`, `task4` ค่าชื่อเกม string)
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
