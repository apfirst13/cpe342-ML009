# CPE342 Machine Learning - Karena Player Intelligence System

Kaggle Competition: [Player Intelligence System - CPE342 ML](https://www.kaggle.com/competitions/cpe342-karena)

## 📁 โครงสร้างโปรเจกต์ (Project Structure)
```text
├── data/                      # Dataset (แตก zip ไว้ที่นี่ - ห้าม commit ขึ้น git)
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   ├── task4/
│   ├── task5/
│   └── sample_submission.csv
├── task1/                     # Task 1: Anti-Cheat Pre-Filter (Binary Classification)
├── task2/                     # Task 2: Player Segment Classification (Multi-Class)
├── task3/                     # Task 3: Monthly Spending Prediction (Regression)
├── task4/                     # Task 4: Game Title Detection (Image Classification)
├── task5/                     # Task 5: Account Security Monitoring (Anomaly Detection)
├── submissions/               # โฟลเดอร์วางไฟล์ผลการทำนายของแต่ละคน
│   ├── sub_task1.csv
│   ├── sub_task2.csv
│   ├── sub_task3.csv
│   ├── sub_task4.csv
│   └── sub_task5.csv
├── merge_submission.py        # รันเพื่อรวมผลลัพธ์เป็น final_submission.csv
├── requirements.txt           # รายการแพ็กเกจ
└── README.md
```

## 🚀 วิธีเริ่มต้นใช้งาน (Getting Started)

### 1. ติดตั้ง Virtual Environment
```bash
python -m venv venv
# บน Windows PowerShell:
.\venv\Scripts\Activate.ps1
# ติดตั้ง dependencies:
pip install -r requirements.txt
```

### 2. นำ Dataset มาวางใน `data/`
ดาวน์โหลดไฟล์ `cpe342-karena.zip` จากหน้า Kaggle แล้วแตกไฟล์ไว้ในโฟลเดอร์ `data/`

### 3. การส่งผลลัพธ์รวม (Workflow การทำงานของทีม)
1. เมื่อสมาชิกในทีมเทรนโมเดลของ task ตัวเองเสร็จแล้ว ให้ export ไฟล์ผลการทำนายมาวางไว้ที่โฟลเดอร์ `submissions/` โดยตั้งชื่อ:
   - `submissions/sub_task1.csv`
   - `submissions/sub_task2.csv`
   - `submissions/sub_task3.csv`
   - `submissions/sub_task4.csv`
   - `submissions/sub_task5.csv`
   (รูปแบบข้างในมี 2 คอลัมน์คือ `id` และค่าที่ทำนายได้)
2. รันคำสั่งรวมไฟล์:
   ```bash
   python merge_submission.py
   ```
3. สคริปต์จะตรวจสอบความถูกต้อง (25,889 แถว, เช็ค NaN, เช็คค่าคลาส) และสร้างไฟล์ `final_submission.csv` ให้ นำไฟล์นี้ไป Submit บน Kaggle ได้ทันที!
