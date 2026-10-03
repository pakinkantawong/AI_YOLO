# AI_YOLO

โปรเจคตรวจจับเหรียญไทยด้วย Ultralytics YOLO รองรับเหรียญ 1, 2, 5 และ 10 บาท

## ไฟล์หลัก

| ไฟล์ | หน้าที่ |
| --- | --- |
| `01-export_dataset.py` | แปลง annotation จาก Label Studio เป็น YOLO และแบ่ง train/validation ตามกลุ่มวิดีโอ |
| `02-train.py` | ฝึกโมเดลต่อจาก checkpoint ที่กำหนด |
| `03-test_image.py` | ทดสอบการตรวจจับจากภาพ |
| `04-test_video.py` | ทดสอบการตรวจจับจากวิดีโอ |
| `05-test-camera.py` | ตรวจจับแบบเรียลไทม์จากเว็บแคม |

## ติดตั้ง

ใช้ Python และสร้าง virtual environment จากโฟลเดอร์โปรเจค:

```powershell
python -m venv env
.\env\Scripts\python.exe -m pip install -r requirements.txt
```

สคริปต์ที่กำหนด `device=0` ต้องใช้ GPU ที่ PyTorch รองรับ และติดตั้ง PyTorch ให้ตรงกับระบบของเครื่อง

## เตรียมข้อมูลและใช้งาน

1. วาง JSON export จาก Label Studio ไว้ที่โฟลเดอร์หลัก และวางภาพต้นฉบับใน `images/` ตาม path ใน annotation โดยแยกโฟลเดอร์ตามกลุ่มวิดีโอ
2. รัน exporter เพื่อสร้าง `dataset/data.yaml` และข้อมูล train/validation คำสั่งนี้จะลบและสร้างโฟลเดอร์ `dataset/` ใหม่
3. เตรียม checkpoint ของโมเดล และตรวจ path ในสคริปต์ก่อน train หรือทดสอบ

```powershell
.\env\Scripts\python.exe 01-export_dataset.py
.\env\Scripts\python.exe 02-train.py
.\env\Scripts\python.exe 03-test_image.py
.\env\Scripts\python.exe 04-test_video.py
.\env\Scripts\python.exe 05-test-camera.py
```

สคริปต์ปัจจุบันใช้ path บนเครื่องเดิมที่ `D:\AI_YOLO` และ checkpoint ที่ `runs/thai_coin_yolo26s_v3/weights/best.pt` ให้แก้ path หากใช้ตำแหน่งอื่น โดย `02-train.py` เป็นการฝึกต่อจาก checkpoint เดิม

กด `q` เพื่อปิดหน้าต่างตรวจจับจากเว็บแคม

## ไฟล์ที่เก็บแยกในเครื่อง

Repository เก็บโค้ด, requirements และภาพทดสอบ `tests/frame_0010.jpg` ส่วน virtual environment, JSON annotation, ภาพ dataset, วิดีโอ, model weights และผลการ train ใน `runs/` ถูกยกเว้นด้วย `.gitignore` ต้องเตรียมไฟล์เหล่านี้แยกหลัง clone
