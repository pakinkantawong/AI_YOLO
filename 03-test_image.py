from ultralytics import YOLO

# โหลดโมเดลที่ผ่านการฝึก (Trained Model)
model = YOLO(r"D:\AI_YOLO\runs\thai_coin_yolo26s_v3\weights\best.pt")

# กำหนดชื่อไฟล์วิดีโอที่ต้องการนำมาทดสอบ
video_to_test = "tests/frame_0010.jpg"  # สามารถเปลี่ยนเป็นไฟล์วิดีโออื่น ๆ ได้ตามต้องการ

# นำโมเดลไปทดสอบกับวิดีโอ
results = model.predict(
    source=video_to_test,   # กำหนดไฟล์วิดีโอที่ต้องการทดสอบ
    save=True,              # บันทึกวิดีโอผลลัพธ์หลังจากตรวจจับวัตถุ
    show=True,              # แสดงผลการตรวจจับวัตถุแบบเรียลไทม์ขณะประมวลผล
    conf=0.25                # กำหนดค่า Confidence ขั้นต่ำที่ 50%
)

# แสดงข้อความเมื่อการทดสอบเสร็จสิ้น
# วิดีโอผลลัพธ์จะถูกบันทึกไว้ในโฟลเดอร์ runs/detect/predict
print("ทดสอบเสร็จสิ้น! สามารถเข้าดูวิดีโอผลลัพธ์ได้ที่โฟลเดอร์ runs/detect/predict")