from ultralytics import YOLO


if __name__ == "__main__":

    # โหลดโมเดลที่เทรนแล้ว
    model = YOLO(
        r"D:\AI_YOLO\runs\thai_coin_yolo26s_v3\weights\best.pt"
    )

    # วิดีโอสำหรับทดสอบ
    video_to_test = r"D:\AI_YOLO\tests\IMG_7195.MOV"

    # ทำนายจากวิดีโอ
    results = model.predict(
        source=video_to_test,
        conf=0.25,
        save=True,
        show=True,
        device=0
    )

    print("ทดสอบเสร็จสิ้น")
    print("ดูผลลัพธ์ได้ในโฟลเดอร์ runs/detect/predict")