import cv2
from django.core.serializers import python
from ultralytics import YOLO


def main():
    # โหลดโมเดลที่ผ่านการฝึก
    model = YOLO(
        r"D:\AI_YOLO\runs\thai_coin_yolo26s_v3\weights\best.pt"
    )

    # เปิดกล้องเว็บแคม
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return

    print("กด 'q' เพื่อออกจากโปรแกรม")

    while True:
        success, frame = cap.read()

        if not success:
            print("ไม่สามารถอ่านภาพจากกล้องได้")
            break

        # ตรวจจับด้วย GPU
        results = model.predict(
            source=frame,
            conf=0.25,
            device=0,
            verbose=False
        )

        # วาด Bounding Box
        annotated_frame = results[0].plot()

        # แสดงผล
        cv2.imshow(
            "YOLO26 Real-time Detection",
            annotated_frame
        )

        # กด q เพื่อออก
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()