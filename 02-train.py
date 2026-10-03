from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO(
        r"D:\AI_YOLO\runs\thai_coin_yolo26s_v3\weights\best.pt"
    )

    results = model.train(
        data=r"D:\AI_YOLO\dataset\data.yaml",

        epochs=50,
        patience=12,
        imgsz=960,
        batch=8,
        device=0,

        optimizer="MuSGD",

        # ลด augmentation ลงจาก v3
        # เพื่อให้โมเดลเรียนรายละเอียดเหรียญจริงมากขึ้น
        degrees=10.0,
        shear=1.5,
        perspective=0.0002,

        fliplr=0.5,
        flipud=0.5,

        # ลดความแรงของ mosaic / mixup
        mosaic=0.5,
        mixup=0.0,

        # ปิด mosaic เร็วขึ้น
        close_mosaic=15,

        project=r"D:\AI_YOLO\runs",
        name="thai_coin_yolo26s_v4"
    )