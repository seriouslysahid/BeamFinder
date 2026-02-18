from ultralytics import YOLO

model = YOLO("yolo26n.pt")

if __name__ == "__main__":
    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=-1,
        patience=10,
        cache="disk",
        workers=2,
        amp=True,
        cos_lr=True,
        close_mosaic=10,
        freeze=10,
        save=True,
        project="runs",
        name="drone_detect",
    )

    # validation metrics
    metrics = model.val()
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
