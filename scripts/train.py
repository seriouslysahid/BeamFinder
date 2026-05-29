import shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    model = YOLO("yolo26s.pt")

    model.train(
        data="configs/dataset.yaml",
        epochs=100,
        patience=20,
        imgsz=960,
        rect=True,
        batch=0.90,
        optimizer="auto",
        cos_lr=True,
        box=10.0,
        degrees=0.0,
        shear=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        scale=0.5,
        translate=0.1,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        cache="ram",
        workers=0,
        deterministic=False,
        compile=True,
        project="runs",
        name="drone_detect_s23_v2",
        exist_ok=True,
        save_period=10,
    )

    metrics = model.val(imgsz=960, half=True)
    print(f"Val mAP50: {metrics.box.map50:.4f}")

    test_metrics = model.val(split="test", imgsz=960, half=True)
    print(f"Test mAP50: {test_metrics.box.map50:.4f}")

    # save best weights to root
    best_w = Path("runs/drone_detect_s23_v2/weights/best.pt")
    if best_w.exists():
        shutil.copy2(best_w, "best.pt")
        print("Saved best.pt to root")

if __name__ == "__main__":
    main()
