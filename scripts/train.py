import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    model = YOLO("yolo26s.pt")

    model.train(
        data="configs/dataset.yaml",
        epochs=1000,
        imgsz=960,
        batch=0.90,
        patience=50,
        cache="ram",
        workers=0,
        cos_lr=True,
        deterministic=False,
        compile=True,
        project="runs",
        name="drone_detect_s23",
        exist_ok=True,
        rect=True,
        save_period=50,
        degrees=15.0,
        scale=0.9,
        translate=0.2,
    )

    metrics = model.val(imgsz=960, half=True)
    print(f"Val mAP50: {metrics.box.map50:.4f}")

    test_metrics = model.val(split="test", imgsz=960, half=True)
    print(f"Test mAP50: {test_metrics.box.map50:.4f}")

    # save best weights to root
    best_w = Path("runs/drone_detect_s23/weights/best.pt")
    if best_w.exists():
        shutil.copy2(best_w, "best.pt")
        print("Saved best.pt to root")

if __name__ == "__main__":
    main()
