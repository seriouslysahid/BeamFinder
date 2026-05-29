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
        batch=64,  # Optimal for A100 40GB (AutoBatch found this)
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
        cache="ram",  # 40GB VRAM + dataset fits in RAM
        workers=16,  # A100 has 30 CPUs, use 16 workers for data loading
        device=0,
        deterministic=False,
        compile=True,
        amp=True,
        project="runs",
        name="drone_detect_s23_v2",
        exist_ok=True,
        save_period=10,
        plots=False,  # Disable plotting to save I/O time
        val=True,
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
