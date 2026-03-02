from pathlib import Path

import torch
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    # A100: maximize GPU throughput
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # auto-tune convolutions for fixed imgsz=960

    model = YOLO("yolo26s.pt")

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=960,
        batch=0.90,
        patience=20,
        cache="ram",
        workers=8,
        cos_lr=True,
        deterministic=False,
        compile=True,  # torch.compile — 10-30% faster on A100 + PyTorch 2.x
        project=str(SCRIPT_DIR / "runs"),
        name="drone_detect",
        exist_ok=True,
        rect=True,
        save_period=10,
        degrees=15.0,
        flipud=0.5,
        scale=0.9,
        translate=0.2,
    )

    metrics = model.val(imgsz=960, half=True)
    print(f"Val  — mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")

    test_metrics = model.val(split="test", imgsz=960, half=True)
    print(f"Test — mAP50: {test_metrics.box.map50:.4f}  mAP50-95: {test_metrics.box.map:.4f}")
