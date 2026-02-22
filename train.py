import argparse
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train drone detector")
    parser.add_argument("--model", default="yolo26s.pt",
                        help="pretrained weights to start from")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="0", help="GPU id or 'cpu'")
    parser.add_argument("--batch", type=int, default=-1,
                        help="-1 lets ultralytics pick based on available VRAM")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="input resolution (960 since our images are 960x540)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=20,
        cache="ram",
        workers=0,
        amp=True,
        cos_lr=True,
        close_mosaic=10,
        freeze=10,
        save=True,
        save_period=10,
        project="runs",
        name="drone_detect",
        exist_ok=True,
        rect=True,
        plots=True,
        degrees=15.0,
        flipud=0.5,
        scale=0.9,
        translate=0.2,
        copy_paste=0.1,
    )

    metrics = model.val(augment=True)
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
