import csv
from pathlib import Path

import torch
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL = str(SCRIPT_DIR / "runs" / "drone_detect" / "weights" / "best.pt")
IMAGE_DIR = SCRIPT_DIR / "data" / "images" / "test"
OUTPUT_DIR = SCRIPT_DIR / "output"
CONF = 0.4
IMGSZ = 960
# ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # A100: maximize GPU throughput
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model = YOLO(MODEL)
    csv_path = OUTPUT_DIR / "detections.csv"
    annotated_dir = OUTPUT_DIR / "annotated"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "x_center", "y_center", "width", "height", "confidence", "class"])

        results = model.predict(
            source=str(IMAGE_DIR), conf=CONF, imgsz=IMGSZ,
            save=True, project=str(OUTPUT_DIR), name="annotated",
            exist_ok=True, half=True, batch=16,
        )
        for r in results:
            name = Path(r.path).name
            if r.boxes is not None and len(r.boxes):
                for box in r.boxes:
                    cx, cy, w, h = box.xywh[0].tolist()
                    writer.writerow([name, round(cx, 2), round(cy, 2),
                                     round(w, 2), round(h, 2),
                                     round(box.conf.item(), 4),
                                     r.names[int(box.cls.item())]])
                    total += 1

    print(f"{total} detections saved to {csv_path}")
