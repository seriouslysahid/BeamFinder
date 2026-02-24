import csv
from pathlib import Path
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────
MODEL = "runs/drone_detect/weights/best.pt"
IMAGE_DIR = Path("data/images/test")
OUTPUT_DIR = Path("output")
CONF = 0.4
IMGSZ = 960
USE_SAHI = False  # sliced inference for small drones
# ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = YOLO(MODEL)
    csv_path = OUTPUT_DIR / "detections.csv"
    annotated_dir = OUTPUT_DIR / "annotated"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "x_center", "y_center", "width", "height", "confidence", "class"])

        if USE_SAHI:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction

            sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics", model_path=MODEL,
                confidence_threshold=CONF, image_size=IMGSZ,
            )
            for img in sorted(IMAGE_DIR.glob("*.jpg")):
                result = get_sliced_prediction(
                    str(img), sahi_model,
                    slice_height=512, slice_width=512,
                    overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                    verbose=0,
                )
                result.export_visuals(export_dir=str(annotated_dir), file_name=img.stem)
                for p in result.object_prediction_list:
                    b = p.bbox
                    cx, cy = (b.minx + b.maxx) / 2, (b.miny + b.maxy) / 2
                    w, h = b.maxx - b.minx, b.maxy - b.miny
                    writer.writerow([img.name, round(cx, 2), round(cy, 2),
                                     round(w, 2), round(h, 2),
                                     round(p.score.value, 4), p.category.name])
                    total += 1
        else:
            results = model.predict(
                source=str(IMAGE_DIR), conf=CONF, imgsz=IMGSZ,
                save=True, project=str(OUTPUT_DIR), name="annotated",
                exist_ok=True,
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
