from ultralytics import YOLO
from pathlib import Path
import csv
import sys

model_path = "yolo26n.pt"
image_dir = Path("data/images")
output_csv = Path("output/detections.csv")
output_images = Path("output/annotated")
confidence = 0.25
img_size = 640

if not image_dir.exists():
    print(f"Error: Image directory '{image_dir}' not found.")
    sys.exit(1)

output_csv.parent.mkdir(parents=True, exist_ok=True)
output_images.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("  BeamFinder - Drone Detection Pipeline")
print("=" * 50)
print(f"  Model:      {model_path}")
print(f"  Confidence: {confidence}")
print(f"  Image Size: {img_size}px")
print(f"  Input:      {image_dir}")
print(f"  Output:     {output_csv}")
print(f"  Annotated:  {output_images}")
print("=" * 50)
print()

model = YOLO(model_path)
results = model(str(image_dir), conf=confidence, imgsz=img_size, stream=True, verbose=False)

total_detections = 0
image_count = 0

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "x", "y", "width", "height", "confidence", "class"])

    for result in results:
        image_count += 1
        image_name = Path(result.path).name
        boxes = result.boxes

        # save annotated image with bounding boxes drawn
        result.save(filename=str(output_images / image_name))

        if boxes is None or len(boxes) == 0:
            writer.writerow([image_name, "", "", "", "", "", ""])
            print(f"  [{image_count}] {image_name} - 0 detections")
            continue

        count = len(boxes)
        total_detections += count

        for i in range(count):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            w = x2 - x1
            h = y2 - y1
            conf = round(boxes.conf[i].item(), 4)
            cls_id = int(boxes.cls[i].item())
            cls_name = result.names[cls_id]
            writer.writerow([image_name, round(x1, 2), round(y1, 2), round(w, 2), round(h, 2), conf, cls_name])

        print(f"  [{image_count}] {image_name} - {count} detection(s)")

print()
print("=" * 50)
print(f"  Done! Processed {image_count} images")
print(f"  Total detections: {total_detections}")
print(f"  Results saved to: {output_csv}")
print(f"  Annotated images: {output_images}")
print("=" * 50)
