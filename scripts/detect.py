import csv
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO

def get_closest_beam(x, y, centroids):
    if not centroids:
        return None, None
    query = np.array([x, y])
    best_beam = None
    min_dist = float('inf')
    
    for b_idx, (cx, cy) in centroids.items():
        dist = float(np.linalg.norm(query - np.array([cx, cy])))
        if dist < min_dist:
            min_dist = dist
            best_beam = b_idx
            
    return best_beam, round(min_dist, 6)

def main():
    if not Path("best.pt").exists():
        print("Model best.pt not found")
        return
        
    model = YOLO("best.pt")
    out_dir = Path("output")
    csv_out = out_dir / "detections.csv"
    annotated = out_dir / "annotated"
    
    out_dir.mkdir(exist_ok=True)
    annotated.mkdir(exist_ok=True)

    # load centroids mapping
    centroids = {}
    centroid_file = Path("output/beam_model/beam_centroids.csv")
    if centroid_file.exists():
        with open(centroid_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                centroids[int(row["beam_index"])] = (float(row["centroid_x"]), float(row["centroid_y"]))
        print(f"Loaded {len(centroids)} beam centroids")
    
    total = 0
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "x_center", "y_center", "width", "height", "confidence", "class", "beam_index", "beam_distance"])

        results = model.predict(source="data/processed/images/test", conf=0.4, imgsz=960, save=True, project="output", name="annotated", exist_ok=True, half=True, batch=32)
        
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
                
            name = Path(r.path).name
            for box in r.boxes:
                cx, cy, w, h = box.xywh[0].tolist()
                b_idx, b_dist = get_closest_beam(cx, cy, centroids)
                
                writer.writerow([name, round(cx, 2), round(cy, 2), round(w, 2), round(h, 2), round(box.conf.item(), 4), r.names[int(box.cls.item())], b_idx, b_dist])
                total += 1

    print(f"Saved {total} detections to {csv_out}")

if __name__ == "__main__":
    main()
