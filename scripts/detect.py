import csv
import numpy as np
import joblib
from pathlib import Path
from ultralytics import YOLO

def get_top5_beams(x_abs, y_abs, knn):
    """Predict top-5 beam indices using k-NN classifier probabilities."""
    point = np.array([[x_abs, y_abs]])
    proba = knn.predict_proba(point)[0]
    
    # Sort classes by descending probability
    sorted_indices = np.argsort(proba)[::-1]
    top5_classes = knn.classes_[sorted_indices[:5]]
    
    return top5_classes.tolist()

def main():
    if not Path("best.pt").exists():
        print("Model best.pt not found")
        return
    
    knn_file = Path("output/beam_model/beam_knn.joblib")
    if not knn_file.exists():
        print(f"k-NN classifier {knn_file} not found")
        return
        
    model = YOLO("best.pt")
    knn = joblib.load(knn_file)
    print(f"Loaded k-NN classifier from {knn_file}")
    
    out_dir = Path("output")
    csv_out = out_dir / "detections.csv"
    annotated = out_dir / "annotated"
    
    out_dir.mkdir(exist_ok=True)
    annotated.mkdir(exist_ok=True)
    
    total = 0
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "x_center", "y_center", "width", "height", "confidence", "class", "beam_top1", "beam_top2", "beam_top3", "beam_top4", "beam_top5"])

        results = model.predict(source="data/processed/images/test", conf=0.4, imgsz=960, save=True, project="output", name="annotated", exist_ok=True, half=True, batch=32)
        
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
                
            name = Path(r.path).name
            for box in r.boxes:
                # Get absolute pixel coordinates
                cx, cy, w, h = box.xywh[0].tolist()
                top5 = get_top5_beams(cx, cy, knn)
                
                # Pad with None if fewer than 5 classes exist
                while len(top5) < 5:
                    top5.append(None)
                
                writer.writerow([name, round(cx, 2), round(cy, 2), round(w, 2), round(h, 2), round(box.conf.item(), 4), r.names[int(box.cls.item())]] + top5[:5])
                total += 1

    print(f"Saved {total} detections to {csv_out}")

if __name__ == "__main__":
    main()
