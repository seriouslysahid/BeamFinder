import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.neighbors import KNeighborsClassifier

def main():
    img_dir = "data/processed/images/train"
    model_path = "best.pt"
    beam_lookup_path = "data/processed/beam_lookup.csv"
    out_dir = Path("output/beam_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load beam lookup
    beam_lookup = pd.read_csv(beam_lookup_path)
    stem_to_beam = dict(zip(beam_lookup["stem"], beam_lookup["beam"]))
    
    # 2. Run detector and collect absolute pixel coordinates
    model = YOLO(model_path)
    points = []
    labels = []
    
    images = list(Path(img_dir).glob("*.jpg"))
    for img in tqdm(images, desc="Collecting Training Points"):
        stem = img.stem
        if stem not in stem_to_beam:
            continue
            
        beam = stem_to_beam[stem]
        res = model(str(img), conf=0.25, verbose=False)
        boxes = res[0].boxes
        
        if boxes is None or len(boxes) == 0:
            continue
            
        # Get highest-confidence box center in absolute pixel coordinates
        best = int(np.argmax(boxes.conf.cpu().numpy()))
        x_norm, y_norm = float(boxes.xywhn[best][0]), float(boxes.xywhn[best][1])
        x_abs = x_norm * 960
        y_abs = y_norm * 540
        
        points.append([x_abs, y_abs])
        labels.append(beam)
    
    # 3. Fit k-NN classifier
    X = np.array(points)
    y = np.array(labels)
    
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(X, y)
    
    # Save classifier
    model_file = out_dir / "beam_knn.joblib"
    joblib.dump(knn, model_file)
    
    n_beams = len(np.unique(y))
    print(f"Trained k-NN classifier on {len(X)} points across {n_beams} distinct beams")
    print(f"Saved classifier to {model_file}")

if __name__ == "__main__":
    main()