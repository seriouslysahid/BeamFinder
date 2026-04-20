import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

def get_seq(filename):
    match = re.match(r"image_BS1_(\d+)_\d+_\d+_\d+", Path(filename).stem)
    if match: return int(match.group(1))
    return None

def main():
    csv_path = "data/raw/scenario23.csv"
    img_dir = "data/processed/images/train"
    model_path = "best.pt"
    out_dir = Path("output/beam_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. build lookup
    df = pd.read_csv(csv_path)
    df = df[["index", "unit1_rgb", "unit1_beam_index"]].copy()
    df.columns = ["seq", "img", "beam"]
    df = df.dropna(subset=["beam"])
    df["beam"] = df["beam"].astype(int)
    
    seq_beam_map = dict(zip(df["seq"], df["beam"]))

    # 2. get coords
    model = YOLO(model_path)
    beam_coords = {}
    
    images = list(Path(img_dir).glob("*.jpg"))
    for img in tqdm(images, desc="Mapping Drone Geometry"):
        seq = get_seq(img.name)
        if seq is None or seq not in seq_beam_map:
            continue
            
        beam = seq_beam_map[seq]
        res = model(str(img), conf=0.25, verbose=False)
        boxes = res[0].boxes
        
        if boxes is None or len(boxes) == 0:
            continue
            
        best = int(np.argmax(boxes.conf.cpu().numpy()))
        x, y = float(boxes.xywhn[best][0]), float(boxes.xywhn[best][1])
        
        if beam not in beam_coords:
            beam_coords[beam] = []
        beam_coords[beam].append((x, y))

    # 3. compute centroids
    centroids = {}
    rows = []
    
    for beam, coords in sorted(beam_coords.items()):
        cx, cy = np.array(coords).mean(axis=0)
        centroids[beam] = (float(cx), float(cy))
        rows.append({"beam_index": beam, "num_samples": len(coords), "centroid_x": round(cx, 6), "centroid_y": round(cy, 6)})
        print(f"Beam {beam}: {len(coords)} samples, centroid = ({cx:.4f}, {cy:.4f})")

    # save artifacts
    pd.DataFrame(rows).to_csv(out_dir / "beam_centroids.csv", index=False)
    
    with open(out_dir / "beam_centroids.pkl", "wb") as f:
        pickle.dump(centroids, f)
        
    with open(out_dir / "beam_centroids.json", "w") as f:
        json.dump(centroids, f, indent=2)

    print(f"Centroids generated and saved to {out_dir}")

if __name__ == "__main__":
    main()