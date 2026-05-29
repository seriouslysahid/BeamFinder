import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.neighbors import KNeighborsClassifier

def collect_points(img_dir, model, stem_to_beam, desc="Collecting Points"):
    """Run detector and collect (x_abs, y_abs, beam) points from images."""
    points = []
    labels = []
    
    images = list(Path(img_dir).glob("*.jpg"))
    for img in tqdm(images, desc=desc):
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
    
    return np.array(points), np.array(labels)

def evaluate_topk(knn, X_val, y_val):
    """Compute top-1, top-3, top-5 accuracy on validation set."""
    proba = knn.predict_proba(X_val)
    classes = knn.classes_
    
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    
    for i, true_beam in enumerate(y_val):
        # Get top-k predicted beams by probability
        top_indices = np.argsort(proba[i])[::-1]
        top_beams = classes[top_indices]
        
        if true_beam in top_beams[:1]:
            top1_correct += 1
        if true_beam in top_beams[:3]:
            top3_correct += 1
        if true_beam in top_beams[:5]:
            top5_correct += 1
    
    n = len(y_val)
    return top1_correct / n, top3_correct / n, top5_correct / n

def main():
    train_img_dir = "data/processed/images/train"
    val_img_dir = "data/processed/images/val"
    model_path = "best.pt"
    beam_lookup_path = "data/processed/beam_lookup.csv"
    out_dir = Path("output/beam_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load beam lookup
    beam_lookup = pd.read_csv(beam_lookup_path)
    stem_to_beam = dict(zip(beam_lookup["stem"], beam_lookup["beam"]))
    
    # 2. Load detector
    model = YOLO(model_path)
    
    # 3. Collect training points
    X_train, y_train = collect_points(train_img_dir, model, stem_to_beam, desc="Collecting Training Points")
    print(f"Collected {len(X_train)} training points across {len(np.unique(y_train))} distinct beams")
    
    # 4. Collect validation points
    X_val, y_val = collect_points(val_img_dir, model, stem_to_beam, desc="Collecting Validation Points")
    print(f"Collected {len(X_val)} validation points")
    
    # 5. Hyperparameter sweep: find best k on validation set
    k_candidates = [5, 7, 11, 15, 21]
    results = []
    
    print("\n" + "="*60)
    print("k-NN Hyperparameter Sweep (Validation Set)")
    print("="*60)
    print(f"{'k':<5} {'Top-1':<10} {'Top-3':<10} {'Top-5':<10}")
    print("-"*60)
    
    for k in k_candidates:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn.fit(X_train, y_train)
        
        top1, top3, top5 = evaluate_topk(knn, X_val, y_val)
        results.append((k, top1, top3, top5))
        
        print(f"{k:<5} {top1:<10.4f} {top3:<10.4f} {top5:<10.4f}")
    
    # 6. Select best k based on validation top-3 accuracy
    best_k = max(results, key=lambda x: x[2])[0]  # x[2] is top-3 accuracy
    best_top1, best_top3, best_top5 = [r for r in results if r[0] == best_k][0][1:]
    
    print("-"*60)
    print(f"Best k: {best_k} (Val Top-3: {best_top3:.4f})")
    print("="*60 + "\n")
    
    # 7. Train final model with best k on full training set
    final_knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
    final_knn.fit(X_train, y_train)
    
    # 8. Save classifier
    model_file = out_dir / "beam_knn.joblib"
    joblib.dump(final_knn, model_file)
    
    print(f"Trained final k-NN classifier with k={best_k}")
    print(f"Training set: {len(X_train)} points, {len(np.unique(y_train))} beams")
    print(f"Saved classifier to {model_file}")

if __name__ == "__main__":
    main()
