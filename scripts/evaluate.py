import json
import pandas as pd
from pathlib import Path

def main():
    detections_path = Path("output/detections.csv")
    scenario_path = Path("data/raw/scenario23.csv")
    test_images_dir = Path("data/processed/images/test")
    output_path = Path("output/beam_metrics.json")
    
    if not detections_path.exists():
        print(f"Error: {detections_path} not found")
        return
    
    if not scenario_path.exists():
        print(f"Error: {scenario_path} not found")
        return
    
    # 1. Load detections
    detections = pd.read_csv(detections_path)
    print(f"Loaded {len(detections)} detections from {detections_path}")
    
    # 2. Load ground-truth beams
    scenario = pd.read_csv(scenario_path)
    scenario = scenario[["unit1_rgb", "unit1_beam_index"]].copy()
    scenario.columns = ["img", "beam_gt"]
    scenario = scenario.dropna(subset=["beam_gt"])
    scenario["beam_gt"] = scenario["beam_gt"].astype(int)
    
    # Extract stem from image filename
    scenario["stem"] = scenario["img"].apply(lambda x: Path(x).stem if pd.notna(x) else None)
    scenario = scenario.dropna(subset=["stem"])
    stem_to_beam_gt = dict(zip(scenario["stem"], scenario["beam_gt"]))
    
    # 3. Join detections with ground truth
    detections["stem"] = detections["image"].apply(lambda x: Path(x).stem)
    detections["beam_gt"] = detections["stem"].map(stem_to_beam_gt)
    
    # Filter out detections without ground truth
    detections_with_gt = detections.dropna(subset=["beam_gt"])
    detections_with_gt["beam_gt"] = detections_with_gt["beam_gt"].astype(int)
    
    print(f"Matched {len(detections_with_gt)} detections with ground-truth beams")
    
    # Count test images
    test_images = list(test_images_dir.glob("*.jpg"))
    n_test_images = len(test_images)
    n_detected_images = detections["image"].nunique()
    coverage = n_detected_images / n_test_images if n_test_images > 0 else 0
    
    print(f"Test set: {n_test_images} images, {n_detected_images} had detections ({coverage:.1%} coverage)")
    
    if len(detections_with_gt) == 0:
        print("No detections with ground truth to evaluate")
        return
    
    # 4. Compute top-k accuracies
    def in_topk(row, k):
        gt = row["beam_gt"]
        topk = [row[f"beam_top{i}"] for i in range(1, k+1) if pd.notna(row.get(f"beam_top{i}"))]
        return gt in topk
    
    top1_correct = detections_with_gt.apply(lambda row: in_topk(row, 1), axis=1).sum()
    top3_correct = detections_with_gt.apply(lambda row: in_topk(row, 3), axis=1).sum()
    top5_correct = detections_with_gt.apply(lambda row: in_topk(row, 5), axis=1).sum()
    
    n = len(detections_with_gt)
    top1_acc = top1_correct / n
    top3_acc = top3_correct / n
    top5_acc = top5_correct / n
    
    # 5. Compute distance-based accuracy (DBA)
    def beam_distance(row):
        gt = row["beam_gt"]
        pred = row["beam_top1"]
        if pd.isna(pred):
            return float('inf')
        return abs(int(pred) - gt)
    
    detections_with_gt["beam_dist"] = detections_with_gt.apply(beam_distance, axis=1)
    
    dba_within_1 = (detections_with_gt["beam_dist"] <= 1).sum() / n
    dba_within_2 = (detections_with_gt["beam_dist"] <= 2).sum() / n
    
    # 6. Print results table
    print("\n" + "="*60)
    print("BEAM PREDICTION ACCURACY")
    print("="*60)
    print(f"Top-1 Accuracy:        {top1_acc:.4f}  ({top1_correct}/{n})")
    print(f"Top-3 Accuracy:        {top3_acc:.4f}  ({top3_correct}/{n})")
    print(f"Top-5 Accuracy:        {top5_acc:.4f}  ({top5_correct}/{n})")
    print("-"*60)
    print(f"DBA (within ±1 beam):  {dba_within_1:.4f}")
    print(f"DBA (within ±2 beams): {dba_within_2:.4f}")
    print("="*60)
    
    # 7. Save metrics to JSON
    metrics = {
        "test_images": n_test_images,
        "detected_images": n_detected_images,
        "coverage": round(coverage, 4),
        "detections_evaluated": n,
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "dba_within_1": round(dba_within_1, 4),
        "dba_within_2": round(dba_within_2, 4),
    }
    
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {output_path}")

if __name__ == "__main__":
    main()
