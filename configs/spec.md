# Spec: YOLO26s Drone Detection & k-NN Beam Prediction on DeepSense Scenario 23

## Goal

Fine-tune `yolo26s.pt` (COCO pretrained) on the **full DeepSense Scenario 23 dataset** to produce a single-class drone detector, then use a k-Nearest-Neighbors classifier on detected bounding box centers to predict optimal mmWave beam indices. 

**Output:** `best.pt` (trained weights), `beam_knn.joblib` (beam classifier), `detections.csv` (test predictions), and `beam_metrics.json` (accuracy report).

**Reference paper:** Charan et al., "Towards Real-World 6G Drone Communication: Position and Camera Aided Beam Prediction" (arXiv:2205.12187)

---

## Dataset Source

All data comes from `data/raw/`, the original DeepSense 6G Scenario 23 download.

| Resource | Path | Count |
|----------|------|-------|
| Images | `data/raw/unit1/camera_data/*.jpg` | 11,387 |
| Labels | `data/raw/resources/bbox_labels_final/*.txt` | 11,387 |
| Metadata | `data/raw/scenario23.csv` | 11,387 rows |

Labels are **ground-truth** YOLO-format bounding box annotations shipped with the DeepSense download. Image and label filenames match 1:1 by stem.

---

## Phase 1 — Dataset Preparation (`scripts/build_dataset.py`)

Reorganize dataset into the standard YOLO layout. We shuffle with `random.seed(42)` and carve a 70/30 train/test split. From the train pool, we carve 15% as a validation set for YOLO monitoring. To save space, images and labels are symlinked into `data/processed/` (falls back to copy on Windows).

Additionally, this script creates `data/processed/beam_lookup.csv` by joining each image stem to its `unit1_beam_index` from `scenario23.csv`, giving the beam classifier a clean lookup without regex parsing.

**Output Structure:**
```
data/processed/
├── images/ (train, val, test)
├── labels/ (train, val, test)
└── beam_lookup.csv
```

---

## Phase 2 — Data YAML (`configs/dataset.yaml`)

```yaml
path: data/processed
train: images/train
val: images/val
test: images/test

nc: 1
names: ["drone"]
```

---

## Phase 3 — Training (`scripts/train.py`)

Fine-tune YOLO26s from COCO pretrained weights on the prepared dataset. The training objective is **accurate bounding box centers**, since the downstream beam classifier only uses the center coordinates.

### Hyperparameter Rationale
| Parameter | Value | Why |
|-----------|-------|-----|
| `model` | `yolo26s.pt` | COCO pretrained small variant |
| `epochs=100` | Fine-tune budget | YOLO26s was pretrained for 70 epochs; 100 with early stopping is the documented fine-tuning range for this dataset size (not arbitrary) |
| `patience=20` | Early stopping | Prevents overfitting |
| `imgsz=960` | Native width | Images are 960×540; preserves resolution |
| `rect=True` | Preserve 16:9 | Avoids letterbox padding |
| `batch=0.90` | Auto-batch | Uses 90% of available VRAM |
| `box=10.0` | Localization gain | Emphasizes bounding box center accuracy (IoU gain); S recipe default is ~9.83, nudged up for better center prediction |
| `cache="ram"` | Fast I/O | Fits in RAM, bypassing disk latency |

### Augmentation Settings (Critical)
| Parameter | Value | Why |
|-----------|-------|-----|
| `degrees=0.0` | No rotation | Base station camera is **fixed and level**; rotation would misalign position-to-beam mapping |
| `shear=0.0` | No shear | Same reason as rotation |
| `fliplr=0.0` | **No horizontal flip** | **CRITICAL:** Flipping moves the drone to a mirrored x-position, but the beam index does NOT flip. This would break the absolute-position-to-beam mapping that the beam stage depends on. |
| `mosaic=0.0` | No mosaic | Fabricates multi-drone tiled scenes; beam depends on the single real drone's absolute position |
| `mixup=0.0` | No mixup | Same reason as mosaic |
| `copy_paste=0.0` | No copy-paste | Same reason as mosaic |
| `scale=0.5` | Mild scale jitter | Drone apparent size varies with distance |
| `translate=0.1` | Mild position jitter | Drone position varies slightly |
| `hsv_h=0.015` | Color jitter | Outdoor lighting/weather varies |
| `hsv_s=0.4` | Saturation jitter | Outdoor lighting/weather varies |
| `hsv_v=0.4` | Value jitter | Outdoor lighting/weather varies |

**Core insight:** The beam index is tied to the drone's **absolute position in the camera frame**. Any augmentation that changes the drone's position without correspondingly changing the beam label (flip, mosaic, mixup, copy-paste) will corrupt the training signal for the downstream beam classifier.

After training, the script copies the best weights to the repository root as `best.pt`.

---

## Phase 4 — Beam Classifier Training (`scripts/classify.py`)

Unlike the original DeepSense paper which used a heavy ResNet-50 block to solve both detection and classification simultaneously, we separate the logic with a lightweight k-NN classifier.

1. This script runs the trained `best.pt` over all training images.
2. For each image, it extracts the highest-confidence bounding box center in **absolute pixel coordinates** (x_abs = x_norm × 960, y_abs = y_norm × 540).
3. It loads the ground-truth beam index from `beam_lookup.csv`.
4. It fits a k-Nearest-Neighbors classifier (k=5, distance-weighted) on the (x, y) → beam mapping.
5. It saves the trained classifier to `output/beam_model/beam_knn.joblib`.

**Why k-NN?** Distance-weighted k=5 handles rare beams (some have <5 samples) better than uniform voting or simple centroid lookup. It also provides probability distributions for top-k beam ranking.

---

## Phase 5 — Inference (`scripts/detect.py`)

This handles end-to-end evaluation on the test set.

1. Loads `best.pt` and `beam_knn.joblib`.
2. Runs detection on `data/processed/images/test` with conf=0.4.
3. For each detected drone, extracts the bounding box center in absolute pixel coordinates.
4. Uses `knn.predict_proba()` to rank all beam classes by probability, returning the top-5 beam indices.
5. Saves `output/detections.csv` with columns: `image, x_center, y_center, width, height, confidence, class, beam_top1, beam_top2, beam_top3, beam_top4, beam_top5`.
6. Saves annotated images with bounding boxes to `output/annotated/`.

---

## Phase 6 — Evaluation (`scripts/evaluate.py`)

This script computes beam prediction accuracy metrics (the main thing we report).

1. Loads `output/detections.csv` (test predictions).
2. Loads ground-truth beams by joining each image stem to `unit1_beam_index` from `scenario23.csv`.
3. Computes **top-k accuracy**: fraction of detections where the ground-truth beam appears in the predicted top-1, top-3, or top-5. (Standard DeepSense metric.)
4. Computes **distance-based accuracy (DBA)**: fraction of detections where the top-1 predicted beam is within ±1 or ±2 of the true beam. (Literature's "soft" metric for adjacent beam overlap.)
5. Reports test set coverage (how many test images had detections).
6. Prints a results table and saves all metrics to `output/beam_metrics.json`.

---

## Outputs Map

| Path | Contents |
|------|----------|
| `data/processed/` | YOLO-format dataset splits + beam_lookup.csv |
| `best.pt` | Best detector checkpoint (root) |
| `output/beam_model/beam_knn.joblib` | Trained k-NN beam classifier |
| `output/detections.csv` | Test predictions with top-5 beam rankings |
| `output/beam_metrics.json` | Accuracy metrics (top-1/3/5, DBA) |
| `output/annotated/` | Visual bounding boxes drawn over test images |

---

## Differences from the DeepSense Paper

| Aspect | Paper | This Implementation |
|--------|-------|---------------------|
| Architecture | ResNet-50 (end-to-end) | YOLO26s (Detection) + k-NN (Beam Classification) |
| Beam Classifier | Neural network | Distance-weighted k-NN (k=5) on absolute pixel centers |
| Augmentation | Standard | Disabled flip/mosaic/mixup to preserve position-to-beam mapping |
| Training Focus | General detection | Localization-focused (box gain 10.0) for accurate centers |
| Optimizer | Adam | Auto (YOLO26 default) |
| Split | Random 70/30 | Random 70/30 (val carved from train for YOLO monitoring) |

This disjoint architecture is faster, more interpretable, and allows independent tuning of detection and beam prediction.
