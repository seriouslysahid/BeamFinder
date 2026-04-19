# Spec: YOLO26s Drone Detection & Beam Clustering on DeepSense Scenario 23

## Goal

Fine-tune `yolo26s.pt` (COCO pretrained) on the **full DeepSense Scenario 23 dataset** to produce a single-class drone detector, and cascade the detection bounding boxes through a geometric clustering algorithm to predict optimal mmWave beam indices. 

**Output:** `best.pt` (Weight file) and `detections.csv` (Final unified inference map).

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

Reorganize dataset into the standard YOLO layout cleanly. 
We shuffle with `random.seed(42)` and carve a 70/30 train/test split. From the train pool, we carve 15% as a validation set for YOLO monitoring. To save space, images and labels are cleanly symlinked into `data/processed/`.

**Output Structure:**
```
data/processed/
├── images/ (train, val, test)
└── labels/ (train, val, test)
```

---

## Phase 2 — Data YAML (`configs/dataset.yaml`)

```yaml
path: ../data/processed
train: images/train
val: images/val
test: images/test

nc: 1
names: ["drone"]
```

---

## Phase 3 — Training (`scripts/train.py`)

Fine-tune YOLO26s from COCO pretrained weights on the prepared dataset.

### Hyperparameter Rationale
| Parameter | Value | Why |
|-----------|-------|-----|
| `model` | `yolo26s.pt` | COCO pretrained small variant |
| `epochs=1000` | Max budget | Early-stops using `patience=50` |
| `imgsz=960` | Native width | Images are 960×540; preserves resolution |
| `rect=True` | Preserve 16:9 | Avoids letterbox padding |
| `batch=0.90` | Auto-batch | Uses 90% of available VRAM |
| `cache="ram"` | Fast I/O | Fits in RAM, bypassing disk latency |
| `degrees=15` | Augmentation | Drones roll and pitch dynamically |

After training, the script natively copies the optimal weights to the repository root as `best.pt`.

---

## Phase 4 — Beam Centroid Mapping (`scripts/classify.py`)

Unlike the original DeepSense paper which used a heavy ResNet-50 block to solve both detection and classification simultaneously, we separate the logic algebraically.

1. This script sweeps the training images with our newly trained `best.pt`.
2. It groups the resulting drone `(x, y)` coordinates via their ground-truth `beam_index` defined in `scenario23.csv`.
3. It takes the mathematical centroid (average) per beam class, and saves the geometry map out to `output/beam_model/beam_centroids.csv`.

*(Note: This completely eliminates the need for a secondary Neural Network).*

---

## Phase 5 — Unified Inference (`scripts/detect.py`)

This handles end-to-end evaluation via geometric projection.
1. It loads `best.pt` and `beam_centroids.csv`.
2. Sweeps the `test/` partition images.
3. Obtains drone `(x_center, y_center)`.
4. Calculates the Euclidean distance from the drone to all pre-calculated centroids in the map, pulling the closest one as the beam prediction.
5. Saves a unified CSV with coordinates, YOLO confidences, and algorithmic beam grouping simultaneously.

---

## Outputs Map

| Path | Contents |
|------|----------|
| `data/processed/` | YOLO-format dataset splits |
| `best.pt` | Best detector checkpoint (root) |
| `output/beam_model/` | Centroid CSV mappings |
| `output/detections.csv` | Final multi-class inference |
| `output/annotated/` | Visual bounding boxes drawn over evaluation imagery |

---

## Differences from the DeepSense Paper

| Aspect | Paper | This Spec |
|--------|-------|-----------|
| Architecture | ResNet-50 (end-to-end) | YOLO26s (Detection) + Euclidean Centroids (Classification) |
| Optimizer | Adam | MuSGD (YOLO26 default) |
| Split | Random 70/30 | Random 70/30 (val carved from train for YOLO explicit monitoring) |

This disjoint architecture runs significantly faster and allows distinct tuning of drone tracking geometries devoid of beam-sweep noise.
