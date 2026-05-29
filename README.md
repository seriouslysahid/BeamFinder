# BeamFinder

Camera-based beam prediction for 6G drone communication using YOLO26s detection and k-NN classification.

## What Problem Are We Solving?

In 6G wireless networks, base stations use narrow, directional radio beams (called mmWave beams) to communicate with devices. Think of it like a spotlight that has to follow a moving target. For a drone flying around, the base station needs to constantly point the right beam at it.

The naive approach is to try all possible beams until you find the one with the best signal. But with 56+ beams to check, that's slow and wastes time when the drone is moving fast.

The better approach: the base station already has a camera pointed at the drone. We can look at where the drone appears in the camera image and use that position to guess which beam to use. If the drone is in the top-left of the frame, use beam 12. If it's bottom-right, use beam 48. And so on.

This project builds that system using the DeepSense 6G Scenario 23 dataset, which has 11,387 images of drones with labels for both where the drone is (bounding box) and which beam worked best (beam index 2-60).

## How It Works (In Plain Terms)

The system has two stages:

1. **Find the drone**: A computer vision model (YOLO26s) looks at the camera image and draws a bounding box around the drone, giving us its pixel coordinates.

2. **Pick the beam**: A k-nearest-neighbors classifier looks at where the drone is in the image (its x, y position) and predicts which beam index to use based on what worked for drones in similar positions during training.

That's it. No trying all 56 beams. Just: see drone → get position → predict beam.

## The Approach in Detail

**Drone detection**: We fine-tune YOLO26s (a pretrained object detector) on the dataset to detect drones. The key is getting accurate bounding box *centers*, since the beam classifier only cares about position, not box size. We train with `box=10.0` (higher localization weight) and disable augmentations like horizontal flips, mosaics, and rotation. Why? Because the beam index is tied to the drone's *absolute position in the frame*. If you flip the image horizontally, the drone moves to the opposite side, but the beam label doesn't flip with it. That would corrupt the training signal for the beam stage.

**Beam prediction**: We use a k-nearest-neighbors classifier (k=5, distance-weighted) trained on (x, y) → beam mappings from the training set. For each test image, we extract the detected drone's center in absolute pixel coordinates (e.g., x=480, y=270 in a 960×540 image) and ask the k-NN model: "What beam indices were used for the 5 closest training examples?" It returns a ranked list of the top-5 most likely beams. Distance weighting means closer neighbors count more, which handles rare beams (some have <10 training samples) better than simple centroid lookup or uniform voting.

**Why k-NN?** It's simple, interpretable, and works well for this spatial mapping problem. An earlier version used centroid-per-beam (just average all drone positions for each beam), but k-NN gives better top-3 and top-5 accuracy because it captures local variations in the beam layout.

## Repository Structure

```
BeamFinder/
├── configs/
│   ├── dataset.yaml          # YOLO data config (paths, class names)
│   └── spec.md               # Detailed project specification
├── data/
│   ├── raw/                  # Place DeepSense Scenario 23 dataset here
│   │   ├── unit1/camera_data/           # 11,387 source images (960×540 JPG)
│   │   ├── resources/bbox_labels_final/ # Ground-truth YOLO labels
│   │   └── scenario23.csv               # Metadata with beam indices
│   └── processed/            # Auto-generated YOLO-format splits
│       ├── images/           # train/val/test image splits (symlinked)
│       ├── labels/           # train/val/test label splits (symlinked)
│       └── beam_lookup.csv   # Clean stem → beam mapping
├── scripts/
│   ├── build_dataset.py      # Step 1: Create YOLO splits + beam lookup
│   ├── train.py              # Step 2: Fine-tune YOLO26s detector
│   ├── classify.py           # Step 3: Train k-NN beam classifier
│   ├── detect.py             # Step 4: Run inference on test set
│   └── evaluate.py           # Step 5: Compute accuracy metrics
├── output/
│   ├── beam_model/
│   │   └── beam_knn.joblib   # Trained k-NN classifier
│   ├── detections.csv        # Test predictions with top-5 beams
│   ├── beam_metrics.json     # Accuracy results
│   └── annotated/            # Test images with bounding boxes drawn
├── notebooks/                # Colab-ready versions of scripts
├── paper/                    # Reference paper (Charan et al.)
├── presentation/             # Demo slides
├── best.pt                   # Trained YOLO model weights (root)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

### What Each File Does

| File | Purpose |
|------|---------|
| `scripts/build_dataset.py` | Splits raw data into train/val/test (70/15/15), creates symlinks to save space, generates `beam_lookup.csv` |
| `scripts/train.py` | Fine-tunes YOLO26s for 100 epochs with localization-focused settings, saves `best.pt` |
| `scripts/classify.py` | Runs detector on training images, extracts drone centers, fits k-NN on (x,y)→beam, saves `beam_knn.joblib` |
| `scripts/detect.py` | Runs detector + k-NN on test set, outputs `detections.csv` with top-5 beam predictions per detection |
| `scripts/evaluate.py` | Compares predictions to ground truth, computes top-1/3/5 accuracy and distance-based accuracy, saves `beam_metrics.json` |
| `configs/dataset.yaml` | Tells YOLO where to find train/val/test images and that we have 1 class ("drone") |
| `configs/spec.md` | Full technical specification with hyperparameter rationale and design decisions |
| `best.pt` | Trained YOLO26s weights (copied from `runs/` after training) |
| `output/beam_model/beam_knn.joblib` | Trained k-NN classifier (scikit-learn model) |
| `output/detections.csv` | Test predictions: image name, bbox coords, confidence, top-5 beam indices |
| `output/beam_metrics.json` | Final accuracy numbers (top-1/3/5, distance-based accuracy) |

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   (Assumes CUDA and PyTorch are already installed. On Lightning.ai A100, they come pre-installed.)

2. **Download the dataset**: Get the DeepSense 6G Scenario 23 dataset and place it in `data/raw/` so you have:
   - `data/raw/unit1/camera_data/*.jpg` (11,387 images)
   - `data/raw/resources/bbox_labels_final/*.txt` (11,387 YOLO labels)
   - `data/raw/scenario23.csv` (metadata with beam indices)

## How to Run It

Run these commands in order from the repository root:

```bash
# 1. Build dataset splits (creates data/processed/)
python scripts/build_dataset.py

# 2. Train YOLO detector (creates best.pt, takes ~2-4 hours on A100)
python scripts/train.py

# 3. Train beam classifier (creates output/beam_model/beam_knn.joblib)
python scripts/classify.py

# 4. Run inference on test set (creates output/detections.csv)
python scripts/detect.py

# 5. Evaluate accuracy (creates output/beam_metrics.json)
python scripts/evaluate.py
```

**What each step produces**:
- Step 1: `data/processed/` with train/val/test splits + `beam_lookup.csv`
- Step 2: `best.pt` (trained detector weights)
- Step 3: `output/beam_model/beam_knn.joblib` (trained beam classifier)
- Step 4: `output/detections.csv` (test predictions with top-5 beams)
- Step 5: `output/beam_metrics.json` (accuracy metrics)

## Results

| Metric | Value |
|--------|-------|
| **Top-1 Accuracy** | (pending A100 run) |
| **Top-3 Accuracy** | (pending A100 run) |
| **Top-5 Accuracy** | (pending A100 run) |
| **DBA (within ±1 beam)** | (pending A100 run) |
| **DBA (within ±2 beams)** | (pending A100 run) |
| **Test Coverage** | (pending A100 run) |

*Top-k accuracy: fraction of detections where the true beam appears in the predicted top-k.*  
*DBA (distance-based accuracy): fraction where the top-1 prediction is within ±k beam indices of the true beam (soft metric for adjacent beam overlap).*

Results will be saved to `output/beam_metrics.json` after running the full pipeline.

## Dataset & Reference

**Dataset**: DeepSense 6G Scenario 23 — 11,387 images (960×540) of drones with ground-truth bounding boxes and beam indices (56 unique beams, range 2-60).

**Reference paper**: Charan et al., "Towards Real-World 6G Drone Communication: Position and Camera Aided Beam Prediction" (arXiv:2205.12187). Their approach used ResNet-50 for end-to-end detection and classification. We use YOLO26s + k-NN for a faster, more interpretable two-stage pipeline.

## Limitations / Future Work

- **Single drone only**: The dataset and beam labels assume one drone per image. Multi-drone scenarios would need a different approach.
- **Fixed camera**: The base station camera is stationary and level. If the camera moves or tilts, the position-to-beam mapping breaks.
- **No temporal modeling**: We treat each frame independently. A tracker or LSTM could smooth predictions across time.
- **Beam layout is dataset-specific**: The k-NN model learns the beam geometry from this particular base station setup. A different antenna array would need retraining.
- **Detection failures**: If YOLO misses the drone (low confidence or occlusion), we can't predict a beam. Current test coverage is not yet measured (pending A100 run).

Possible improvements: add a fallback beam for missed detections, use optical flow to predict drone motion, try a lightweight neural network instead of k-NN for the beam stage.
