# BeamFinder

This project detects drones using the YOLO26s architecture and predicts optimal mmWave transmission beam indices using a k-Nearest-Neighbors classifier on detected bounding box centers. It is built to run on the DeepSense 6G Scenario 23 dataset.

## Repository Structure

The codebase is organized as follows:

```text
BeamFinder/
├── configs/            # Contains dataset.yaml for YOLO data pathing and spec.md
├── data/               
│   ├── raw/            # Place the unmodified scenario23 dataset here (including the CSV)
│   └── processed/      # Built automatically; holds YOLO-formatted train/val/test splits + beam_lookup.csv
├── notebooks/          # Colab-ready versions of the executable scripts
├── paper/              # Contains the reference paper PDF and extracted Markdown text
├── scripts/            # Contains the core execution pipeline
└── output/             # Built automatically; holds trained models, predictions, metrics, and annotated images
```

## Setup & Requirements

Before running the scripts, make sure you have installed the necessary Python packages:
```bash
pip install -r requirements.txt
```

## How to Run the Pipeline

The project is designed to be executed sequentially from the root directory. 

**1. Build the Dataset**
```bash
python scripts/build_dataset.py
```
This parses the raw DeepSense data, handles image-label alignment, and generates a 70/30 train/test split (with 15% validation carved from training). It also creates `beam_lookup.csv` for clean beam label lookup.

**2. Train the Model**
```bash
python scripts/train.py
```
This fine-tunes the YOLO26s model on the processed dataset with localization-focused hyperparameters (box gain 10.0, 100 epochs with early stopping). Augmentations like flip, mosaic, and rotation are disabled because they break the absolute-position-to-beam mapping that the beam classifier depends on. Once training is complete, the optimal model weights are automatically saved to the root directory as `best.pt`.

**3. Train the Beam Classifier**
```bash
python scripts/classify.py
```
This runs the trained detector over all training images, extracts the highest-confidence bounding box centers in absolute pixel coordinates, and fits a distance-weighted k-NN classifier (k=5) on the (x, y) → beam mapping. The trained classifier is saved to `output/beam_model/beam_knn.joblib`.

**4. Run Inference**
```bash
python scripts/detect.py
```
This script handles test set inference. It loads `best.pt` and `beam_knn.joblib`, runs drone detection on the test images, and uses k-NN probability ranking to predict the top-5 beam indices for each detection. Outputs are saved to `output/detections.csv` with columns for bounding box coordinates, confidence, and top-5 beam predictions. Annotated images with bounding boxes are saved to `output/annotated/`.

**5. Evaluate Beam Prediction Accuracy**
```bash
python scripts/evaluate.py
```
This script computes beam prediction accuracy metrics by comparing test predictions to ground-truth beam indices. It reports:
- **Top-1, Top-3, Top-5 accuracy**: fraction of detections where the true beam appears in the predicted top-k
- **Distance-based accuracy (DBA)**: fraction of detections where the top-1 prediction is within ±1 or ±2 beam indices of the true beam

Results are printed to the console and saved to `output/beam_metrics.json`.

## Outputs

When the pipeline finishes, check the `output/` folder. It will contain:
- `beam_model/beam_knn.joblib`: Trained k-NN beam classifier
- `detections.csv`: Test predictions with bounding boxes and top-5 beam rankings
- `beam_metrics.json`: Accuracy metrics (top-1/3/5, DBA)
- `annotated/`: Annotated test images with bounding boxes

## Key Design Decisions

**Why k-NN instead of centroids?** Distance-weighted k-NN (k=5) handles rare beams better than simple centroid lookup and provides probability distributions for top-k ranking.

**Why disable flip/mosaic/mixup augmentations?** The beam index is tied to the drone's absolute position in the camera frame. Augmentations that change the drone's position without correspondingly changing the beam label (flip, mosaic, mixup, copy-paste) corrupt the training signal for the downstream beam classifier.

**Why box gain 10.0?** The downstream beam classifier only uses bounding box centers, so we emphasize localization accuracy over general detection metrics.

**Why 100 epochs instead of 1000?** YOLO26s was pretrained for 70 epochs; 100 with early stopping (patience=20) is the documented fine-tuning range for this dataset size.

## Expected Results

(To be filled after the A100 run)

- **Top-1 Beam Accuracy**: TBD
- **Top-3 Beam Accuracy**: TBD
- **Top-5 Beam Accuracy**: TBD
- **DBA (within ±1 beam)**: TBD
- **DBA (within ±2 beams)**: TBD
