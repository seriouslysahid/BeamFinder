# Technology Stack

## Core Dependencies

- **Python 3.x**: Primary language
- **Ultralytics (≥8.4.0)**: YOLO26s model training and inference
- **PyTorch**: Deep learning framework (pre-installed on Lightning.ai A100)
- **CUDA**: GPU acceleration (pre-installed on Lightning.ai A100)
- **Matplotlib (≥3.7.0)**: Visualization and plotting

## Model Architecture

- **YOLO26s**: Small variant of YOLO26, COCO pretrained
- **Single-class detection**: Trained to detect "drone" class only
- **Geometric clustering**: Euclidean distance-based beam classification (no neural network)

## Common Commands

### Setup
```bash
pip install -r requirements.txt
```

### Pipeline Execution (Sequential)

**1. Build Dataset**
```bash
python scripts/build_dataset.py
```
Parses raw DeepSense data, creates YOLO-format splits (70/30 train/test with 15% validation)

**2. Train Model**
```bash
python scripts/train.py
```
Fine-tunes YOLO26s, saves `best.pt` to root directory

**3. Generate Beam Centroids**
```bash
python scripts/classify.py
```
Calculates geometric centroid clusters for each beam index

**4. Run Inference**
```bash
python scripts/detect.py
```
Performs detection and beam prediction on test set, outputs `detections.csv`

## Training Configuration

Key hyperparameters in `train.py`:
- `epochs=1000` with `patience=50` early stopping
- `imgsz=960` (native image width)
- `rect=True` (preserves 16:9 aspect ratio)
- `batch=0.90` (auto-batch using 90% VRAM)
- `cache="ram"` (fast I/O)
- `degrees=15` (rotation augmentation)
- `cos_lr=True` (cosine learning rate)
- `compile=True` (PyTorch 2.0 optimization)

## Development Environment

- Designed for Lightning.ai A100 GPU instances
- CUDA and PyTorch come pre-installed
- Uses symlinks for space efficiency (falls back to copy on Windows)
