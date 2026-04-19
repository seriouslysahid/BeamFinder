# BeamFinder

This project detects drones using the YOLO26s architecture and mathematically groups them into optimal transmission beam clusters. It is built to run on the DeepSense 6G Scenario 23 dataset.

## Repository Structure

The codebase is organized as follows:

```text
BeamFinder/
├── configs/            # Contains dataset.yaml for YOLO data pathing
├── data/               
│   ├── raw/            # Place the unmodified scenario23 dataset here (including the CSV)
│   └── processed/      # Built automatically; holds YOLO-formatted train/val/test splits
├── paper/              # Contains the reference paper PDF and extracted Markdown text
├── scripts/            # Contains the core execution pipeline
└── output/             # Built automatically; holds generated model weights, metrics, and logs
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
This parses the raw DeepSense data, handles image-label alignment, and generates a 70/30 train/test split.

**2. Train the Model**
```bash
python scripts/train.py
```
This fine-tunes the YOLO26s model on the processed dataset. Once training is complete, the optimal model weights are automatically saved to the root directory as `best.pt`.

**3. Generate the Beam Centroids**
```bash
python scripts/classify.py
```
This sweeps the dataset using the trained model to calculate geometric centroid clusters for each beam index. The map is saved out as a `.csv` and `.pkl` object.

**4. Run Unified Inference**
```bash
python scripts/detect.py
```
This script handles the final evaluation. It takes the test images, runs drone detection using `best.pt`, computes the Euclidean distance to the nearest beam centroid, and outputs the final classifications to the `output/` folder.

## Outputs
When the pipeline finishes, check the `output/` folder. It will contain annotated images showing the drone bounding boxes, along with a unified `detections.csv` containing coordinates, confidence scores, and predicted beam indices.
