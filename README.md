# BeamFinder — Drone Detection for THz Beam Steering

A detection pipeline that uses YOLOv26n to locate drones in images and outputs bounding box coordinates to a CSV file. Built as part of a study on line-of-sight beam steering for THz communication.

## How It Works

1. **Detection:** Runs YOLO26n on images to detect objects and output bounding boxes
2. **Training:** Fine-tunes YOLO26n on a drone dataset for better accuracy (requires annotations)

> **Note:** The pretrained model uses COCO weights. "Drone" is not a COCO class, so detections will be generic objects until the model is fine-tuned on drone-specific data.

## Project Structure

```
BeamFinder/
├── detect.py          # Detection script (outputs bounding boxes)
├── train.py           # Training script (fine-tune on drone data)
├── data.yaml          # Dataset configuration for training
├── issues.md          # Known issues for professor review
├── yolo26n.pt         # Pretrained YOLO26n model weights
├── requirements.txt   # Python dependencies
├── data/images/       # Dataset (not tracked in git)
│   ├── train/         # 7,970 training images
│   └── validation/    # 3,416 validation images
├── output/            # Detection results (CSV)
└── runs/              # Training results (auto-generated)
```

## Setup

```bash
pip install -r requirements.txt
```

## Detection

```bash
python detect.py
```

Outputs bounding box coordinates to `output/detections.csv`.

## Training

Requires bounding box annotation files (`.txt` per image in YOLO format). See [issues.md](issues.md) for details.

```bash
python train.py
```

Training results are saved to `runs/drone_detect/`.

## Output Format (detections.csv)

| Column | Description |
|--------|-------------|
| image_name | Source image filename |
| x | Top-left X coordinate |
| y | Top-left Y coordinate |
| width | Bounding box width |
| height | Bounding box height |
| confidence | Detection confidence (0-1) |
| class | Detected object class |

## Requirements

- Python 3.10+
- ultralytics

## References

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [YOLO Predict Mode](https://docs.ultralytics.com/modes/predict/)
- [YOLO Train Mode](https://docs.ultralytics.com/modes/train/)
