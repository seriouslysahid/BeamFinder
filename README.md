# BeamFinder — Drone Detection for THz Beam Steering

A detection pipeline that uses YOLO26s to locate drones in images and outputs bounding box coordinates to a CSV file. Built as part of a study on line-of-sight beam steering for THz communication.

## How It Works

1. **Training:** Fine-tunes YOLO26s on the DeepSense Scenario 23 drone dataset
2. **Detection:** Runs the fine-tuned model on test images and outputs bounding boxes to CSV

## Project Structure

```
BeamFinder/
├── detect.py          # Detection script (outputs bounding boxes)
├── train.py           # Training script (fine-tune on drone data)
├── data.yaml          # Dataset configuration for training
├── issues.md          # Known issues and notes
├── yolo26s.pt         # Pretrained YOLO26s model weights
├── requirements.txt   # Python dependencies
├── data/              # Dataset (not tracked in git)
│   ├── images/
│   │   ├── train/         # 7,970 images
│   │   ├── validation/    # 1,708 images
│   │   └── test/          # 1,709 images
│   └── labels/            # Matching YOLO-format .txt files
├── output/            # Detection results (CSV + annotated images)
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
| image | Source image filename |
| x_center | Bounding box center X |
| y_center | Bounding box center Y |
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
