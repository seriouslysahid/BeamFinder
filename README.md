# BeamFinder

Drone detection pipeline for line-of-sight (LoS) communication beam-steering.

## Overview

BeamFinder uses [YOLO26n](https://docs.ultralytics.com/models/yolo26/) to detect drones in images captured by a stationary ground camera. The detected bounding box coordinates `(x, y, width, height)` are output to a CSV file, which feeds into a line-of-sight communication system for automatic beam selection.

## Prerequisites

- Python 3.10 or higher

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd BeamFinder
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - **Windows:**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux/macOS:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your drone images in the `data/images/` directory (JPEG, PNG, BMP, or TIFF).

2. Run the detection pipeline:
   ```bash
   python -m src.detect
   ```

3. Results are saved to `output/detections.csv` with columns:
   | Column | Description |
   |--------|-------------|
   | `image_name` | Source image filename |
   | `x` | Bounding box X coordinate |
   | `y` | Bounding box Y coordinate |
   | `width` | Bounding box width |
   | `height` | Bounding box height |
   | `confidence` | Detection confidence score |
   | `class` | Detected object class |

## Project Structure

```
BeamFinder/
├── src/
│   ├── __init__.py          # Package marker
│   ├── config.py            # Configuration (paths, model, thresholds)
│   └── detect.py            # Detection pipeline (Phase 2)
├── data/
│   └── images/              # Input: drone images go here
├── output/                  # Output: CSV detection results
├── .gsd/                    # Project management docs
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

## Configuration

All settings are centralized in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `yolo26n.pt` | YOLO26 model variant |
| `CONFIDENCE_THRESHOLD` | `0.25` | Minimum detection confidence |
| `IMAGE_SIZE` | `640` | Inference image size (px) |
| `IMAGE_DIR` | `data/images/` | Input image directory |
| `OUTPUT_CSV` | `output/detections.csv` | Output CSV path |

## License

MIT
