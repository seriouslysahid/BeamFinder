---
phase: 2
plan: 1
wave: 1
---

# Plan 2.1: Detector & CSV Writer Modules

## Objective
Create the two core modules: a detector that loads YOLO26n and runs inference on images, and a CSV writer that saves bounding box results. These are the building blocks for the main detection script.

## Context
- .gsd/SPEC.md
- src/config.py

## Tasks

<task type="auto">
  <name>Create detector module</name>
  <files>src/detector.py</files>
  <action>
    Create src/detector.py with:

    ```python
    class DroneDetector:
        def __init__(self, model_name, confidence, image_size):
            # Load YOLO26n model using ultralytics.YOLO
            # Store confidence threshold and image size

        def detect(self, image_path: Path) -> list[dict]:
            # Run model inference on a single image
            # Return list of detections, each as:
            #   {"x": float, "y": float, "width": float, "height": float,
            #    "confidence": float, "class": str}
    ```

    Implementation details:
    - Use `from ultralytics import YOLO` to load model
    - Call `model(image_path, conf=confidence, imgsz=image_size)` for inference
    - YOLO26 returns Results objects with `.boxes` attribute
    - Extract bounding boxes via `results[0].boxes.xywh` (center x, center y, width, height)
    - IMPORTANT: Ultralytics returns xywh (center-format). Convert to top-left format:
      x = center_x - width/2, y = center_y - height/2
    - Get confidence via `results[0].boxes.conf`
    - Get class names via `results[0].names[class_id]`
    - Return empty list if no detections found
    - Add proper docstrings to class and method
  </action>
  <verify>
    ```powershell
    python -c "from src.detector import DroneDetector; print('DroneDetector imported successfully')"
    ```
    Should print success message without errors.
  </verify>
  <done>src/detector.py exists, DroneDetector class imports successfully with detect() method</done>
</task>

<task type="auto">
  <name>Create CSV writer module</name>
  <files>src/writer.py</files>
  <action>
    Create src/writer.py with:

    ```python
    class CSVWriter:
        COLUMNS = ["image_name", "x", "y", "width", "height", "confidence", "class"]

        def __init__(self, output_path: Path):
            # Store output path
            # Ensure parent directory exists

        def write(self, image_name: str, detections: list[dict]):
            # Append detections to CSV
            # If file doesn't exist, write header first
            # If detections is empty, write a row with image_name and empty values

        def reset(self):
            # Delete existing CSV file if it exists (for fresh runs)
    ```

    Implementation details:
    - Use standard library `csv` module (no pandas dependency needed)
    - Append mode so multiple images can write sequentially
    - write() checks if file exists to decide whether to write header
    - For zero-detection images, write row: image_name, "", "", "", "", "", ""
    - This ensures every input image has at least one row in output
    - Add proper docstrings
  </action>
  <verify>
    ```powershell
    python -c "from src.writer import CSVWriter; print('CSVWriter imported successfully')"
    ```
    Should print success message without errors.
  </verify>
  <done>src/writer.py exists, CSVWriter class imports successfully with write() and reset() methods</done>
</task>

## Success Criteria
- [ ] `DroneDetector` class loads YOLO26n model and has `detect()` method
- [ ] `CSVWriter` class writes detection results to CSV with correct columns
- [ ] Both modules import without errors
