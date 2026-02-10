# REQUIREMENTS.md

## Format

| ID | Requirement | Source | Status |
|----|-------------|--------|--------|
| REQ-01 | Accept a directory of images (JPEG/PNG) as input | SPEC goal 1 | Pending |
| REQ-02 | Load YOLO26n pretrained model (yolo26n.pt) | SPEC goal 1 | Pending |
| REQ-03 | Run inference on each image on CPU | SPEC goals 1, 3 | Pending |
| REQ-04 | Extract bounding boxes as (x, y, width, height) per detection | SPEC goal 2 | Pending |
| REQ-05 | Write detections to CSV with columns: image_name, x, y, width, height, confidence, class | SPEC goal 2 | Pending |
| REQ-06 | Handle images with zero detections gracefully | SPEC goal 4 | Pending |
| REQ-07 | Handle images with multiple detections | SPEC goal 4 | Pending |
| REQ-08 | Configurable confidence threshold | SPEC goal 4 | Pending |
| REQ-09 | Clean, documented code with docstrings | SPEC goal 4 | Pending |
| REQ-10 | README with setup and usage instructions | SPEC goal 4 | Pending |
