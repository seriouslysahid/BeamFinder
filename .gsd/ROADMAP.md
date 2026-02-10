# ROADMAP.md

> **Project**: BeamFinder
> **Current Phase**: Not started
> **Milestone**: v1.0

## Must-Haves (from SPEC)

- [ ] Drone detection using YOLO26n on pre-recorded images
- [ ] Bounding box output (x, y, width, height) to CSV
- [ ] CPU-only execution
- [ ] Clean, documented Python codebase

## Phases

### Phase 1: Foundation
**Status**: ✅ Complete
**Objective**: Set up Python project structure, dependencies, and basic configuration
**Deliverables**:
- Project directory structure (`src/`, `data/`, `output/`)
- `requirements.txt` with Ultralytics and dependencies
- Configuration module (paths, model settings, confidence threshold)
- README.md with setup instructions

### Phase 2: Detection Pipeline
**Status**: ✅ Complete
**Objective**: Implement the core YOLO26n detection pipeline
**Deliverables**:
- Image loading and preprocessing module
- YOLO26n model loading and inference
- Bounding box extraction (x, y, width, height, confidence, class)
- CSV output writer
- Main detection script (`detect.py`)

### Phase 3: Testing & Validation
**Status**: ⬜ Not Started
**Objective**: Verify pipeline correctness on the drone image dataset
**Deliverables**:
- Run detection on full image dataset
- Validate CSV output format and correctness
- Document results and detection accuracy
- Handle edge cases (no detections, multiple detections)

### Phase 4: Documentation & Submission
**Status**: ⬜ Not Started
**Objective**: Polish code and prepare for academic submission
**Deliverables**:
- Code comments and docstrings
- Complete README with usage instructions
- Sample output / results documentation
- Clean git history

---

## Future Milestones (Post v1.0)

### v1.1 — Fine-Tuned Model
- Custom drone dataset labeling
- YOLO26n fine-tuning on drone-specific data
- Evaluation metrics (mAP, precision, recall)

### v1.2 — Real-Time Video
- Live video feed support (webcam / RTSP)
- Frame-by-frame processing
- Real-time output stream

### v1.3 — Tracking & Integration
- Multi-frame object tracking (ByteTrack / BoT-SORT)
- Network socket output for beam-steering system
- Latency optimization
