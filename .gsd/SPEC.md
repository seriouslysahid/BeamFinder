# SPEC.md — Project Specification

> **Project**: BeamFinder
> **Status**: `FINALIZED`
> **Date**: 2026-02-10

## Vision

BeamFinder is a drone detection pipeline that uses a YOLO26-nano model to localize drones in images captured by a stationary camera. The system outputs bounding box coordinates (x, y, width, height) that feed into a line-of-sight (LoS) communication system, enabling automatic beam selection to maintain a communication link with the drone.

## Goals

1. **Detect drones** in pre-recorded images using YOLO26n (pretrained COCO model)
2. **Output bounding boxes** as `(x, y, width, height)` to a CSV file
3. **Run on CPU** without requiring GPU hardware
4. **Keep it simple** — clean, maintainable Python codebase suitable for an academic project

## Non-Goals (Out of Scope — Phase 1)

- Real-time video feed processing
- Object tracking across frames
- Fine-tuning / custom training on drone-specific dataset
- Integration with the beam-steering hardware
- Web UI or visualization dashboard
- GPU optimization

## Users

- **Primary**: Academic researchers / students working on LoS drone communication
- **Secondary**: Project evaluators / professors reviewing the submission

## Domain Context

The drone operates using line-of-sight communication. A stationary ground camera captures images of the drone. The bounding box output from YOLO26 provides the drone's position in frame, which is used by the communication system to select the appropriate directional beam. Accurate and timely detection is critical for maintaining the communication link.

## Technical Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.10+ | Academic standard, Ultralytics support |
| Detector | YOLO26n (`yolo26n.pt`) | Latest Ultralytics model, NMS-free, 43% faster on CPU |
| Framework | Ultralytics | Official YOLO26 implementation |
| Output | CSV | Simple, portable, easy to parse downstream |
| Hardware | CPU-only | Accessible for academic use |

## Constraints

- **Model**: YOLO26n pretrained on COCO (no custom training for now)
- **Hardware**: Must run on CPU — no GPU dependency
- **Input**: Pre-recorded image files (JPEG/PNG) from a stationary camera
- **Academic**: Code must be clean, well-documented, and reproducible
- **COCO limitation**: COCO does not have a dedicated "drone" class; detection may rely on related classes or general object detection capability

## Future Considerations (Later Phases)

- Fine-tune YOLO26 on drone-specific dataset for improved accuracy
- Real-time video feed processing (webcam / RTSP stream)
- Object tracking (e.g., ByteTrack, BoT-SORT) for temporal consistency
- Real-time output via network socket for beam-steering integration
- Larger model variants (yolo26s/m) if accuracy needs improvement
- Visualization / annotated output images

## Success Criteria

- [ ] Pipeline accepts a directory of images as input
- [ ] YOLO26n model loads and runs inference on CPU
- [ ] Bounding box detections are written to a CSV with columns: `image_name, x, y, width, height, confidence, class`
- [ ] Script runs end-to-end without errors on the provided dataset
- [ ] Code is clean, documented, and follows Python best practices
