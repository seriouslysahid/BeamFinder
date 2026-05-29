# Product Overview

BeamFinder is a drone detection and mmWave beam prediction system for 6G wireless communication. It uses computer vision to detect drones in camera imagery and predicts optimal transmission beam indices for directional communication.

## Core Functionality

- **Drone Detection**: Fine-tunes YOLO26s on the DeepSense 6G Scenario 23 dataset to detect drones in images
- **Beam Clustering**: Uses geometric centroid clustering to map detected drone positions to optimal mmWave beam indices
- **Unified Inference**: Combines detection and beam prediction into a single pipeline

## Key Innovation

Unlike the reference paper (Charan et al., arXiv:2205.12187) which uses ResNet-50 for end-to-end detection and classification, BeamFinder separates concerns:
- YOLO26s handles drone detection
- Euclidean distance to pre-calculated centroids handles beam classification

This disjoint architecture is faster and allows independent tuning of detection and beam prediction.

## Dataset

Built for the DeepSense 6G Scenario 23 dataset:
- 11,387 images with ground-truth bounding boxes
- 70/30 train/test split (with 15% validation carved from training)
- Images are 960×540 resolution

## Outputs

- `best.pt`: Trained YOLO model weights
- `detections.csv`: Final inference results with coordinates, confidence scores, and beam predictions
- Annotated images with bounding boxes
