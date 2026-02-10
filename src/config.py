"""BeamFinder Configuration Module.

Centralizes all configurable parameters for the detection pipeline.
"""

from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_CSV = OUTPUT_DIR / "detections.csv"

# === Model Settings ===
MODEL_NAME = "yolo26n.pt"  # YOLO26 nano — CPU-optimized
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence to keep a detection
IMAGE_SIZE = 640  # Inference image size (pixels)

# === Supported Image Extensions ===
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
