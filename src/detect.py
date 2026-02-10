"""BeamFinder Detection Pipeline.

Detects drones in images using YOLO26n and outputs bounding box
coordinates to a CSV file for line-of-sight beam-steering.

Usage:
    python -m src.detect
"""

import sys
from pathlib import Path

from src.config import (
    CONFIDENCE_THRESHOLD,
    IMAGE_DIR,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    MODEL_NAME,
    OUTPUT_CSV,
)
from src.detector import DroneDetector
from src.writer import CSVWriter


def get_image_files(image_dir: Path) -> list[Path]:
    """Find all supported image files in a directory.

    Args:
        image_dir: Directory to search for images.

    Returns:
        Sorted list of image file paths.
    """
    files = [
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files)


def main() -> None:
    """Run the BeamFinder detection pipeline."""
    print("=" * 55)
    print("  BeamFinder — Drone Detection Pipeline")
    print("=" * 55)
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Confidence: {CONFIDENCE_THRESHOLD}")
    print(f"  Image Size: {IMAGE_SIZE}px")
    print(f"  Input:      {IMAGE_DIR}")
    print(f"  Output:     {OUTPUT_CSV}")
    print("=" * 55)
    print()

    # Validate input directory
    if not IMAGE_DIR.exists():
        print(f"ERROR: Image directory not found: {IMAGE_DIR}")
        print("Create the directory and add your drone images.")
        sys.exit(1)

    # Find images
    images = get_image_files(IMAGE_DIR)
    if not images:
        print(f"ERROR: No images found in {IMAGE_DIR}")
        print(f"Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}")
        sys.exit(1)

    print(f"Found {len(images)} image(s)\n")

    # Initialize detector and writer
    print("Loading YOLO26n model...")
    detector = DroneDetector(MODEL_NAME, CONFIDENCE_THRESHOLD, IMAGE_SIZE)
    writer = CSVWriter(OUTPUT_CSV)
    writer.reset()  # Fresh CSV for each run
    print("Model loaded successfully\n")

    # Process images
    total_detections = 0
    for idx, image_path in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] {image_path.name}", end="")

        detections = detector.detect(image_path)
        writer.write(image_path.name, detections)

        count = len(detections)
        total_detections += count
        print(f" — {count} detection(s)")

    # Summary
    print()
    print("=" * 55)
    print("  COMPLETE")
    print(f"  Images processed: {len(images)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Output saved to:  {OUTPUT_CSV}")
    print("=" * 55)


if __name__ == "__main__":
    main()
