"""BeamFinder Detection Pipeline.

Detects drones in images using YOLO26n and outputs bounding box
coordinates to a CSV file for line-of-sight beam-steering.

Usage:
    python -m src.detect
    python -m src.detect --confidence 0.4
    python -m src.detect --input path/to/images --output path/to/out.csv
"""

import argparse
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="beamfinder",
        description="BeamFinder — Drone Detection Pipeline",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"minimum confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=IMAGE_DIR,
        help=f"path to image directory (default: {IMAGE_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help=f"path to output CSV file (default: {OUTPUT_CSV})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the BeamFinder detection pipeline.

    Args:
        argv: Optional argument list for testing (defaults to sys.argv).
    """
    args = parse_args(argv)

    confidence = args.confidence
    image_dir = args.input
    output_csv = args.output

    print("=" * 55)
    print("  BeamFinder — Drone Detection Pipeline")
    print("=" * 55)
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Confidence: {confidence}")
    print(f"  Image Size: {IMAGE_SIZE}px")
    print(f"  Input:      {image_dir}")
    print(f"  Output:     {output_csv}")
    print("=" * 55)

    # COCO pretrained model warning
    print()
    print("  ⚠  NOTE: Using COCO pretrained model.")
    print("     'drone' is NOT a COCO class — detections will be")
    print("     generic objects (person, car, bird, etc.).")
    print("     Fine-tune on a drone dataset for best results.")
    print()

    # Validate input directory
    if not image_dir.exists():
        print(f"ERROR: Image directory not found: {image_dir}")
        print("Create the directory and add your drone images.")
        sys.exit(1)

    # Find images
    images = get_image_files(image_dir)
    if not images:
        print(f"ERROR: No images found in {image_dir}")
        print(f"Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}")
        sys.exit(1)

    print(f"Found {len(images)} image(s)\n")

    # Initialize detector and writer
    print("Loading YOLO26n model...")
    detector = DroneDetector(MODEL_NAME, confidence, IMAGE_SIZE)
    writer = CSVWriter(output_csv)
    writer.reset()  # Fresh CSV for each run
    print("Model loaded successfully\n")

    # Process images
    total_detections = 0
    errors = 0
    for idx, image_path in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] {image_path.name}", end="")

        try:
            detections = detector.detect(image_path)
            writer.write(image_path.name, detections)

            count = len(detections)
            total_detections += count
            print(f" — {count} detection(s)")
        except Exception as exc:
            errors += 1
            print(f" — ERROR: {exc}")

    # Summary
    print()
    print("=" * 55)
    print("  COMPLETE")
    print(f"  Images processed: {len(images)}")
    if errors:
        print(f"  Errors:           {errors}")
    print(f"  Total detections: {total_detections}")
    print(f"  Output saved to:  {output_csv}")
    print("=" * 55)


if __name__ == "__main__":
    main()
