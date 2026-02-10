"""BeamFinder CSV Writer Module.

Writes detection results to CSV files for downstream beam-steering consumption.
"""

import csv
from pathlib import Path


class CSVWriter:
    """Writes bounding box detections to a CSV file.

    CSV columns: image_name, x, y, width, height, confidence, class

    Attributes:
        output_path: Path to the output CSV file.
    """

    COLUMNS = ["image_name", "x", "y", "width", "height", "confidence", "class"]

    def __init__(self, output_path: Path) -> None:
        """Initialize the CSV writer.

        Args:
            output_path: Path where the CSV file will be written.
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, image_name: str, detections: list[dict]) -> None:
        """Append detection results for one image to the CSV file.

        If the CSV file doesn't exist, a header row is written first.
        If no detections are found, a row with the image name and empty
        values is written to ensure every input image appears in output.

        Args:
            image_name: Name of the source image file.
            detections: List of detection dicts from DroneDetector.detect().
        """
        write_header = not self.output_path.exists()

        with open(self.output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)

            if write_header:
                writer.writeheader()

            if not detections:
                # Write a row with image name but no detection data
                writer.writerow({
                    "image_name": image_name,
                    "x": "",
                    "y": "",
                    "width": "",
                    "height": "",
                    "confidence": "",
                    "class": "",
                })
            else:
                for det in detections:
                    writer.writerow({
                        "image_name": image_name,
                        **det,
                    })

    def reset(self) -> None:
        """Delete existing CSV file for a fresh run."""
        if self.output_path.exists():
            self.output_path.unlink()
