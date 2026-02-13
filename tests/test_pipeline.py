"""BeamFinder Pipeline Tests.

Self-contained tests for the detection pipeline components.
Does NOT require drone images to be present — uses temp files and built-in
unittest infrastructure for isolation.
"""

import csv
import os
import tempfile
import unittest
from pathlib import Path


class TestCSVWriter(unittest.TestCase):
    """Tests for src.writer.CSVWriter."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        )
        self.tmp.close()
        self.csv_path = Path(self.tmp.name)
        # Remove file so writer can create it fresh
        self.csv_path.unlink(missing_ok=True)

        from src.writer import CSVWriter
        self.writer = CSVWriter(self.csv_path)

    def tearDown(self):
        if self.csv_path.exists():
            self.csv_path.unlink()

    # ── Header ───────────────────────────────────────────────────
    def test_writes_header(self):
        """CSV should have the correct column headers."""
        self.writer.write("test.jpg", [])
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        expected = ["image_name", "x", "y", "width", "height", "confidence", "class"]
        self.assertEqual(header, expected)

    # ── Detection rows ───────────────────────────────────────────
    def test_writes_detections(self):
        """Detection dicts should appear as CSV rows."""
        dets = [
            {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0,
             "confidence": 0.85, "class": "person"},
            {"x": 50.0, "y": 60.0, "width": 70.0, "height": 80.0,
             "confidence": 0.72, "class": "car"},
        ]
        self.writer.write("img.png", dets)
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["image_name"], "img.png")
        self.assertEqual(rows[0]["class"], "person")
        self.assertEqual(rows[1]["class"], "car")

    # ── Empty row (no detections) ────────────────────────────────
    def test_writes_empty_row(self):
        """Zero-detection images should still get a row with blanks."""
        self.writer.write("empty.jpg", [])
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image_name"], "empty.jpg")
        self.assertEqual(rows[0]["x"], "")

    # ── Reset clears file ────────────────────────────────────────
    def test_reset_clears_file(self):
        """reset() should delete the CSV file."""
        self.writer.write("test.jpg", [])
        self.assertTrue(self.csv_path.exists())
        self.writer.reset()
        self.assertFalse(self.csv_path.exists())

    # ── Multiple writes append ───────────────────────────────────
    def test_multiple_writes_append(self):
        """Successive write() calls should append, not overwrite."""
        self.writer.write("a.jpg", [])
        self.writer.write("b.jpg", [])
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        names = [r["image_name"] for r in rows]
        self.assertIn("a.jpg", names)
        self.assertIn("b.jpg", names)


class TestDetector(unittest.TestCase):
    """Smoke tests for src.detector.DroneDetector."""

    @classmethod
    def setUpClass(cls):
        from src.detector import DroneDetector
        from src.config import MODEL_NAME, CONFIDENCE_THRESHOLD, IMAGE_SIZE
        cls.detector = DroneDetector(MODEL_NAME, CONFIDENCE_THRESHOLD, IMAGE_SIZE)

    # ── Model loads ──────────────────────────────────────────────
    def test_model_loads(self):
        """YOLO26n model should load without errors."""
        self.assertIsNotNone(self.detector.model)

    # ── detect() returns list ────────────────────────────────────
    def test_detect_returns_list(self):
        """detect() should return a list (even if empty)."""
        # Create a tiny blank image for smoke test
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            # Write a minimal valid JPEG (1x1 white pixel)
            _write_tiny_jpeg(tmp_path)
            result = self.detector.detect(tmp_path)
            self.assertIsInstance(result, list)
        finally:
            tmp_path.unlink(missing_ok=True)

    # ── Detection keys ───────────────────────────────────────────
    def test_detection_keys(self):
        """Each detection dict should have the required keys."""
        required = {"x", "y", "width", "height", "confidence", "class"}
        # Create a small image — may or may not produce detections
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _write_tiny_jpeg(tmp_path)
            results = self.detector.detect(tmp_path)
            for det in results:
                self.assertTrue(
                    required.issubset(det.keys()),
                    f"Missing keys: {required - det.keys()}",
                )
        finally:
            tmp_path.unlink(missing_ok=True)


class TestImageDiscovery(unittest.TestCase):
    """Tests for src.detect.get_image_files."""

    def setUp(self):
        from src.detect import get_image_files
        self.get_image_files = get_image_files
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ── Finds images ─────────────────────────────────────────────
    def test_finds_images(self):
        """Should find files with supported extensions."""
        for name in ["a.jpg", "b.png", "c.bmp"]:
            (Path(self.tmp_dir) / name).touch()
        result = self.get_image_files(Path(self.tmp_dir))
        self.assertEqual(len(result), 3)

    # ── Ignores non-images ───────────────────────────────────────
    def test_ignores_non_images(self):
        """Non-image files should be skipped."""
        (Path(self.tmp_dir) / "readme.txt").touch()
        (Path(self.tmp_dir) / "data.csv").touch()
        (Path(self.tmp_dir) / "photo.jpg").touch()
        result = self.get_image_files(Path(self.tmp_dir))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "photo.jpg")

    # ── Empty directory ──────────────────────────────────────────
    def test_empty_directory(self):
        """Empty directory should return empty list."""
        result = self.get_image_files(Path(self.tmp_dir))
        self.assertEqual(result, [])

    # ── Returns sorted ───────────────────────────────────────────
    def test_returns_sorted(self):
        """Results should be sorted by filename."""
        for name in ["z.jpg", "a.jpg", "m.jpg"]:
            (Path(self.tmp_dir) / name).touch()
        result = self.get_image_files(Path(self.tmp_dir))
        names = [f.name for f in result]
        self.assertEqual(names, ["a.jpg", "m.jpg", "z.jpg"])


# ── Helpers ──────────────────────────────────────────────────────

def _write_tiny_jpeg(path: Path) -> None:
    """Write a minimal 1x1 white JPEG to disk."""
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img.save(str(path))
    except ImportError:
        # Fallback: write raw JFIF header (enough for YOLO to attempt load)
        import struct
        path.write_bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xd9"
        )


if __name__ == "__main__":
    unittest.main()
