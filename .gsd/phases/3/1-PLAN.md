---
phase: 3
plan: 1
wave: 1
---

# Plan 3.1: Validation Script & Edge Case Handling

## Objective
Create a validation script that tests the detection pipeline against edge cases (no detections, multiple detections, missing/corrupt files) and verifies CSV output correctness. This ensures the pipeline is robust before running on the full drone dataset.

## Context
- .gsd/SPEC.md
- src/config.py
- src/detector.py
- src/writer.py
- src/detect.py

## Tasks

<task type="auto">
  <name>Create validation test script</name>
  <files>tests/test_pipeline.py</files>
  <action>
    Create tests/test_pipeline.py with Python's built-in unittest:

    ```python
    class TestCSVWriter(unittest.TestCase):
        # test_writes_header: verify CSV has correct columns
        # test_writes_detections: verify detection rows are written
        # test_writes_empty_row: verify zero-detection images get a row
        # test_reset_clears_file: verify reset() removes CSV

    class TestDetector(unittest.TestCase):
        # test_model_loads: verify YOLO26n loads without errors
        # test_detect_returns_list: verify detect() returns list
        # test_detection_keys: verify each detection has required keys (x, y, width, height, confidence, class)

    class TestImageDiscovery(unittest.TestCase):
        # test_finds_images: verify get_image_files finds supported formats
        # test_ignores_non_images: verify non-image files are skipped
        # test_empty_directory: verify empty dir returns empty list
    ```

    Implementation details:
    - Use unittest and tempfile for isolated tests
    - Create temp directories with test files for image discovery tests
    - For CSVWriter tests, use temp files and verify CSV content with csv.reader
    - For Detector tests, use the model load test (don't need real images for basic smoke test)
    - Each test should be self-contained (setup/teardown with tempfile)
    - Do NOT require drone images to be present — tests should work anywhere
  </action>
  <verify>
    ```powershell
    python -m pytest tests/test_pipeline.py -v
    ```
    All tests should pass.
  </verify>
  <done>tests/test_pipeline.py exists with 9+ tests covering CSVWriter, Detector, and image discovery; all pass</done>
</task>

<task type="auto">
  <name>Improve pipeline edge case handling</name>
  <files>src/detect.py</files>
  <action>
    Enhance src/detect.py to handle edge cases gracefully:

    1. Add try/except around individual image processing so one bad image doesn't crash the whole pipeline
    2. Log errors for unreadable/corrupt images but continue processing
    3. Add a --confidence flag via argparse for CLI override of confidence threshold
    4. Add a --input and --output flag for custom paths
    5. Print a warning if using COCO pretrained model (reminder that "drone" isn't a COCO class)

    Keep the existing main() flow, just add robustness.
  </action>
  <verify>
    ```powershell
    python -m src.detect --help
    ```
    Should show help with available arguments.
  </verify>
  <done>detect.py handles corrupt images gracefully, supports --confidence/--input/--output CLI args</done>
</task>

## Success Criteria
- [ ] All unit tests pass with `python -m pytest tests/ -v`
- [ ] Pipeline handles corrupt/missing images without crashing
- [ ] CLI supports --confidence, --input, --output arguments
