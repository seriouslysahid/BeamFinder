---
phase: 2
plan: 2
wave: 2
---

# Plan 2.2: Main Detection Script

## Objective
Create the main entry point that ties the detector and CSV writer together into a complete pipeline: load images → detect → write CSV. This is the script users will run.

## Context
- .gsd/SPEC.md
- src/config.py
- src/detector.py (from Plan 2.1)
- src/writer.py (from Plan 2.1)

## Tasks

<task type="auto">
  <name>Create main detection script</name>
  <files>src/detect.py</files>
  <action>
    Create src/detect.py with:

    ```python
    """BeamFinder Detection Pipeline.

    Usage: python -m src.detect
    """

    def get_image_files(image_dir: Path) -> list[Path]:
        # Find all image files in directory matching IMAGE_EXTENSIONS
        # Sort alphabetically for reproducible output
        # Return list of Path objects

    def main():
        # 1. Print banner with config info
        # 2. Validate IMAGE_DIR exists and has images
        # 3. Initialize DroneDetector with config values
        # 4. Initialize CSVWriter with OUTPUT_CSV path, call reset()
        # 5. Loop through images:
        #    - Print progress (e.g., "[3/50] Processing image_003.jpg")
        #    - Call detector.detect(image_path)
        #    - Call writer.write(image_name, detections)
        #    - Print detection count per image
        # 6. Print summary (total images, total detections, output path)

    if __name__ == "__main__":
        main()
    ```

    Implementation details:
    - Import from src.config for all settings
    - Import DroneDetector from src.detector
    - Import CSVWriter from src.writer
    - Use pathlib for all file operations
    - Print clear progress to stdout (this is academic — user wants to see what's happening)
    - Handle edge case: IMAGE_DIR doesn't exist or has no images → print error and exit
    - Do NOT catch exceptions in detect() — let errors surface clearly
    - Ensure it runs as module: `python -m src.detect`
  </action>
  <verify>
    ```powershell
    python -c "from src.detect import main, get_image_files; print('detect.py imports OK')"
    ```
    Should print success message without errors.
  </verify>
  <done>src/detect.py exists, imports correctly, has main() and get_image_files() functions</done>
</task>

<task type="checkpoint:human-verify">
  <name>End-to-end pipeline test</name>
  <files>output/detections.csv</files>
  <action>
    Run the full pipeline on any available test image:
    1. If user has images in data/images/, run on those
    2. If no images available, download a sample image for testing (e.g., from Ultralytics test assets)
    3. Run: python -m src.detect
    4. Verify output/detections.csv is created with correct columns
    5. Print CSV contents for user to inspect
  </action>
  <verify>
    ```powershell
    Test-Path "output/detections.csv"
    Get-Content "output/detections.csv" | Select-Object -First 5
    ```
    CSV file exists with header row and detection data.
  </verify>
  <done>Pipeline runs end-to-end: images → YOLO26n → CSV output with correct format</done>
</task>

## Success Criteria
- [ ] `python -m src.detect` runs without errors
- [ ] `output/detections.csv` is created with correct columns
- [ ] CSV contains bounding box data for detected objects
- [ ] Progress output is printed to console during execution
