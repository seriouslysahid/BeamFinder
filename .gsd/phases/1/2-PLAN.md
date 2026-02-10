---
phase: 1
plan: 2
wave: 1
---

# Plan 1.2: Configuration Module & README

## Objective
Create the configuration module that centralizes all settings (paths, model name, confidence threshold) and a README with setup/usage instructions. This enables the detection pipeline (Phase 2) to be cleanly parameterized.

## Context
- .gsd/SPEC.md
- .gsd/ROADMAP.md
- .gsd/phases/1/1-PLAN.md (directory structure from Plan 1.1)

## Tasks

<task type="auto">
  <name>Create configuration module</name>
  <files>src/config.py</files>
  <action>
    Create src/config.py with:
    ```python
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
    ```

    Design decisions:
    - Use pathlib.Path for cross-platform compatibility
    - PROJECT_ROOT is computed relative to config.py location
    - Confidence threshold at 0.25 (YOLO default) — can be tuned later
    - IMAGE_SIZE at 640 (standard YOLO input size)
    - Do NOT use environment variables for now — this is an academic project, simplicity wins
  </action>
  <verify>
    ```powershell
    python -c "from src.config import PROJECT_ROOT, MODEL_NAME, CONFIDENCE_THRESHOLD; print(f'Root: {PROJECT_ROOT}, Model: {MODEL_NAME}, Conf: {CONFIDENCE_THRESHOLD}')"
    ```
    Should print correct values without errors.
  </verify>
  <done>src/config.py imports successfully and exposes PROJECT_ROOT, IMAGE_DIR, OUTPUT_DIR, OUTPUT_CSV, MODEL_NAME, CONFIDENCE_THRESHOLD, IMAGE_SIZE, IMAGE_EXTENSIONS</done>
</task>

<task type="auto">
  <name>Create README.md</name>
  <files>README.md</files>
  <action>
    Create README.md with:
    - Project title and one-line description
    - Overview section explaining the drone detection → beam-steering use case
    - Prerequisites (Python 3.10+)
    - Installation steps (clone, create venv, pip install -r requirements.txt)
    - Usage section (placeholder — will be filled in Phase 2)
    - Project structure tree
    - Configuration section referencing src/config.py
    - License section (MIT or as user prefers)

    Keep it clean and professional — this is an academic submission.
    Do NOT add badges or unnecessary decoration.
  </action>
  <verify>
    ```powershell
    Test-Path "README.md"
    ```
    File exists with complete content.
  </verify>
  <done>README.md exists with project overview, setup instructions, and project structure</done>
</task>

## Success Criteria
- [ ] `src/config.py` exists and imports correctly with `python -c "from src.config import *"`
- [ ] `README.md` exists with setup instructions and project overview
- [ ] Configuration values are sensible defaults (confidence 0.25, image size 640, model yolo26n.pt)
