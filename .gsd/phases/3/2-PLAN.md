---
phase: 3
plan: 2
wave: 2
---

# Plan 3.2: Dataset Run & Results Documentation

## Objective
Run the pipeline on the user's drone images, validate the output, and document the results. This is the final validation step before polishing for submission.

## Context
- .gsd/SPEC.md
- src/detect.py (updated from Plan 3.1)
- output/detections.csv

## Tasks

<task type="checkpoint:human-verify">
  <name>Run pipeline on drone dataset</name>
  <files>output/detections.csv</files>
  <action>
    1. Confirm user has placed drone images in data/images/
    2. Run: python -m src.detect
    3. Display output/detections.csv summary:
       - Total images processed
       - Total detections found
       - Detection rate (images with at least one detection / total images)
       - Class distribution
    4. Print first 10 rows of CSV for user inspection
    5. If detection rate is very low: note that COCO pretrained model may not detect drones well, recommend future fine-tuning
  </action>
  <verify>
    ```powershell
    Test-Path "output/detections.csv"
    (Import-Csv "output/detections.csv" | Measure-Object).Count
    ```
    CSV exists with rows for each processed image.
  </verify>
  <done>Pipeline runs on user's drone dataset, CSV output reviewed by user</done>
</task>

<task type="auto">
  <name>Create results summary document</name>
  <files>RESULTS.md</files>
  <action>
    Create RESULTS.md at project root with:
    - Run configuration (model, confidence, image count)
    - Detection statistics (total detections, detection rate, class breakdown)
    - Sample output rows
    - Observations and notes
    - Recommendations for improvement (fine-tuning, confidence tuning)

    This serves as documentation for the academic submission.
  </action>
  <verify>
    ```powershell
    Test-Path "RESULTS.md"
    ```
  </verify>
  <done>RESULTS.md exists with run statistics and analysis</done>
</task>

## Success Criteria
- [ ] Pipeline runs successfully on user's drone images
- [ ] output/detections.csv contains results for all images
- [ ] RESULTS.md documents the run with statistics and observations
