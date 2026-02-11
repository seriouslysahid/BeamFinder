# FINDINGS.md — BeamFinder Weekly Research Report

> **Project**: BeamFinder — Drone Detection for LoS Beam-Steering
> **Author**: Sahid
> **Week**: 2026-02-11
> **Status**: Phase 2 Complete (Detection Pipeline Built), Phase 3 Planned

---

## 1. Dataset Findings

### 1.1 Dataset Structure

| Split | Subfolders | Images | Folder Range | Avg/Folder |
|-------|-----------|--------|--------------|------------|
| train | 56 | 7,950 | 2–60 | 142 |
| train_stable | 51 | 7,970 | 0–50 | 156 |
| validation | 56 | 3,437 | 2–60 | 61 |
| validation_stable | 51 | 3,416 | 0–50 | 67 |
| **Total** | — | **22,773** | — | — |

- **Resolution**: 960 × 540 px (uniform across all images)
- **Format**: RGB JPG
- **Naming convention**: `image_BS1_XXXXX_HH_MM_SS.jpg` (timestamp-based, BS1 = base station 1)

### 1.2 Subfolder Semantics — Beam Indices

The numbered subfolders represent **beam indices** (ground-truth labels), not classes or scenes. Evidence:

- **5,559 images** (70%) are shared between `train` and `train_stable`, but appear in **different subfolders**
  - Example: `image_BS1_8564_17_51_58.jpg` → `train/52` vs `train_stable/45`
  - Example: `image_BS1_9805_17_55_32.jpg` → `train/33` vs `train_stable/28`
- This indicates two different **beam codebook schemes** applied to the same image set
- `train` / `validation`: beam indices **2–60** (56 beams)
- `train_stable` / `validation_stable`: beam indices **0–50** (51 beams)

**⚠️ Open Question**: What is the relationship between the two codebook schemes? Is one a compressed/quantized version of the other? The offset between paired labels appears non-constant (52→45 = -7, 33→28 = -5, 59→50 = -9), suggesting a non-trivial remapping.

### 1.3 Class Imbalance

Image distribution across beam indices is **heavily imbalanced**:
- **Train**: 1 to 1,162 images per folder (142 avg)
- **Validation**: 0 to 498 images per folder (61 avg)
- **5 validation folders are completely empty**: folders 12, 24, 35, 56, 60

This imbalance may bias any classifier toward high-frequency beam indices. Mitigation strategies (oversampling, class weighting, focal loss) should be considered.

---

## 2. Visual Findings

### 2.1 Target Object — Hexacopter Drone

All images contain a single **hexacopter drone** (6-rotor) against a clear blue sky, captured from a **stationary ground-level upward-pointing camera**. The drone varies in:
- **Position**: Appears at different (x, y) coordinates across images, spanning the full frame
- **Apparent size**: Typically 20–40 px across in a 960×540 frame (~2–4% of image width)

### 2.2 Visual Occlusions and Noise

| Issue | Severity | Frequency | Description |
|-------|----------|-----------|-------------|
| **Sun glare** | Medium | ~80% of images | Bright lens flare streak crosses the frame diagonally, may interfere with detection |
| **Human head occlusion** | High | ~20% of images | A person's head partially visible in upper-left corner, could trigger false positives or occlude the drone |
| **Foliage at frame edge** | Low | ~5% of images | Tree leaves visible at the extreme top-left corner |
| **Very small target** | High | All images | Drone occupies <1% of total pixel area, challenging for standard object detectors |

### 2.3 Background Uniformity

The uniform blue sky background is a **double-edged finding**:
- **Pro**: High contrast between dark drone and bright sky enables simple thresholding approaches
- **Con**: YOLO and similar detectors rely on contextual features — a featureless background provides little context for learning

---

## 3. Architectural & Methodological Issues

### 3.1 COCO Pretrained Model Cannot Detect Drones

The current pipeline uses YOLO26n pretrained on **COCO dataset**. COCO has 80 classes and **"drone" is not one of them**. The pretrained model will:
- ❌ Not detect the hexacopter as "drone"
- ⚠️ May detect it as "bird" or "kite" with low confidence, or miss it entirely
- ❌ Will likely detect the human head as "person" (false positive for this use case)

**Implication**: The pretrained model serves as a pipeline skeleton only. Fine-tuning on drone-annotated data or an alternative detection method is required for actual detection.

### 3.2 Missing Bounding Box Annotations

The dataset provides **beam index labels** (via subfolder structure) but appears to have **no bounding box annotations** (no `.txt`, `.xml`, or `.json` annotation files alongside images).

This creates a gap:
- **For detection**: YOLO fine-tuning requires bounding box annotations in YOLO format (`class cx cy w h` per line). These need to be created (manually, semi-automatically, or via an auto-labeling pipeline).
- **For classification**: Beam index labels exist and can be used directly for image classification tasks.

### 3.3 Problem Reformulation Needed

The original SPEC defined the task as:
> "Detect drones using YOLO26n, output (x, y, w, h) to CSV"

Given the dataset structure, the actual research problem is more nuanced:

| Approach | Task | Model | Labels Needed | Feasibility |
|----------|------|-------|---------------|-------------|
| **A. Detection only** | Locate drone → output bbox | YOLO26n (fine-tuned) | Bounding box annotations | Requires annotation effort |
| **B. Classification only** | Image → beam index | ResNet/EfficientNet/ViT | Beam labels (have these ✅) | Directly feasible |
| **C. Hybrid** | Detect → position → beam | YOLO + simple mapper | Both bbox + beam labels | Most complete, most effort |
| **D. Simple CV** | Threshold/contour → position → beam | OpenCV (no ML) | Beam labels | Quick baseline, exploits uniform sky |

**Recommendation**: Start with **Approach D** as a baseline (contour detection on blue sky is trivial), then build toward **Approach B or C** for the actual contribution.

### 3.4 Train vs Stable Split Purpose

The existence of two parallel datasets (`train` / `train_stable` and `validation` / `validation_stable`) with different beam indexing on the same images raises questions:
- Are these two different antenna array configurations?
- Is "stable" a temporal stability-filtered subset?
- Which codebook should be used as the primary target?

**⚠️ This needs clarification from the professor.**

---

## 4. Pipeline Status & Technical Findings

### 4.1 Current Pipeline State

| Component | Status | Notes |
|-----------|--------|-------|
| Project structure | ✅ Complete | `src/`, `data/`, `output/` |
| Configuration module | ✅ Complete | `src/config.py` |
| Detector module | ✅ Complete | `src/detector.py` — YOLO26n wrapper |
| CSV writer | ✅ Complete | `src/writer.py` — appending CSV output |
| Main script | ✅ Complete | `src/detect.py` — end-to-end pipeline |
| Unit tests | ⬜ Planned | Phase 3 |
| Bounding box annotations | ❌ Missing | Needed for fine-tuning |
| Model fine-tuning | ❌ Not started | Blocked by annotations |

### 4.2 Pipeline Limitation — Flat Directory Assumption

The current `detect.py` scans images from a **flat directory** (`data/images/`), but the dataset uses **nested subdirectories** (beam-labeled folders). The pipeline needs to be updated to:
1. Accept a specific split path (e.g., `data/images/train`)
2. Recursively discover images in subfolders
3. Preserve the subfolder (beam index) metadata in CSV output

### 4.3 CPU-Only Constraint

Running YOLO26n inference on 22,773 images with CPU-only will be slow. Estimated throughput:
- ~0.3–0.5 sec/image on modern CPU
- **Full dataset: ~3–4 hours** processing time
- This is acceptable for batch processing but rules out real-time applications

---

## 5. Open Questions for Professor

1. **Beam codebook**: What is the relationship between the two codebook schemes (folders 2–60 vs 0–50)? Which should be used as the primary target?
2. **Bounding box annotations**: Are bounding box annotations available separately, or should we create them? Is auto-labeling (e.g., contour-based) acceptable?
3. **Problem scope**: Is the expected output (a) drone bounding box only, (b) beam index prediction, or (c) both?
4. **Stable vs non-stable**: What defines the "stable" variant? Is it a different antenna configuration, temporal filtering, or label smoothing?
5. **Evaluation metric**: What metric should be used — detection mAP, beam prediction accuracy (top-1, top-K), or positional error?

---

## 6. Next Steps (Planned)

1. [ ] Update pipeline to handle nested directory structure
2. [ ] Run YOLO26n on sample images to establish baseline (even with COCO model)
3. [ ] Implement simple contour-based detection as alternative baseline
4. [ ] Begin bounding box annotation (if needed) or classification approach
5. [ ] Create Phase 3 tests and validation framework

---

*Last updated: 2026-02-11 05:31 IST*
