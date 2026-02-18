# Known Issues & Problems

Issues encountered during development of the BeamFinder pipeline. Documented for professor review.

---

## 1. COCO Model Does Not Have a "Drone" Class

**Status:** Known limitation

The pretrained YOLO26n model is trained on the COCO dataset, which has 80 object classes (person, car, bird, etc.). There is no "drone" class in COCO. Any object detection with this model will return generic objects, not drones specifically. Fine-tuning on a drone-annotated dataset is required for actual drone detection.

---

## 2. No Bounding Box Annotations in Dataset

**Status:** Blocking for training

The dataset contains 7,970 training images and 3,416 validation images organized in numbered subfolders (0-50). However, there are no corresponding label files (`.txt` files with bounding box coordinates in YOLO format).

YOLO detection training requires a `.txt` file per image with lines in the format:
```
<class_id> <x_center> <y_center> <width> <height>
```
where all values are normalized (0-1).

**Options to resolve:**
1. Manually annotate using a tool like [Label Studio](https://labelstud.io/) or [Roboflow](https://roboflow.com/)
2. Use the pretrained COCO model to auto-label images, then manually review and correct
3. Obtain pre-annotated drone detection datasets from sources like Roboflow

**Awaiting professor guidance on annotation approach.**

---

## 3. Severe Class Imbalance Across Subfolders

**Status:** Observation

The training images are unevenly distributed across the 51 subfolders — some folders have as few as 1 image while others have over 1,000. If the folders represent different capture scenarios, this imbalance may affect training depending on how diverse the conditions are (lighting, angles, backgrounds).

---

## 4. Windows Multiprocessing Error

**Status:** Resolved

YOLO26 docs warn that on Windows, training scripts must wrap code in `if __name__ == "__main__":` to avoid `RuntimeError` from Python's multiprocessing. Both `train.py` and `detect.py` have been written accounting for this.

`workers` is set to 2 (instead of default 8) to prevent multiprocessing issues on Windows.

---

## 5. Memory Usage with Large Dataset

**Status:** Mitigated

With 11,386 images at 960x540 (~650 MB total), loading all images into RAM could cause overflow. Mitigated by:
- Using `cache='disk'` instead of `cache=True` in training
- Using `stream=True` in detection
- Using `batch=-1` for automatic memory-based batch sizing

---

## 6. Aspect Ratio Mismatch

**Status:** Known limitation

Images are 960x540 (16:9 widescreen) but YOLO resizes to 640x640 square for inference. This introduces letterboxing (black padding) which slightly reduces effective resolution. This is standard YOLO behavior and should not significantly affect results.
