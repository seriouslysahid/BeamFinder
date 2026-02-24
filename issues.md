# Known Issues & Limitations

Problems we ran into while building the BeamFinder pipeline, and how we dealt with them.

---

## 1. COCO pretrained model has no "drone" class

**Status:** Expected, fixed by fine-tuning

YOLO26s comes pretrained on COCO (80 classes - person, car, bird, etc). There's no drone class, so out-of-the-box inference won't detect drones. This is why we fine-tune on our own annotated dataset from DeepSense Scenario 23.

---

## 2. Finding the bounding box annotations

**Status:** Resolved

Initially thought the dataset didn't include bbox labels. Turns out the 11,387 YOLO-format `.txt` files were in the original DeepSense download. They've been paired with images and organized into the standard YOLO directory layout under `data/` with a 70/15/15 train/val/test split (7,970 / 1,708 / 1,709).

---

## 3. Uneven distribution across capture sessions

**Status:** Noted

The raw images come from 51 different capture subfolders with very uneven counts (some have 1 image, others 1000+). Since we shuffle before splitting, the train/val/test sets should have a reasonable mix of conditions. Haven't seen this cause problems in practice.

---

## 4. Windows multiprocessing doesn't work

**Status:** Worked around

On Windows, setting `workers > 0` in Ultralytics causes a `RuntimeError` from Python's multiprocessing module. Both scripts use `if __name__ == "__main__":` (required) and `workers=0` (required on our machine). Training is a bit slower because of single-threaded data loading, but `cache="ram"` compensates for this.

---

## 5. Memory considerations

**Status:** Under control

The dataset is about 650MB of images. We cache everything in RAM during training (`cache="ram"`) which needs about 4GB of system memory but eliminates disk I/O as a bottleneck. If your machine has less than 16GB RAM, change it to `cache="disk"` in train.py.

VRAM-wise, the RTX 3050 has 4GB. We use `batch=0.85` (85% GPU memory utilization) and `amp=True` (FP16) so Ultralytics picks the largest batch that fits. At `imgsz=960` this is usually batch 2-4.

---

## 6. Aspect ratio mismatch

**Status:** Mitigated with rect=True

Images are 960x540 (16:9) but YOLO defaults to square inputs. Without `rect=True`, about 44% of pixels would be black padding. We enable rectangular training/inference which preserves the aspect ratio and avoids wasting compute on padding.
