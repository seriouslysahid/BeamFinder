import os
import random
import shutil
from pathlib import Path

# Config
SEED = 42
TRAIN_RATIO = 0.70
VAL_FROM_TRAIN = 0.15

SRC_IMAGES = Path("data/raw/unit1/camera_data")
SRC_LABELS = Path("data/raw/resources/bbox_labels_final")
DST_ROOT   = Path("data/processed")

def link_or_copy(src, dst):
    # try symlinking first to save space, fallback to copy if windows blocks it
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)

def main():
    stems = sorted(p.stem for p in SRC_IMAGES.glob("*.jpg"))
    print(f"Found {len(stems)} images")

    # check labels
    missing = [s for s in stems if not (SRC_LABELS / f"{s}.txt").exists()]
    if missing:
        stems = [s for s in stems if s not in set(missing)]
        print(f"Found {len(missing)} missing labels, dropping them")

    random.seed(SEED)
    random.shuffle(stems)

    n_train_pool = int(len(stems) * TRAIN_RATIO)
    train_pool = stems[:n_train_pool]
    test_stems = stems[n_train_pool:]

    n_val = int(len(train_pool) * VAL_FROM_TRAIN)
    val_stems = train_pool[-n_val:]
    train_stems = train_pool[:-n_val]

    print(f"train: {len(train_stems)}, val: {len(val_stems)}, test: {len(test_stems)}")

    # create folders
    for split in ("train", "val", "test"):
        (DST_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    splits = {"train": train_stems, "val": val_stems, "test": test_stems}

    for name, split_stems in splits.items():
        print(f"Copying {name} files...")
        for i, stem in enumerate(split_stems):
            src_img = SRC_IMAGES / f"{stem}.jpg"
            src_lbl = SRC_LABELS / f"{stem}.txt"
            dst_img = DST_ROOT / "images" / name / f"{stem}.jpg"
            dst_lbl = DST_ROOT / "labels" / name / f"{stem}.txt"

            if dst_img.exists() or dst_img.is_symlink(): dst_img.unlink()
            if dst_lbl.exists() or dst_lbl.is_symlink(): dst_lbl.unlink()

            link_or_copy(src_img, dst_img)
            link_or_copy(src_lbl, dst_lbl)

    print("Done building dataset")

if __name__ == "__main__":
    main()
