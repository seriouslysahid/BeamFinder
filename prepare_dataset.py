import os
import random
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

IMAGE_SRC = Path("dataset/unit1/camera_data")
LABEL_SRC = Path("dataset/resources/bbox_labels_final")

DATA_ROOT = Path("data")

SPLITS = {
    "train":      (DATA_ROOT / "images" / "train",      DATA_ROOT / "labels" / "train"),
    "validation": (DATA_ROOT / "images" / "validation",  DATA_ROOT / "labels" / "validation"),
    "test":       (DATA_ROOT / "images" / "test",        DATA_ROOT / "labels" / "test"),
}


def find_matched_pairs():
    images = sorted(IMAGE_SRC.glob("*.jpg"))
    matched = []
    skipped = []

    for img in images:
        label = LABEL_SRC / (img.stem + ".txt")
        if label.exists():
            matched.append((img, label))
        else:
            skipped.append(img.name)

    return matched, skipped


def create_symlinks(pairs, img_dir, lbl_dir):
    count = 0
    for img_path, lbl_path in pairs:
        img_dst = img_dir / img_path.name
        lbl_dst = lbl_dir / lbl_path.name

        if not img_dst.exists():
            os.symlink(img_path.resolve(), img_dst)
        if not lbl_dst.exists():
            os.symlink(lbl_path.resolve(), lbl_dst)
        count += 1
    return count


def main():
    print("=" * 55)
    print("  BeamFinder - Dataset Preparation")
    print("=" * 55)
    print()

    matched, skipped = find_matched_pairs()
    print(f"  Images found:   {len(matched) + len(skipped)}")
    print(f"  With labels:    {len(matched)}")
    if skipped:
        print(f"  Missing labels: {len(skipped)}")
        for name in skipped[:5]:
            print(f"    - {name}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")
    print()

    if not matched:
        print("  ERROR: no matched pairs found. Check IMAGE_SRC and LABEL_SRC paths.")
        return

    random.seed(SEED)
    random.shuffle(matched)

    train_end = int(len(matched) * TRAIN_RATIO)
    val_end = train_end + int(len(matched) * VAL_RATIO)

    splits = {
        "train":      matched[:train_end],
        "validation": matched[train_end:val_end],
        "test":       matched[val_end:],
    }

    for name, pairs in splits.items():
        print(f"  {name:12s}: {len(pairs)} samples")
    print()

    for split_name, (img_dir, lbl_dir) in SPLITS.items():
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n = create_symlinks(splits[split_name], img_dir, lbl_dir)
        print(f"  Linked {n} pairs -> {split_name}/")

    print()
    print("=" * 55)
    print(f"  Done! Dataset ready at: {DATA_ROOT.resolve()}")
    print("=" * 55)


if __name__ == "__main__":
    main()
