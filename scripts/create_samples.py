"""
Copy a small, fixed set of labeled images out of the (gitignored) converted
dataset into data/samples/ — a handful of real examples that get committed,
so anyone cloning the repo (including a reviewer) has something to look at
without downloading the full PIDray archive.
"""
import random
import shutil
from pathlib import Path

SEED = 42
NUM_SAMPLES = 6

PROCESSED_DIR = Path("data/processed/pidray")
SAMPLES_DIR = Path("data/samples")


def main() -> None:
    random.seed(SEED)

    image_files = sorted((PROCESSED_DIR / "images" / "train").glob("*.png"))
    if len(image_files) < NUM_SAMPLES:
        raise RuntimeError("Not enough converted images — run convert_pidray_to_yolo.py first.")

    # only pick images that actually have a non-empty label file, so the
    # samples are useful for a box-alignment check
    labeled_images = [
        img for img in image_files
        if (PROCESSED_DIR / "labels" / "train" / f"{img.stem}.txt").stat().st_size > 0
    ]
    chosen = random.sample(labeled_images, NUM_SAMPLES)

    images_out = SAMPLES_DIR / "images"
    labels_out = SAMPLES_DIR / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for img_path in chosen:
        label_path = PROCESSED_DIR / "labels" / "train" / f"{img_path.stem}.txt"
        shutil.copy(img_path, images_out / img_path.name)   # follows the symlink, copies real bytes
        shutil.copy(label_path, labels_out / label_path.name)
        print(f"copied {img_path.name}")

    shutil.copy(PROCESSED_DIR / "classes.txt", SAMPLES_DIR / "classes.txt")
    print(f"\n{NUM_SAMPLES} samples written to {SAMPLES_DIR}/")


if __name__ == "__main__":
    main()