"""
Convert PIDray's COCO-style JSON annotations into Ultralytics YOLO format.

Expects:
    data/raw/pidray/{train,easy,hard,hidden}/*.png
    data/raw/pidray/annotations/{xray_train,
                                xray_test_easy,
                                xray_test_hard,
                                xray_test_hidden}.json

Produces (gitignored, regenerate anytime by re-running this script):
    data/processed/pidray/images/{train,easy,hard,hidden}/*.png  (symlinks)
    data/processed/pidray/labels/{train,easy,hard,hidden}/*.txt
    data/processed/pidray/classes.txt
    data/processed/pidray/pidray_data.yaml
"""
import json
import shutil
from pathlib import Path

import yaml

RAW_DIR = Path("data/raw/pidray")
OUT_DIR = Path("data/processed/pidray")
SPLITS = ["train", "easy", "hard", "hidden"]

ANNOTATION_FILES = {
    "train": "xray_train.json",
    "easy": "xray_test_easy.json",
    "hard": "xray_test_hard.json",
    "hidden": "xray_test_hidden.json",
}


def convert_split(split: str, category_id_to_index: dict[int, int]) -> None:
    with open(RAW_DIR / "annotations" / ANNOTATION_FILES[split]) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}

    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    img_out_dir = OUT_DIR / "images" / split
    lbl_out_dir = OUT_DIR / "labels" / split
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    for image_id, image in images_by_id.items():
        file_name = Path(image["file_name"]).name  # strip any nested path
        width, height = image["width"], image["height"]

        src_image = RAW_DIR / split / file_name
        dst_image = img_out_dir / file_name
        if not dst_image.exists():
            try:
                dst_image.symlink_to(src_image.resolve())
            except OSError:
                shutil.copy(src_image, dst_image)  # Windows fallback

        lines = []
        for ann in anns_by_image.get(image_id, []):
            x, y, w, h = ann["bbox"]  # COCO: top-left x, y, width, height
            cx, cy = (x + w / 2) / width, (y + h / 2) / height
            nw, nh = w / width, h / height
            class_idx = category_id_to_index[ann["category_id"]]
            lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (lbl_out_dir / f"{Path(file_name).stem}.txt").write_text("\n".join(lines))

    print(f"[{split}] converted {len(images_by_id)} images")


def main() -> None:
    with open(RAW_DIR / "annotations" / ANNOTATION_FILES["train"]) as f:
        categories = sorted(json.load(f)["categories"], key=lambda c: c["id"])
    category_id_to_index = {c["id"]: i for i, c in enumerate(categories)}
    class_names = [c["name"] for c in categories]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "classes.txt").write_text("\n".join(class_names))
    print("classes:", class_names)

    for split in SPLITS:
        convert_split(split, category_id_to_index)

    yaml.dump(
        {
            "path": str(OUT_DIR.resolve()),
            "train": "images/train",
            "val": "images/easy",   # see note below
            "test": "images/hard",
            "names": {i: name for i, name in enumerate(class_names)},
        },
        open(OUT_DIR / "pidray_data.yaml", "w"),
        sort_keys=False,
    )


if __name__ == "__main__":
    main()