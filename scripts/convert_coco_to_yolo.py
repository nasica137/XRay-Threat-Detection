"""
Convert PIDray COCO annotations into Ultralytics YOLO format.

Creates:

data/processed/pidray/

images/
    train/
    val/
    easy/
    hard/
    hidden/

labels/
    train/
    val/
    easy/
    hard/
    hidden/

The official PIDray test sets (easy/hard/hidden)
are never used during training.
"""

import json
import shutil
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split


RAW_DIR = Path("data/raw/pidray")
OUT_DIR = Path("data/processed/pidray")


def coco_to_yolo_line(annotation, width, height, category_map):

    x, y, w, h = annotation["bbox"]

    cx = (x + w / 2) / width
    cy = (y + h / 2) / height

    nw = w / width
    nh = h / height

    cls = category_map[
        annotation["category_id"]
    ]

    return (
        f"{cls} "
        f"{cx:.6f} "
        f"{cy:.6f} "
        f"{nw:.6f} "
        f"{nh:.6f}"
    )


def process_split(
    json_name,
    image_folder,
    output_split,
    category_map
):

    with open(
        RAW_DIR /
        "annotations" /
        json_name
    ) as f:
        coco = json.load(f)


    images = {
        img["id"]: img
        for img in coco["images"]
    }


    annotations = {}

    for ann in coco["annotations"]:

        annotations.setdefault(
            ann["image_id"],
            []
        ).append(ann)


    img_out = (
        OUT_DIR /
        "images" /
        output_split
    )

    lbl_out = (
        OUT_DIR /
        "labels" /
        output_split
    )


    img_out.mkdir(
        parents=True,
        exist_ok=True
    )

    lbl_out.mkdir(
        parents=True,
        exist_ok=True
    )


    for image_id, image in images.items():

        filename = Path(
            image["file_name"]
        ).name


        source = (
            RAW_DIR /
            image_folder /
            filename
        )

        destination = (
            img_out /
            filename
        )


        if not destination.exists():

            shutil.copy(
                source,
                destination
            )


        labels = []

        for ann in annotations.get(
            image_id,
            []
        ):

            labels.append(
                coco_to_yolo_line(
                    ann,
                    image["width"],
                    image["height"],
                    category_map
                )
            )


        (
            lbl_out /
            f"{Path(filename).stem}.txt"
        ).write_text(
            "\n".join(labels)
        )


    print(
        f"{output_split}: {len(images)} images"
    )



def process_train_validation(category_map):

    with open(
        RAW_DIR /
        "annotations" /
        "xray_train.json"
    ) as f:

        coco = json.load(f)


    train_images = coco["images"]


    train_part, val_part = train_test_split(
        train_images,
        test_size=0.2,
        random_state=42
    )


    image_lookup = {
        img["id"]: img
        for img in train_images
    }


    annotations = {}

    for ann in coco["annotations"]:

        annotations.setdefault(
            ann["image_id"],
            []
        ).append(ann)


    for split_name, images in [
        ("train", train_part),
        ("val", val_part)
    ]:

        img_out = (
            OUT_DIR /
            "images" /
            split_name
        )

        lbl_out = (
            OUT_DIR /
            "labels" /
            split_name
        )


        img_out.mkdir(
            parents=True,
            exist_ok=True
        )

        lbl_out.mkdir(
            parents=True,
            exist_ok=True
        )


        for image in images:

            filename = Path(
                image["file_name"]
            ).name


            shutil.copy(
                RAW_DIR /
                "train" /
                filename,

                img_out /
                filename
            )


            labels = []


            for ann in annotations.get(
                image["id"],
                []
            ):

                labels.append(
                    coco_to_yolo_line(
                        ann,
                        image["width"],
                        image["height"],
                        category_map
                    )
                )


            (
                lbl_out /
                f"{Path(filename).stem}.txt"
            ).write_text(
                "\n".join(labels)
            )


        print(
            f"{split_name}: {len(images)} images"
        )



def main():

    with open(
        RAW_DIR /
        "annotations" /
        "xray_train.json"
    ) as f:

        categories = sorted(
            json.load(f)["categories"],
            key=lambda x: x["id"]
        )


    category_map = {
        c["id"]: idx
        for idx, c in enumerate(categories)
    }


    names = {
        idx: c["name"]
        for idx, c in enumerate(categories)
    }


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    (
        OUT_DIR /
        "classes.txt"
    ).write_text(
        "\n".join(
            names.values()
        )
    )


    # training split
    process_train_validation(
        category_map
    )


    # official test sets
    process_split(
        "xray_test_easy.json",
        "easy",
        "easy",
        category_map
    )

    process_split(
        "xray_test_hard.json",
        "hard",
        "hard",
        category_map
    )

    process_split(
        "xray_test_hidden.json",
        "hidden",
        "hidden",
        category_map
    )


    # training yaml

    yaml.dump(

        {
            "path":
                str(
                    OUT_DIR.resolve()
                ),

            "train":
                "images/train",

            "val":
                "images/val",

            "names":
                names
        },

        open(
            OUT_DIR /
            "pidray_data.yaml",
            "w"
        ),

        sort_keys=False
    )


    # evaluation yamls

    for split in [
        "easy",
        "hard",
        "hidden"
    ]:

        yaml.dump(

            {
                "path": str(OUT_DIR.resolve()),

                # Required by Ultralytics even though it is unused
                "train": "images/train",

                # Evaluation split
                "val": f"images/{split}",

                "names": names,
            },

            open(
                OUT_DIR / f"pidray_{split}_eval.yaml",
                "w"
            ),

            sort_keys=False,
        )


if __name__ == "__main__":
    main()