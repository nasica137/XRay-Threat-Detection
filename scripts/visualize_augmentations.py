"""
Visualize Albumentations applied to YOLO-format PIDray samples.

Reads:
    data/samples/images/*.png
    data/samples/labels/*.txt

Writes:
    data/samples/augmented/*.png

Purpose:
    Verify that augmentations preserve bounding box alignment.
"""

from pathlib import Path

import cv2
import albumentations as A


IMAGE_DIR = Path("data/samples/images")
LABEL_DIR = Path("data/samples/labels")
OUTPUT_DIR = Path("data/samples/augmented")


def load_yolo_labels(label_path):
    """
    YOLO format:
    class_id cx cy width height

    Albumentations needs:
    x_min y_min x_max y_max
    """

    boxes = []
    class_ids = []

    if not label_path.exists():
        return boxes, class_ids

    with open(label_path) as f:
        for line in f:
            if not line.strip():
                continue

            cls, cx, cy, w, h = map(float, line.split())

            x_min = cx - w / 2
            y_min = cy - h / 2
            x_max = cx + w / 2
            y_max = cy + h / 2

            boxes.append(
                [
                    x_min,
                    y_min,
                    x_max,
                    y_max
                ]
            )

            class_ids.append(int(cls))

    return boxes, class_ids


def save_yolo_labels(path, boxes, class_ids):

    with open(path, "w") as f:

        for box, cls in zip(boxes, class_ids):

            x_min, y_min, x_max, y_max = box

            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2

            w = x_max - x_min
            h = y_max - y_min

            f.write(
                f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
            )


def draw_boxes(image, boxes, class_ids):

    h, w = image.shape[:2]

    for box, cls in zip(boxes, class_ids):

        x_min, y_min, x_max, y_max = box

        x1 = int(x_min * w)
        y1 = int(y_min * h)

        x2 = int(x_max * w)
        y2 = int(y_max * h)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"class {cls}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return image


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # These are reasonable for X-ray images
    transform = A.Compose(
        [
            A.HorizontalFlip(
                p=0.5
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),

            A.GaussNoise(
                p=0.3
            ),

            A.Rotate(
                limit=5,
                p=0.5
            ),

            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=0.05,
                p=0.5
            ),
        ],
        bbox_params=A.BboxParams(
            format="albumentations",
            label_fields=[
                "class_ids"
            ],
            min_visibility=0.3
        )
    )


    images = sorted(
        IMAGE_DIR.glob("*.png")
    )


    for image_path in images:

        image = cv2.imread(
            str(image_path)
        )

        label_path = (
            LABEL_DIR /
            f"{image_path.stem}.txt"
        )

        boxes, class_ids = load_yolo_labels(
            label_path
        )


        augmented = transform(
            image=image,
            bboxes=boxes,
            class_ids=class_ids
        )


        aug_image = augmented["image"]
        aug_boxes = augmented["bboxes"]
        aug_classes = augmented["class_ids"]


        annotated = draw_boxes(
            aug_image,
            aug_boxes,
            aug_classes
        )


        output_path = (
            OUTPUT_DIR /
            f"{image_path.stem}_aug.png"
        )


        cv2.imwrite(
            str(output_path),
            annotated
        )


        print(
            "saved",
            output_path
        )


if __name__ == "__main__":
    main()