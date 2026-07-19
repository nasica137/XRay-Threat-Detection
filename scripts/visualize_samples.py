"""
Visualize YOLO-format PIDray samples.

Reads:
    data/samples/images/*.png
    data/samples/labels/*.txt
    data/samples/classes.txt

Writes:
    data/samples/visualized/*.png

Purpose:
    Sanity-check that YOLO labels match the actual objects
    before training.
"""

from pathlib import Path

import cv2


SAMPLES_DIR = Path("data/samples")
IMAGE_DIR = SAMPLES_DIR / "images"
LABEL_DIR = SAMPLES_DIR / "labels"
OUTPUT_DIR = SAMPLES_DIR / "visualized"
CLASSES_FILE = SAMPLES_DIR / "classes.txt"


def load_classes():
    with open(CLASSES_FILE) as f:
        return [line.strip() for line in f.readlines()]


def yolo_to_xyxy(label, img_width, img_height):
    """
    Convert YOLO format:

        class cx cy w h

    normalized [0,1]

    into OpenCV coordinates:

        x1 y1 x2 y2
    """

    class_id, cx, cy, w, h = map(float, label.split())

    cx *= img_width
    cy *= img_height
    w *= img_width
    h *= img_height

    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    return int(class_id), x1, y1, x2, y2


def draw_boxes(image, label_file, classes):
    height, width = image.shape[:2]

    if not label_file.exists():
        return image

    with open(label_file) as f:
        labels = f.readlines()

    for label in labels:
        if not label.strip():
            continue

        class_id, x1, y1, x2, y2 = yolo_to_xyxy(
            label,
            width,
            height
        )

        class_name = (
            classes[class_id]
            if class_id < len(classes)
            else f"class_{class_id}"
        )

        # draw rectangle
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # label text
        text = class_name

        cv2.putText(
            image,
            text,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return image


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    classes = load_classes()

    images = sorted(
        list(IMAGE_DIR.glob("*.png"))
    )

    if not images:
        raise RuntimeError(
            "No sample images found. "
            "Run create_samples.py first."
        )

    for image_path in images:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(f"Could not read {image_path}")
            continue

        label_path = (
            LABEL_DIR /
            f"{image_path.stem}.txt"
        )

        image = draw_boxes(
            image,
            label_path,
            classes
        )

        output_path = (
            OUTPUT_DIR /
            image_path.name
        )

        cv2.imwrite(
            str(output_path),
            image
        )

        print(
            f"saved {output_path}"
        )


if __name__ == "__main__":
    main()