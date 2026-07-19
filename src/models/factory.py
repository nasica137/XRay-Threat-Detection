from ultralytics import YOLO


def build_model(model_name: str) -> YOLO:
    """
    Creates a YOLO model from:
    - pretrained checkpoint: yolo11n.pt
    - architecture yaml: yolo11n.yaml
    """

    return YOLO(model_name)