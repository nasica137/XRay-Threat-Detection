from dataclasses import dataclass


@dataclass
class BoundingBox:

    x1: float
    y1: float
    x2: float
    y2: float



@dataclass
class Detection:

    label: str
    confidence: float
    bbox: BoundingBox