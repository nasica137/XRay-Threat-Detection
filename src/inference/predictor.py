from pathlib import Path

from ultralytics import YOLO

from src.detection.objects import (
    BoundingBox,
    Detection
)



class Predictor:


    def __init__(
        self,
        weights: str
    ):

        self.model = YOLO(weights)



    def predict(
        self,
        image_path: str | Path,
        conf: float = 0.25
    ) -> list[Detection]:


        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            verbose=False
        )


        result = results[0]


        detections = []


        for box in result.boxes:


            x1, y1, x2, y2 = (
                box.xyxy[0]
                .tolist()
            )


            cls_id = int(
                box.cls[0]
            )


            label = (
                result.names[cls_id]
            )


            confidence = float(
                box.conf[0]
            )


            detections.append(

                Detection(

                    label=label,

                    confidence=confidence,

                    bbox=BoundingBox(
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )
            )


        return detections



    def annotated_image(
        self,
        image_path: str | Path,
        conf: float = 0.25
    ):


        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            verbose=False
        )


        return results[0].plot()