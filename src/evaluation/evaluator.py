from pathlib import Path

from ultralytics import YOLO



EVAL_SPLITS = [
    "easy",
    "hard",
    "hidden"
]



class Evaluator:


    def __init__(
        self,
        weights,
        data_dir,
        conf=0.25,
        imgsz=640,
        device=0
    ):

        self.model = YOLO(
            weights
        )

        self.data_dir = Path(
            data_dir
        )

        self.conf = conf
        self.imgsz = imgsz
        self.device = device



    def run(self):

        results = {}


        for split in EVAL_SPLITS:

            print(
                f"\nEvaluating {split}"
            )


            metrics = self.model.val(

                data=str(
                    self.data_dir /
                    f"pidray_{split}_eval.yaml"
                ),

                conf=self.conf,

                imgsz=self.imgsz,

                device=self.device
            )


            results[split] = {

                "precision":
                    float(metrics.box.mp),

                "recall":
                    float(metrics.box.mr),

                "mAP50":
                    float(metrics.box.map50),

                "mAP50-95":
                    float(metrics.box.map)

            }


        return results