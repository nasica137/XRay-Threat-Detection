from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt

from ultralytics import YOLO



EVAL_SPLITS = [
    "easy",
    "hard",
    "hidden"
]


CLASS_NAMES = {
    0: "Baton",
    1: "Pliers",
    2: "Hammer",
    3: "Powerbank",
    4: "Scissors",
    5: "Wrench",
    6: "Gun",
    7: "Bullet",
    8: "Sprayer",
    9: "HandCuffs",
    10: "Knife",
    11: "Lighter"
}



class QualitativeEvaluator:


    def __init__(
        self,
        weights,
        data_dir,
        output_dir="outputs/evaluation",
        metrics=None,
        conf=0.25,
        imgsz=640,
        device=0,
        seed=42
    ):

        self.model = YOLO(weights)

        self.data_dir = Path(data_dir)

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        self.metrics = metrics

        self.conf = conf
        self.imgsz = imgsz
        self.device = device


        random.seed(seed)



    # -------------------------
    # Ground truth
    # -------------------------

    def load_gt_classes(
        self,
        image,
        split
    ):

        label = (
            self.data_dir /
            "labels" /
            split /
            f"{image.stem}.txt"
        )


        classes=[]


        if not label.exists():
            return classes



        with open(label) as f:

            for line in f:

                classes.append(
                    int(line.split()[0])
                )


        return classes




    # -------------------------
    # Prediction
    # -------------------------

    def predict(
        self,
        image
    ):

        result = self.model.predict(
            source=str(image),
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )[0]


        classes=[]


        for box in result.boxes:

            classes.append(
                int(box.cls[0])
            )


        return result, classes




    # -------------------------
    # Case classification
    # -------------------------

    def classify_case(
        self,
        gt,
        pred
    ):

        gt=set(gt)
        pred=set(pred)


        if len(gt)>0 and len(pred)==0:
            return "MISSED"


        if len(gt)==0 and len(pred)>0:
            return "FALSE_POSITIVE"


        if gt==pred:
            return "GOOD"


        if gt.intersection(pred):
            return "PARTIAL"


        return "MISSED"




    # -------------------------
    # Draw GT
    # -------------------------

    def draw_ground_truth(
        self,
        image,
        split
    ):


        img=cv2.imread(
            str(image)
        )


        h,w=img.shape[:2]


        label=(
            self.data_dir /
            "labels" /
            split /
            f"{image.stem}.txt"
        )


        if label.exists():

            with open(label) as f:

                for line in f:


                    cls,xc,yc,bw,bh = map(
                        float,
                        line.split()
                    )


                    x1=int(
                        (xc-bw/2)*w
                    )

                    y1=int(
                        (yc-bh/2)*h
                    )

                    x2=int(
                        (xc+bw/2)*w
                    )

                    y2=int(
                        (yc+bh/2)*h
                    )


                    cv2.rectangle(
                        img,
                        (x1,y1),
                        (x2,y2),
                        (0,255,0),
                        2
                    )


                    cv2.putText(
                        img,
                        CLASS_NAMES[int(cls)],
                        (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .5,
                        (0,255,0),
                        2
                    )


        return cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )




    # -------------------------
    # Collect examples
    # -------------------------

    def collect_examples(
        self,
        samples=2
    ):

        categories=[
            "GOOD",
            "PARTIAL",
            "MISSED",
            "FALSE_POSITIVE"
        ]


        output=[]


        for split in EVAL_SPLITS:


            images=list(
                (
                    self.data_dir /
                    "images" /
                    split
                ).glob("*.png")
            )


            random.shuffle(images)


            found={
                c:[]
                for c in categories
            }


            for image in images[:500]:

                gt=self.load_gt_classes(
                    image,
                    split
                )

                _,pred=self.predict(
                    image
                )


                case=self.classify_case(
                    gt,
                    pred
                )


                found[case].append(
                    image
                )



            for c in categories:

                chosen=random.sample(
                    found[c],
                    min(
                        samples,
                        len(found[c])
                    )
                )


                for image in chosen:

                    output.append(
                        (
                            split,
                            c,
                            image
                        )
                    )


        return output




    # -------------------------
    # Overview figure
    # -------------------------

    def generate_overview(
        self,
        samples_per_category=2
    ):


        examples=self.collect_examples(
            samples_per_category
        )


        fig,axes=plt.subplots(
            len(examples),
            3,
            figsize=(16,5*len(examples))
        )


        if len(examples)==1:
            axes=[axes]



        for i,(split,case,image) in enumerate(examples):


            original=cv2.imread(
                str(image)
            )

            original=cv2.cvtColor(
                original,
                cv2.COLOR_BGR2RGB
            )


            gt=self.draw_ground_truth(
                image,
                split
            )


            result,_=self.predict(
                image
            )


            pred=result.plot()

            pred=cv2.cvtColor(
                pred,
                cv2.COLOR_BGR2RGB
            )


            score=self.metrics[split]["mAP50"]



            axes[i][0].imshow(
                original
            )

            axes[i][0].set_title(
                f"{split.upper()} | {case}\nOriginal"
            )


            axes[i][1].imshow(
                gt
            )

            axes[i][1].set_title(
                "Ground Truth"
            )


            axes[i][2].imshow(
                pred
            )

            axes[i][2].set_title(
                f"Prediction\nmAP50={score:.3f}"
            )



            for ax in axes[i]:
                ax.axis("off")



        plt.tight_layout()


        path=(
            self.output_dir /
            "qualitative_overview.png"
        )


        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )


        plt.close()


        print(
            f"Saved {path}"
        )



    # -------------------------
    # Failure cases
    # -------------------------

    def generate_failure_cases(
        self,
        samples=10
    ):


        failures=[]


        for split in EVAL_SPLITS:


            images=list(
                (
                    self.data_dir /
                    "images" /
                    split
                ).glob("*.png")
            )


            for image in images[:500]:


                gt=self.load_gt_classes(
                    image,
                    split
                )


                _,pred=self.predict(
                    image
                )


                case=self.classify_case(
                    gt,
                    pred
                )


                if case in [
                    "MISSED",
                    "PARTIAL"
                ]:

                    failures.append(
                        (
                            split,
                            image,
                            case
                        )
                    )



        failures=random.sample(
            failures,
            min(
                samples,
                len(failures)
            )
        )


        fig,axes=plt.subplots(
            len(failures),
            3,
            figsize=(16,5*len(failures))
        )


        if len(failures)==1:
            axes=[axes]



        for i,(split,image,case) in enumerate(failures):


            original=cv2.imread(
                str(image)
            )

            original=cv2.cvtColor(
                original,
                cv2.COLOR_BGR2RGB
            )


            gt=self.draw_ground_truth(
                image,
                split
            )


            result,_=self.predict(
                image
            )


            pred=result.plot()

            pred=cv2.cvtColor(
                pred,
                cv2.COLOR_BGR2RGB
            )


            axes[i][0].imshow(original)
            axes[i][0].set_title(
                f"{split.upper()} {case}"
            )


            axes[i][1].imshow(gt)
            axes[i][1].set_title(
                "Ground Truth"
            )


            axes[i][2].imshow(pred)
            axes[i][2].set_title(
                "Prediction"
            )


            for ax in axes[i]:
                ax.axis("off")



        plt.tight_layout()


        path=(
            self.output_dir /
            "failure_cases.png"
        )


        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )


        plt.close()


        print(
            f"Saved {path}"
        )