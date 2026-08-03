from src.inference.predictor import Predictor

from src.llm.report_generator import (
    generate_report
)


WEIGHTS = (
    "runs/detect/"
    "pidray_yolo11n_baseline/"
    "weights/best.pt"
)


IMAGE = (
    "data/samples/images/"
    "xray_00992.png"
)



def main():


    predictor = Predictor(
        weights=WEIGHTS
    )


    detections = predictor.predict(
        IMAGE,
        conf=0.25
    )


    print("\nDetected objects:")
    

    for d in detections:

        print(
            f"{d.label}: "
            f"{d.confidence:.2f}"
        )



    print("\nGenerating report...\n")


    report = generate_report(
        detections
    )


    print(report)



if __name__ == "__main__":
    main()