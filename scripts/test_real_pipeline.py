from src.inference.predictor import Predictor
from src.llm.report_generator import generate_report


def main():

    predictor = Predictor(
        "runs/detect/pidray_yolo11n_baseline/weights/best.pt"
    )

    detections = predictor.predict(
        "data/samples/images/xray_04521.png"
    )

    print("Detected objects:")
    for d in detections:
        print(
            f"{d.label}: {d.confidence:.2f}"
        )

    print("\nGenerating report...\n")

    report = generate_report(detections)

    print(report)


if __name__ == "__main__":
    main()