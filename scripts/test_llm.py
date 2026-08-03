from src.detection.objects import (
    Detection,
    BoundingBox
)

from src.llm.report_generator import (
    generate_report
)


detections = [

    Detection(
        label="knife",
        confidence=0.94,
        bbox=BoundingBox(
            10,
            20,
            100,
            200
        )
    )

]


report = generate_report(
    detections
)


print(report)