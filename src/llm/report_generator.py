import json
import requests

from src.detection.objects import Detection
from src.llm.prompts import SYSTEM_PROMPT



OLLAMA_URL = (
    "http://localhost:11434/api/chat"
)



def detections_to_json(
    detections: list[Detection]
):


    objects = []


    for d in detections:

        objects.append(

            {
                "object": d.label,
                "confidence": round(
                    d.confidence,
                    3
                )
            }

        )


    return json.dumps(
        objects,
        indent=2
    )




def generate_report(
    detections: list[Detection],
    model: str = "llama3.2"
):


    detection_json = (
        detections_to_json(
            detections
        )
    )


    payload = {


        "model": model,


        "messages":

        [

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },


            {

                "role": "user",

                "content":
                f"""
                Detected objects:

                {detection_json}

                Generate the security report.
                """

            }

        ],


        "stream": False

    }



    response = requests.post(

        OLLAMA_URL,

        json=payload,

        timeout=60

    )


    response.raise_for_status()


    return (

        response
        .json()
        ["message"]
        ["content"]

    )