from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml



def _load_wandb_section(raw):

    wandb_cfg = raw.pop(
        "wandb",
        {}
    )

    return {
        "wandb_project": wandb_cfg.get(
            "project",
            "xray-threat-detection"
        ),
        "wandb_entity": wandb_cfg.get(
            "entity"
        )
    }



@dataclass
class TrainConfig:

    data: str
    model: str

    epochs: int
    batch: int
    imgsz: int

    project: str
    name: str

    device: str

    workers: int = 2
    cache: bool = False
    patience: int = 10

    wandb_project: str = "xray-threat-detection"
    wandb_entity: Optional[str] = None



    @classmethod
    def from_yaml(
        cls,
        path
    ):

        with open(path) as f:
            raw = yaml.safe_load(f)


        wandb = _load_wandb_section(
            raw
        )


        return cls(
            **raw,
            **wandb
        )




@dataclass
class EvalConfig:

    weights: str

    data_dir: str

    conf: float

    iou: float

    imgsz: int

    device: int

    num_examples: int

    save_dir: str


    wandb_project: str = "xray-threat-detection"

    wandb_entity: Optional[str] = None



    @classmethod
    def from_yaml(
        cls,
        path: str | Path
    ):

        with open(path) as f:
            raw = yaml.safe_load(f)


        wandb = _load_wandb_section(
            raw
        )


        return cls(
            **raw,
            **wandb
        )