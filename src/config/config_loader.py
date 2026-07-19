from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


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

    wandb_project: str
    wandb_entity: Optional[str] = None


    @classmethod
    def from_yaml(
        cls,
        path: str
    ):

        with open(path) as f:
            raw = yaml.safe_load(f)


        wandb_cfg = raw.pop(
            "wandb",
            {}
        )


        return cls(
            **raw,
            wandb_project=wandb_cfg.get(
                "project",
                "xray-insight"
            ),
            wandb_entity=wandb_cfg.get(
                "entity"
            )
        )