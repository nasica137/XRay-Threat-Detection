from src.config.config_loader import TrainConfig
from src.models.factory import build_model
from src.tracking.wandb_tracker import WandBTracker
from src.utils.logger import get_logger


logger = get_logger(__name__)


class Trainer:

    def __init__(
        self,
        config: TrainConfig,
        tracker=None
    ):

        self.config = config

        self.tracker = tracker or WandBTracker(
            entity=config.wandb_entity
        )


    def run(self):

        self.tracker.start(
            project=self.config.wandb_project,
            name=self.config.name,
            config=self.config.__dict__
        )


        logger.info(
            "Starting YOLO training"
        )

        logger.info(
            f"Model: {self.config.model}"
        )

        logger.info(
            f"Dataset: {self.config.data}"
        )


        model = build_model(
            self.config.model
        )


        results = model.train(

            data=self.config.data,

            epochs=self.config.epochs,

            batch=self.config.batch,

            imgsz=self.config.imgsz,

            project=self.config.project,

            name=self.config.name,

            device=self.config.device,
        )


        logger.info(
            "Training finished"
        )


        self.tracker.finish()


        return results