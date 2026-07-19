import wandb

from src.tracking.tracker import ExperimentTracker


class WandBTracker(ExperimentTracker):

    def __init__(self, entity=None):
        self.entity = entity


    def start(
        self,
        project: str,
        name: str,
        config: dict
    ):

        wandb.init(
            project=project,
            entity=self.entity,
            name=name,
            config=config
        )


    def log_metrics(
        self,
        metrics: dict,
        step=None
    ):

        wandb.log(
            metrics,
            step=step
        )


    def finish(self):

        wandb.finish()