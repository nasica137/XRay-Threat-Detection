import argparse

from src.config.config_loader import TrainConfig
from src.training.trainer import Trainer


def main():

    parser = argparse.ArgumentParser(
        description="XRay-Threat-Detection CLI"
    )


    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )


    train_parser = subparsers.add_parser(
        "train"
    )

    train_parser.add_argument(
        "--config",
        required=True
    )


    args = parser.parse_args()


    if args.command == "train":

        config = TrainConfig.from_yaml(
            args.config
        )

        trainer = Trainer(config)

        trainer.run()



if __name__ == "__main__":
    main()