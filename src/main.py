import argparse

from src.config.config_loader import (
    TrainConfig,
    EvalConfig
)

from src.training.trainer import Trainer

from src.evaluation.evaluator import Evaluator

from src.evaluation.qualitative import QualitativeEvaluator



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



    eval_parser = subparsers.add_parser(
        "evaluate"
    )


    eval_parser.add_argument(
        "--config",
        required=True
    )



    args = parser.parse_args()



    if args.command == "train":


        config = TrainConfig.from_yaml(
            args.config
        )


        trainer = Trainer(
            config
        )


        trainer.run()



    elif args.command == "evaluate":


        config = EvalConfig.from_yaml(
            args.config
        )



        evaluator = Evaluator(

            weights=config.weights,

            data_dir=config.data_dir,

            conf=config.conf,

            imgsz=config.imgsz,

            device=config.device

        )


        metrics = evaluator.run()



        print(
            "\n===== RESULTS ====="
        )


        for split, values in metrics.items():

            print(
                split.upper()
            )

            for k,v in values.items():

                print(
                    f"{k}: {v:.4f}"
                )



        qualitative = QualitativeEvaluator(

            weights=config.weights,

            data_dir=config.data_dir,

            output_dir=config.save_dir,

            metrics=metrics,

            conf=config.conf,

            imgsz=config.imgsz,

            device=config.device

        )


        qualitative.generate_overview(
            samples_per_category=config.num_examples
        )


        qualitative.generate_failure_cases(
            samples=10
        )




if __name__ == "__main__":
    main()