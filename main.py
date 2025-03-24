import argparse
import torch
from src.inference import inference_streaming
from src.utils.functions import load_config


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config, base_dir = load_config(task=args.task)

    inference_streaming(
        strategy=config["strategy"],
        model=config["model"],
        prompt=config["prompt"],
        device=device,
        base_dir=base_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark inference")

    subparsers = parser.add_subparsers(
        dest="task", required=True, help="run specific task"
    )

    standard = subparsers.add_parser("inference", help="meassure inference speed")

    args = parser.parse_args()

    main(args)
