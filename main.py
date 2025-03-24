import argparse
import torch
from src.standard_inference import standard_streaming
from src.utils.functions import load_config


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config, base_dir = load_config(task=args.task)

    standard_streaming(
        prompt=config["prompt"], model=config["model"], device=device, base_dir=base_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark inference")

    subparsers = parser.add_subparsers(
        dest="task", required=True, help="run specific task"
    )

    standard = subparsers.add_parser("standard", help="use the standard model")

    args = parser.parse_args()

    main(args)
