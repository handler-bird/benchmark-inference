#TODO: Remove the following lines after the issue is fixed
#import os
#os.environ["PYTHONPATH"] = os.getcwd() + ":" + os.environ.get("PYTHONPATH", "")
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Force VLLM to use spawn method
#os.environ["VLLM_USE_SPAWN"] = "1"
# Disable VLLM using multiprocessing if possible
#os.environ["VLLM_WORKERS"] = "1"
# Force PyTorch to use spawn
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

#TODO: Remove when fixed
#import multiprocessing
#multiprocessing.set_start_method('spawn', force=True)

import argparse
import torch
from src.inference import inference_streaming
from src.utils.functions import load_config

#TODO: Remove when fixed
#print(f"Multiprocessing start method: {multiprocessing.get_start_method()}")

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
