import yaml
import os
import shutil
import pandas as pd


def load_config(task: str):
    """
    Looks for the given yaml and loads it

    :param task: the task e.g. generative

    :return: a config file with the parameters and the path to the folder
    """

    file = f"./config/{task}.yaml"
    with open(file, "r") as f:
        config = yaml.safe_load(f)

    dir_path = f"./src/data/"
    task_path = dir_path + task

    if not os.path.exists(task_path):
        os.makedirs(task_path)

    run_path = f"{task_path}/run{config['run']}_{config['comment']}"

    if not os.path.exists(run_path):
        os.makedirs(run_path)
        shutil.copy(file, run_path)

    return config, run_path


def save_metrics(
    save_dir: str,
    model: str,
    prompt: str,
    output: str,
    time_to_first_token: float,
    time_per_token: float,
    total_time: float,
    gpu_min_memory: float,
    gpu_peak_memory: float,
    gpu_min_utilization: float,
    gpu_peak_utilization: float
):
    if os.path.exists(save_dir):
        df = pd.read_csv(save_dir)

        df.loc[len(df)] = [
            model,
            prompt,
            output,
            time_to_first_token,
            time_per_token,
            total_time,
            gpu_min_memory,
            gpu_peak_memory,
            gpu_min_utilization,
            gpu_peak_utilization
        ]

        if len(df) == 5:
            mean_values = df[
                ["time_to_first_token", "time_per_token", "total_time", "gpu_min_memory", "gpu_peak_memory", "gpu_min_utilization", "gpu_peak_utilization"]
            ].mean()
            std_values = df[
                ["time_to_first_token", "time_per_token", "total_time", "gpu_min_memory", "gpu_peak_memory", "gpu_min_utilization", "gpu_peak_utilization"]
            ].std()

            df.loc[len(df)] = [
                "MEAN",
                "",
                "",
                mean_values["time_to_first_token"],
                mean_values["time_per_token"],
                mean_values["total_time"],
                mean_values["gpu_min_memory"],
                mean_values["gpu_peak_memory"],
                mean_values["gpu_min_utilization"],
                mean_values["gpu_peak_utilization"]
            ]
            df.loc[len(df)] = [
                "STD",
                "",
                "",
                std_values["time_to_first_token"],
                std_values["time_per_token"],
                std_values["total_time"],
                std_values["gpu_min_memory"],
                std_values["gpu_peak_memory"],
                std_values["gpu_min_utilization"],
                std_values["gpu_peak_utilization"]
            ]

        df.to_csv(save_dir, index=False)

    else:
        df = pd.DataFrame(
            columns=[
                "model",
                "prompt",
                "output",
                "time_to_first_token",
                "time_per_token",
                "total_time",
                "gpu_min_memory",
                "gpu_peak_memory",
                "gpu_min_utilization",
                "gpu_peak_utilization"
            ]
        )
        df.loc[0] = [
            model,
            prompt,
            output,
            time_to_first_token,
            time_per_token,
            total_time,
            gpu_min_memory,
            gpu_peak_memory,
            gpu_min_utilization,
            gpu_peak_utilization
        ]
        df.to_csv(save_dir, index=False)
