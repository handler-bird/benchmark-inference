import yaml
from datetime import datetime
import os
import shutil


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

    date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    run_path = f"{task_path}/run{config['run']}_{config['comment']}_{date}"
    os.makedirs(run_path)
    shutil.copy(file, run_path)

    return config, run_path
