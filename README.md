# benchmark-inference

## Installation
Run the following command to install all required packages and create a Python environment called `benchmark-inference`:
```sh
conda env create -f environment.yml
```
Activate your environment with:
```sh
conda activate benchmark-inference
```

## Troubleshooting
If the creation of the environment fails, follow these steps:
1. Remove the package `flash-attn` from the `environment.yml` file.
2. Remove the created environment:
    ```sh
    conda env remove -n benchmark-inference
    ```
3. Redo the installation steps:
    ```sh
    conda env create -f environment.yml
    conda activate benchmark-inference
    ```
4. After activating the environment, install `flash-attn` using pip:
    ```sh
    pip install flash-attn
    ```
