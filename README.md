# SIDE: Surgical Instrument Depth & Entity Segmentation

Repository for the Masterthesis of Léon Zeitler with Lennart Maack as a supervisor.

## Project Setup

### Prerequisites

* Git
* Python
* NVIDIA GPU

### Installation

1. `git clone https://github.com/ZeitDev/SIDE`
2. `pip install uv`
3. `uv sync`

## Usage

### Training

1. Set parameters in /configs/experiment_name.yaml
2. For training and testing enter: `uv run main.py` and select config

### View Results

1. `source .venv/bin/activate`
2. `mlflow ui`
3. open mlflow server website

## Features

* **Tasks:** Surgical Instrument Segmentation and Full Scene Disparity Estimation (not yet implemented)
* **Metrics:** IoU + DICE (mean and per instrument class) and MAE + RMSE (not yet implemented)
* **Cross Validation or Full Training:** Leave one sequence out cross validation or train on all sequences for inference (define sequences in /dataset/train/sequence_* or with a custom dataset class)
* **Custom Datasets:** Define custom datasets, paths, sequences in ./data/datasets.py
* **MLflow Logging:** Full experiment tracking, including full repository snapshot, metrics, cross validation summary, best validation or full training model state, segmentation mask validation images, learning rate, loss and all config parameters (extractable as pandas dataframes)
* **Multi-Task Learning:** Training of a shared-feature encoder + seperate task decoders (not yet tested)
* **Multi-Teacher Knowledge Distillation:** Multiple teacher per task for knowledge distillation (not yet verified and tested)
* **Config:** Comprehensible config with modular encoder-decoder, optimizer, loss function, multi-task, multi-teacher, finetuning, transforms, logging settings
* **Evaluation:** Notebooks for evaluation (not fully implemented)
* **Test Cases:** Test cases for verification of metrics
