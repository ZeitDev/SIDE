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

### Test Cases

Run `pytest` in environment.

### Training

1. Set parameters in /configs/experiment_name.yaml
2. For training and testing enter: `uv run main.py` and select config

### View Results

1. `source .venv/bin/activate`
2. `mlflow ui`
3. open mlflow server website (shown in terminal, usually `127.0.0.1:5000`)

## Ongoing
* Implementing disparity task
* Dataset evaluation

## Features

* **Tasks:** Surgical Instrument Segmentation and Full Scene Disparity Estimation (not yet implemented)
* **Metrics:** IoU + DICE (mean and per instrument class) and MAE + RMSE (not yet implemented)
* **Cross Validation or Full Training:** Leave one sequence out cross validation or train on all sequences for inference (define sequences in /dataset/train/sequence_* or with a custom dataset class)
* **Custom Datasets:** Define custom datasets, paths, sequences in ./data/datasets.py
* **MLflow Logging:** Full experiment tracking, including full repository snapshot, metrics, cross validation summary, best validation or full training model state, segmentation mask validation images, learning rate, loss and all config parameters (extractable as pandas dataframes)
* **Multi-Task Learning:** Training of a shared-feature encoder + seperate task decoders (not yet tested)
* **Automatic Weighted Loss:** Uses the homoscedastic uncertainty weighting method by Kendall (2018) for multi-task weight normalization
* **Multi-Teacher Knowledge Distillation:** Multiple teacher per task for knowledge distillation (not yet verified and tested)
* **Config:** Comprehensible settings config with modular encoder-decoder, optimizer, loss function, multi-task, multi-teacher, finetuning, transforms, logging
* **Evaluation:** Notebooks for evaluation (not fully implemented)
* **Test Cases:** Test cases for verification of central components `uv run pytest` (tests every file that starts with 'test*')
* **LR Finder:** Learning rate finder by fastai exponential (increasing the LR in an exponential manner)

## MLflow Logging Intervals

* Experiment Run *(e.g. 251125:1636)*
    * Complete config *(saved in Parameters, propagated to subruns)*
    * Snapshot of all relevant files in repository *(saved in Artifacts)*
    * Parent name, run type, description *(saved in Tags)*
* Full/CV Training Subrun *(e.g. 251125:1636/train)*
    * Parent name, run mode
    * *Each Epoch (saved in Model Metrics)*
        * Learning Rate
        * Automatic Task Weights
        * Training Weighted Loss 
        * Training Raw Task Loss 
        * Validation Weighted Loss *(only CV)*
        * Validation Raw Loss *(only CV)*
        * Best Validation Loss *(only CV)*
        * Segmentation Overlay Images *(only CV and if set by config)*
    * *Each Run (saved in Models)*
        * Best model determined by lowest validation loss *(CV)* or after all epochs *(Full) [better approach for Full Training?]*
* Test Subrun *(e.g. 211125:1636/test)*
    * Performance Metrics *(saved in Model Metrics)*
        * Mean and per class
            * IoU, DICE

## Notes

### Modular Encoder + Decoder
* Encoder needs to return feature map
* Standard Decoder resembles mostly a U-Net with custom Heads for each task. Can adjust dynamically to corresponding feature maps provided by different encoders.

### Test Cases
* Verification of correct IoU and DICE calculation

### Negative Weighted Loss
* When the raw loss of a task converges to 0.0 the AutomaticWeightedLoss gains confidence (Task Weight Value). To maximize the confidence, it lowers the uncertainty *s* to become negative, such that the task weight *e^-s* goes up. The equation looks like this for example: *e^-s * L_raw + 0.5 * s = 5.0 * 0.0001 + 0.5 * (-2) = -1*.

### Encoder LR Mod
* As the encoder is already pretrained, we lower the corresponding learning rate to a more conversative value to preserve the pretrained knowledge.

### Cosine Annealing with Warmup
* TODO: read paper
* Can escape saddle points. SOTA for transformers, but also applicable to CNNs?

### LRFinder
* Automatic Weighted Loss needs to have frozen uncertainty weights, because when the exponential learning rate explodes the Automatic Weights will fight against it, preventing the loss moutain we want to see at the end of the graph.

* Best learning rate in the middle of the steepest slide down, before the exploding cliff. Because this point indicates "maximum speed" and far away of exploding cliff (divergence). Do not trust the red dot.

### DimitrisPs/ris2017_toolkit
* The toolkit uses INTER_LINEAR to resize masks, that creates classes that did not exist before. Thats fine for them, cause they only generate binary masks. Masks should always be interpolated with INTER_NEAREST.
* The toolkit does not deinterlace the masks like they do with the RGB images, as the masks appear to be clean already. If zoomed on the edge of an instrument, there is a step pattern present. Segmentation on interlaced rgb images would result in comb like structures in the mask.

### DimitrisPs/MSDESIS
* The paper evaluates every image individually, averages them for the batch and saves these in a running average.
* This project differs by accumulating these metrices for the entire dataset and finally averaging them per epoch. 
* This approach ensures metrics are weighted by valid pixel count, preventing outliers from disproportionately skewing the results.