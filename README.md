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

## Status
* Training and Testing Pipeline finished
* TODO: Evaluation Notebooks

## Features

* **Tasks:** Surgical Instrument Segmentation and Full Scene Disparity Estimation 
* **Metrics:** IoU, DICE (Segmentation) and Bad3 Rate, EPE Pixel, MAE mm (Disparity)
* **Cross Validation or Full Training:** Leave one sequence out cross validation or train on all sequences for inference (define sequences in /dataset/train/sequence_* or with a custom dataset class)
* **Custom Datasets:** Define custom datasets, paths, sequences in ./data/datasets.py
* **MLflow Logging:** Full experiment tracking (via local Server or extractable as pandas dataframes)
* **Multi-Task Learning:** Training of a shared-feature encoder + seperate task decoders
* **Multi-Teacher Knowledge Distillation:** Teacher per task for knowledge distillation
* **Automatic Weighted Loss:** Uses the homoscedastic uncertainty weighting method by Kendall (2018) for multi-task weight normalization
* **Config:** Comprehensible settings config with modular encoder-decoder, optimizer, loss function, multi-task, multi-teacher, finetuning, transforms, logging, and more settings
* **Evaluation:** Notebooks for evaluation
* **Test Cases:** Test cases for verification of central components `uv run pytest` (tests every file that starts with 'test*')
* **LR Finder:** Learning rate finder by fastai exponential (increasing the LR in an exponential manner)

## MLflow Logging Intervals

* Experiment Run *(e.g. 251125:1636)*
    * Complete config *(saved in Overview/Parameters, propagated to subruns)*
    * Repository Snapshot of all relevant files  *(saved in Artifacts)*
    * *(saved in Tags)* 
        * Description
        * Parent name
        * Run Type 
* Full/CV Training Subrun *(e.g. 251125:1636/train/fold_1)*
    * *(saved in Overview)*
        * Description
        * Parent Name
        * Run Mode
        * Run Type
        * Validation Subset
        * Fold Index
    * *Each Epoch (saved in Model Metrics)*
        * Training
            * Learning Rate
            * Auto Weighted Loss Sum
            * Auto Weights per Task and Teacher
            * Raw Loss per Task and Teacher
        * Validation *(only CV)*
            * Auto Weighted Loss Sum
            * Best Auto Weighted Loss Sum
            * Auto Weights per Task
            * Raw Loss per Task
            * If Segmentation
                * Mean, std, and per class
                * IoU Score
                * DICE Score
                * Overlay Image *(saved in Artifacts)*
            * If Disparity
                * Bad3 Rate
                * EPE Pixel
                * MAE mm
                * Overlay Image *(saved in Artifacts)*
    * *Each Run*
        * Best model determined by lowest validation loss *(CV)* or after all epochs *(Full)* *(saved in Models)*
        * Cross Validation Summary *(saved in Model Metrics)*
            * Best Fold Index
            * Best Fold Auto Weighted Loss Sum
            * Comparison per Fold, mean, std 
                * Auto Weighted Loss Sum
                * Auto Weights per Task
                * Raw Loss per Task
                * If Segmentation
                    * IoU Score
                    * DICE Score
                * If Disparity
                    * Bad3 Rate
                    * EPE Pixel
                    * MAE mm
* Test Subrun *(e.g. 211125:1636/test)*
    * Performance Metrics *(saved in Model Metrics)*
        * If Segmentation
            * Mean, std, and per class
                * IoU Score
                * DICE Score
        * If Disparity
            * Bad3 Rate
            * EPE Pixel
            * MAE mm

## Notes

### Kullback-Leibler Divergence Loss for Knowledge Distillation
* Logits are divided by a temperature T to soften the probability distribution
* Final loss is divided by image size to normalize it, as the KL Divergence is a sum over all pixels and classes. This prevents the loss from exploding and allows for better convergence.

### Modular Encoder + Decoder
* Encoder needs to return feature map
* Standard Decoder resembles mostly a U-Net with custom Heads for each task. Can adjust dynamically to corresponding feature maps provided by different encoders and is stereo compatible.

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