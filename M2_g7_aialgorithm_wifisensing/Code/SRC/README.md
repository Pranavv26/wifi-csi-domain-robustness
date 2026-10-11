## Overview

This directory contains the implementation of a plain WiFlexFormer baseline for human activity recognition using Wi-Fi Channel State Information (CSI).

## Files

- **`createWidar3g6d.py`** — Preprocesses the Widar3.0-G6D dataset and generates training and test cache files.
- **`widar_baseline_dataset.py`** — Loads cached data and prepares CSI features and activity labels.
- **`wiflexformer.py`** — Defines the WiFlexFormer model.
- **`train_widar_baseline.py`** — Handles model training, validation, checkpoint selection, and final test evaluation.

## Requirements

- Python
- PyTorch
- NumPy
- `einops`
- `scikit-learn`
- `tqdm`

Install the required dependencies in your Python environment before running the scripts.

## How to use

### 1. Preprocess the dataset

```bash
python createWidar3g6d.py --data "<DATASET_PATH>" --mode TRAIN
python createWidar3g6d.py --data "<DATASET_PATH>" --mode TEST
```

Replace `<DATASET_PATH>` with the path to the extracted Widar3.0-G6D dataset.

### 2. Run the sanity check

```bash
python train_widar_baseline.py --train-cache "<TRAIN_CACHE_PATH>" --test-cache "<TEST_CACHE_PATH>" --sanity-check
```

### 3. Train and evaluate

```bash
python train_widar_baseline.py --train-cache "<TRAIN_CACHE_PATH>" --test-cache "<TEST_CACHE_PATH>"
```

Replace the cache placeholders with the paths to the generated cache files.

## Configuration

The implemented baseline uses 30 CSI subcarriers, a sequence length of 220, and six activity classes. Training uses an 80:20 stratified split of the training cache for training and validation. The separate test cache is reserved for final evaluation.

## Outputs

Training produces a model checkpoint selected using validation macro-F1, final test metrics, and a run configuration file under `outputs/wiflexformer_widar_baseline/`.

**Note:** The raw dataset and generated cache files are not included in this directory and must be prepared separately.
