## Overview

This directory contains the Milestone 3 implementation for Group 7: a controlled evaluation of DATTA (domain-adversarial training, test-time adaptation and random weight resetting) on a WiFlexFormer backbone, using the Widar3.0-G6D benchmark for Wi-Fi CSI-based human activity recognition.

The held-out data is split into Val_TTA (3,417 samples, used only to tune test-time adaptation) and Test (30,749 samples, used for all reported results). Five configurations are evaluated with three seeds each.

## Files

- **`createWidar3g6d_lowmem.py`**: preprocesses the Widar3.0-G6D dataset and generates the training and test cache files.
- **`datasets.py`**: loads the cached data and applies CSI augmentation to the training set.
- **`models/wiflexformer_dat.py`**: WiFlexFormer with the domain discriminator, gradient reversal layer and DAT loss.
- **`train_eval.py`**: trains and evaluates the `W`, `W+aug` and `W_DAT` configurations.
- **`<TTA_SCRIPT>.py`**: test-time adaptation and random weight resetting on a trained `W_DAT` checkpoint.
- **`make_report_assets.py`**: generates the report tables and figures from the result files.
- **`aug/default.yaml`**: augmentation settings.

DAT, the confidence-control loss, TTA and random weight resetting are from the authors' implementation (https://github.com/StrohmayerJ/DATTA).

## Requirements

- Python
- PyTorch
- NumPy
- `einops`
- `scikit-learn`
- `tqdm`
- `pyyaml`
- `albumentations==2.0.8`
- `opencv-python-headless`
- `matplotlib`

Training runs on CPU only.

## How to use

### 1. Preprocess the dataset

```bash
python createWidar3g6d_lowmem.py --data "<DATASET_PATH>" --mode TRAIN
python createWidar3g6d_lowmem.py --data "<DATASET_PATH>" --mode TEST
```

Place the two cache files in `data/widar3g6d/` with their names unchanged:

```
widar3-g6_csi_domain_train_cache.pkl
widar3-g6_csi_domain_test_cache.pkl
```

### 2. Train and evaluate

Run the configurations one after another:

```bash
python train_eval.py --config W     --epochs 8 --seeds 1 2 3 --workers 0 --out results_final
python train_eval.py --config W+aug --epochs 8 --seeds 1 2 3 --workers 0 --out results_final
python train_eval.py --config W_DAT --epochs 8 --seeds 1 2 3 --workers 0 --out results_final
```

### 3. Test-time adaptation and random weight resetting

```bash
python <TTA_SCRIPT>.py <ARGUMENTS>
```

### 4. Generate the report tables and figures

```bash
python make_report_assets.py
```

## Configuration

Training uses 8 epochs, batch size 8, AdamW (learning rate 5e-5, weight decay 1e-3) and cosine annealing with warm restarts, on CPU, with seeds 1, 2 and 3. The model has 30 CSI subcarriers, a sequence length of 220 and six activity classes. The checkpoint with the lowest source-validation loss is used for evaluation.

## Outputs

Each run writes `result.json`, `log.csv` and `best_val_loss.pt` to `results_final/<config>_s<seed>/`. Per-run results, logs and report figures are in `Results/`.

**Note:** The raw dataset and generated cache files are not included in this directory and must be prepared separately.
