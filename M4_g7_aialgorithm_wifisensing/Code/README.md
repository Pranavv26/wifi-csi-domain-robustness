# Code (M4)

Source code for the Group 7 study of DATTA (domain-adversarial training, test-time adaptation and random weight resetting) on Widar3.0-G6D with WiFlexFormer. All scripts are in `src/`; run them from inside `src/`.

## Files

- `train_eval.py`: trains and evaluates the `W`, `W+aug` and `W_DAT` configurations (several seeds; resumes from `last.pt`).
- `convert_ckpt.py`: converts a trained checkpoint into the full-model file that the adaptation script loads.
- `DATTA_local.py`: test-time adaptation (TTA) and random weight resetting; evaluates on Test and Val_TTA.
- `eval_perdomain.py`: per-user macro-F1 from the saved predictions.
- `timing.py`: inference cost per sample.
- `make_m4_assets.py`: builds the M4 tables and figures from the result files.
- `datasets.py`, `models/`, `aug/default.yaml`, `utils/`: dataset class, WiFlexFormer with the DAT head, augmentation settings, adaptation helpers, metrics and the Widar3.0-G6D preprocessing script.
- `requirements.txt`: Python packages.

## Run

```bash
pip install -r requirements.txt
# 1. train (example: 30 epochs, W_DAT, seed 3)
python train_eval.py --config W_DAT --epochs 30 --seeds 3 --workers 0 --out results_30ep
# 2. adaptation on a converted checkpoint
python DATTA_local.py --name <run> --workers 0 --eval --momentum_mvg 0.9 --lr 0.0001 --bs 1 --reset_probability 0.0001
```

Use `--reset_probability 0` for TTA without random resetting. The dataset cache (`data/widar3g6d/`) is created from the Widar3.0 data with `utils/createWidar3g6d.py`; see the `Data` folder README. Training settings: AdamW, learning rate 5e-5, weight decay 1e-3, batch size 8, cosine warm restarts, CPU only.
