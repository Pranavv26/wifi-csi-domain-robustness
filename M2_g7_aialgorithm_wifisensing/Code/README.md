# Code

Built on the official DATTA repo: https://github.com/StrohmayerJ/DATTA

## Files
- src/createWidar3g6d_lowmem.py: builds the Widar3.0-G6D train/test cache files from raw Widar3.0 CSI (low-memory version of DATTA's utils/createWidar3g6d.py).
- src/train_eval.py: trains and tests the baselines with DATTA's model, augmentation, split and metrics.
  - W: WiFlexFormer, activity cross-entropy only
  - W+aug: same, with DATTA's CSI augmentation (aug/default.yaml)
  - W_DAT: domain-adversarial training (DATTA's trainDAT loss)

## How to run (Google Colab, T4 GPU)
1. Build the dataset (see ../Data/README.md):
   python3 utils/createWidar3g6d_lowmem.py --mode TRAIN
   python3 utils/createWidar3g6d_lowmem.py --mode TEST
2. Clone DATTA, put the two .pkl files in data/widar3g6d/ and train_eval.py in the repo root.
3. Install: pip install albumentations==1.4.14 albucore==0.0.14 einops wandb
4. Train and test:
   python train_eval.py --config W --epochs N --seeds 1 2 3 --out results
   python train_eval.py --config W+aug --epochs N --seeds 1 2 3 --out results

## Protocol
- Train/Val: TRAIN split (rooms 2, 3), random 80/20, seed 42
- Test: TEST split (room 1, unseen), shuffled, 90% used (10% kept as DATTA's Val_TTA split)
- Model: checkpoint with lowest validation loss
- Metrics: accuracy, macro precision/recall, F1 = 2PR/(P+R), inference time per sample
- Each config is run with 3 seeds and reported as mean ± std

## Deviation from the paper
DATTA trains for up to 4000 epochs. Due to free Colab GPU limits we use a reduced epoch budget (N, reported in Results/). All other hyperparameters match the paper (batch size 8, lr 5e-5, AdamW, cosine warm restarts).
