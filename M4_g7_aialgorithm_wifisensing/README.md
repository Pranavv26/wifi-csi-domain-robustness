# M4: Final Evaluation of DATTA on Widar3.0-G6D

Group 7 · ECE 310 Wireless Communications · AI Algorithms + Benchmark Dataset

Final milestone of our reproduction and component-wise evaluation of DATTA (Strohmayer et al., WACV 2026) for cross-domain Wi-Fi CSI-based human activity recognition. M3 evaluated every component after 8 CPU epochs over three seeds. M4 tests the open question from M3: **does longer training close the gap to the published results?**

## Structure

```
M4_g7_aialgorithm_wifisensing/
├── README.md
├── Report/   M4_g7_aialgorithm_wifisensing_Report.pdf   final report
├── Video/    M4_g7_aialgorithm_wifisensing_video.mp4    20-minute walkthrough
├── Code/     src/ (all scripts) and README.md (how to run)
├── Data/     README.md (download and preprocessing instructions)
└── Results/  result tables (CSV/JSON), raw adaptation outputs and figures
```

## Setup

- Data: Widar3.0-G6D (6 gestures; train rooms 2–3, 7 domains; test room 1, 9 unseen users)
- Split: source train 19,586 / val 4,896; held-out room 1 split into Val_TTA 3,417 (TTA tuning only) and Test 30,749 (all reported numbers)
- Training: AdamW, learning rate 5e-5, weight decay 1e-3, batch size 8, cosine warm restarts, CPU only
- Adaptation: TTA momentum 0.9, learning rate 1e-4, batch size 1; DATTA adds random weight resetting with p = 1e-4
- Metric: macro-F1 (2PR/(P+R) of macro precision and recall)

## Installation and running

```bash
cd Code/src
pip install -r requirements.txt
# build the data caches first (see Data/README.md)
python train_eval.py --config W_DAT --epochs 30 --seeds 2 --workers 0 --out results_30ep
python DATTA_local.py --name <run> --workers 0 --eval --momentum_mvg 0.9 --lr 0.0001 --bs 1 --reset_probability 0.0001
```

Use `--config W`, `W+aug` or `W_DAT`, and `--reset_probability 0` for TTA without resetting. Full details are in `Code/README.md`.

## Key results (Test macro-F1)

| Configuration | 8 epochs, seeds 2–3 | 30 epochs, seeds 2–3 |
|---|---|---|
| W | 47.01 | 45.56 |
| W + aug | 48.25 | 47.68 |
| W_DAT + aug | 46.23 | 46.37 |

| 30-epoch W_DAT | No adaptation | TTA | DATTA |
|---|---|---|---|
| Seed 2 | 45.47 | 35.78 | 36.89 |
| Seed 3 | 47.28 | 45.41 | 47.27 |

Published values [1]: W 40.62, W + aug 49.32, W_DAT + aug 65.66, DATTA 68.13.

## Findings

- Longer training did not help: Test F1 stayed at or below the 8-epoch value although source-validation F1 kept rising.
- W + augmentation remains our best configuration and reproduces the paper's value.
- DAT improved slightly relative to W at 30 epochs but stayed below W + augmentation.
- TTA lowered F1 on the long Test stream (one seed collapsed to 35.8); random resetting recovered part of the loss in every comparison.
- The paper's DAT/DATTA gains (65–68 F1) were not reproduced. We make no state-of-the-art claim. Full analysis, SOTA positioning, limitations and future work are in the report.

## Team

| Member | Main contribution |
|---|---|
| Pranav Jain (AU2420166) | Pipeline integration, TTA code fixes, seed-1 training (8 and 15 epochs), TTA tuning and all adaptation runs, result aggregation and figures |
| Paarth Jawaharani (AU2420056) | Widar3.0-G6D data pipeline (M2), seed-2 training (8 and 30 epochs), M3 evaluation and report writing |
| Kritika Lunkad (AU2320067) | Seed-3 training (8 and 30 epochs), independent TTA check on the 30-epoch checkpoint |
