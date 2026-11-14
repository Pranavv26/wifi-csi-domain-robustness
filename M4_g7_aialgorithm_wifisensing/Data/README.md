# Data: Widar3.0-G6D

The dataset is not stored in this repository because of its size (raw data ~29 GB, caches ~6 GB).

## 1. Download

Widar3.0: https://tns.thss.tsinghua.edu.cn/widar3.0/ (official FTP; we used the Tsinghua Cloud mirror linked from the same page because the FTP was unreachable). Only four CSI archives are needed:

```
CSI_20181130.zip   room 1, users 5, 10–17  (test domains)
CSI_20181204.zip   room 2
CSI_20181209.zip   room 2
CSI_20181211.zip   room 3
```

Unzip them into `Code/src/data/widar3g6d/` so that each date has its own folder (for example `data/widar3g6d/20181130/user5/...`).

## 2. Build the caches

From `Code/src` (about 16 GB RAM needed):

```bash
python utils/createWidar3g6d.py --data data/widar3g6d --mode TRAIN
python utils/createWidar3g6d.py --data data/widar3g6d --mode TEST
```

This creates:

```
data/widar3g6d/widar3-g6_csi_domain_train_cache.pkl   24,482 samples, rooms 2–3, 7 domains
data/widar3g6d/widar3-g6_csi_domain_test_cache.pkl    34,166 samples, room 1, 9 domains
```

Keep these file names; the training and adaptation scripts look for them.

## Preprocessing (as in the DATTA paper)

- 6 gestures: Push&Pull, Sweep, Clap, Slide, Draw-circle, Draw-zigzag
- First antenna, 30 subcarriers of CSI amplitude
- Downsampled from 1,000 Hz to 100 Hz
- Sequences of 120–220 packets kept and zero-padded to 220, giving 30 × 220 samples
- A domain is one room–participant combination; training and test share no domain

## Splits used in our experiments

| Subset | Samples | Use |
|---|---|---|
| Source train | 19,586 | Fit weights |
| Source validation | 4,896 | Checkpoint selection |
| Val_TTA | 3,417 | Tune TTA settings only |
| Test | 30,749 | All reported results |

Val_TTA and Test are a fixed 10/90 split (seed 42) of the 34,166 held-out samples, matching the paper.
