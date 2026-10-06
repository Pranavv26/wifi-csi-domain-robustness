# Data: Widar3.0-G6D

Benchmark used by DATTA (Strohmayer et al., WACV 2026), built from raw Widar3.0 CSI.

## Source
- Widar3.0 (Tsinghua University): https://tns.thss.tsinghua.edu.cn/widar3.0/
- The FTP link in the DATTA README is down (connection refused), so data was taken from the official Tsinghua Cloud mirror listed on the Widar3.0 homepage, folder CSI/.
- Files used: 20181130 (3 parts: user5_10_11, user12_13_14, user15_16_17), 20181204, 20181209, 20181211 (~29 GB zipped).

## Preprocessing
Script: Code/src/createWidar3g6d_lowmem.py (low-memory version of DATTA's utils/createWidar3g6d.py; same logic, padding into a pre-allocated array instead of Python lists).
- 6 gestures: push-pull, sweep, clap, slide, draw-circle, draw-zigzag
- Receivers r1-r6, first antenna, 30 subcarriers
- Downsampled 1000 Hz -> 100 Hz, per-sample max-abs normalisation
- Length filter 120-220, zero-padded to T_MAX = 220
- Run on Google Colab (raw data ~33 GB unzipped)

Note: the 20181130 parts unzip as user folders; they were moved into a 20181130/ folder so the script assigns them to room 1 (TEST).

## Splits (match the paper exactly)
| Split | Rooms | Users | Domains | Samples |
|---|---|---|---|---|
| TRAIN | 2, 3 | 1, 2, 3, 6, 7, 8, 9 | 9-15 | 24,482 |
| TEST | 1 | 5, 10-17 | 0-8 | 34,166 |

Full output in logs/. The generated .pkl files (1.3 GB + 1.7 GB) are not committed; they are stored on the team Google Drive.
