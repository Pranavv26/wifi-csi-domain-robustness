# M2: Baseline Reproduction (Group 7)

Reproducing the baseline of DATTA (Strohmayer et al., WACV 2026) for cross-domain WiFi CSI gesture recognition on Widar3.0-G6D.

Team: Pranav Jain (AU2420166), Paarth Jawaharani (AU2420056), Kritika Lunkad (AU2320067)

## Folders
- Code/: preprocessing and training scripts, run instructions
- Data/: dataset source, preprocessing steps, build logs
- Results/: obtained metrics, comparison with the paper, figures
- Report/: M2 report (PDF)
- Video/: walkthrough video or link

## Summary
| Config | Paper F1 | Ours F1 (3 seeds) |
|---|---|---|
| W (no augmentation) | 40.62 | 44.60 ± 0.62 |
| W + augmentation | 49.32 | 48.53 ± 0.28 |

Dataset rebuilt exactly as in the paper: 24,482 train (rooms 2, 3) and 34,166 test (room 1) samples.
