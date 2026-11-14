# Cross-Domain Wi-Fi CSI-Based Human Activity Recognition via Domain-Adversarial Test-Time Adaptation

ECE 310 Wireless Communications (Monsoon 2026) · Ahmedabad University · Prof. Dhaval Patel
Group 7 · Category: AI Algorithms + Benchmark Dataset

Wi-Fi channel state information (CSI) records how the multipath channel between a transmitter and receiver changes when a person moves. Models trained on CSI in one room, or on one set of people, often fail in a new room or with new people (domain shift). This project is a **reproduction and component-wise evaluation** of **DATTA** (Strohmayer et al., WACV 2026) on the **Widar3.0-G6D** benchmark. It is not a new algorithm.

DATTA combines, on a 41,977-parameter WiFlexFormer backbone:
- **CSI augmentation** (training only)
- **Domain-adversarial training (DAT):** a gradient reversal layer and domain discriminator push the model towards domain-invariant features
- **Test-time adaptation (TTA):** the model adapts online to the unlabelled test stream by aligning feature statistics
- **Random weight resetting:** limits drift and forgetting during adaptation

## Repository structure

```
M1_g7_aialgorithm_wifisensing/   Problem definition, literature review, plan (report, video)
M2_g7_aialgorithm_wifisensing/   Data pipeline and WiFlexFormer baseline (code, results, report)
M3_g7_aialgorithm_wifisensing/   Val_TTA/Test protocol, 5-row ablation over 3 seeds (report)
M4_g7_aialgorithm_wifisensing/   Final submission: report, video, code, data instructions, results
```

Start with `M4_g7_aialgorithm_wifisensing/README.md`.

## Key results (Test = 30,749 unseen room-1 samples, macro-F1)

| Configuration | Ours, 8 epochs (3 seeds) | Paper [1] |
|---|---|---|
| W (WiFlexFormer) | 47.43 ± 0.44 | 40.62 |
| W + augmentation | 48.29 ± 0.11 | 49.32 |
| W_DAT + augmentation | 46.81 ± 0.83 | 65.66 |
| DATTA (TTA + reset) | 46.59 ± 1.35 | 68.13 |

W + augmentation reproduces the paper. The DAT and DATTA gains were not reproduced; training for 15 or 30 epochs (M4) did not close the gap. Details are in the M4 report.

## Timeline

- **M1 (13 Sep):** problem, literature, dataset and protocol scoping
- **M2 (11 Oct):** Widar3.0-G6D pipeline and WiFlexFormer baseline
- **M3 (1 Nov):** augmentation, DAT, TTA and resetting; ablation and error analysis over 3 seeds
- **M4 (14 Nov):** training-length study (15 and 30 epochs), adaptation on longer-trained models, final report and video

## Team (Group 7)

- Pranav Jain (AU2420166)
- Paarth Jawaharani (AU2420056)
- Kritika Lunkad (AU2320067)

## References

[1] J. Strohmayer, R. Sterzinger, M. Wödlinger, M. Kampel, "DATTA: Domain-Adversarial Test-Time Adaptation for Cross-Domain WiFi-Based Human Activity Recognition," WACV 2026. Code: https://github.com/StrohmayerJ/DATTA
Widar3.0 dataset: https://tns.thss.tsinghua.edu.cn/widar3.0/
