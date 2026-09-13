# Cross-Domain Wi-Fi CSI-Based Human Activity Recognition via Domain-Adversarial Test-Time Adaptation

This project investigates the robustness of Wi-Fi Channel State Information (CSI)-based Human Activity Recognition (HAR) when the sensing **domain** changes — a different room, a different user, or different hardware.

Rather than focusing only on overall classification accuracy, we study how a model trained in one setting **generalizes to previously unseen domains**. Because CSI encodes the surrounding multipath channel (room geometry, furniture, line-of-sight state, device) as strongly as it encodes the activity itself, conventional HAR models tend to learn environment-specific channel signatures and degrade sharply once the domain shifts. This is fundamentally a wireless-communications problem, rooted in multipath propagation and channel variability.

## Approach

Our project adopts **DATTA — Domain-Adversarial Test-Time Adaptation** (Strohmayer et al., WACV 2026) as its technical foundation, and is scoped as a **faithful reproduction, controlled evaluation, and component-wise analysis** of that framework rather than a new algorithm. DATTA combines:

- **Domain-adversarial training (DAT)** — a Gradient Reversal Layer trains a feature extractor so a domain discriminator cannot distinguish domains, yielding domain-invariant features.
- **Test-time adaptation (TTA)** — the model adapts to the unlabeled incoming stream at inference, correcting residual domain shift.
- **Random weight resetting** — periodically restores a random subset of adapted weights to their source values to prevent catastrophic forgetting.
- **A lightweight WiFlexFormer backbone** with CSI-specific augmentation, enabling real-time inference.

We reproduce each component in stages (baseline → augmentation → DAT → TTA → random resetting) and measure how much each contributes to cross-domain performance.

## Primary Dataset — Widar3.0-G6D

We use **Widar3.0-G6D**, the 6-gesture subset of Widar3.0 used in the DATTA paper (16 participants, 3 indoor environments; Intel WiFi Link 5300, 5 GHz, 3 antennas × 30 subcarriers). A *domain* is a room–participant combination, and the data is split by domain so that test domains are unseen during training. The dataset is not stored in this repository (tens of GB); acquisition and preparation steps are documented in `Code/DATASET.md`.

## Repository Structure

```
M1_g7_aialgorithm_wifisensing/
├── README.md
├── Report/     M1 technical report (PDF)
├── Video/      M1 video walkthrough link
└── Code/       DATASET.md (Widar3.0-G6D acquisition) + M2 implementation plan
```

## Timeline

- **M1 (13 Sep)** — Problem definition, literature review, SOTA positioning, dataset and protocol scoping (this milestone).
- **M2 (11 Oct)** — Prepare Widar3.0-G6D; reproduce the WiFlexFormer baseline; construct the domain-based Train / Val / Val_TTA / Test split; measure the cross-domain gap.
- **M3 (1 Nov)** — Add augmentation, DAT, TTA, and random resetting; run the component-wise ablation against the DAT and ViTTA baselines.
- **M4 (22 Nov)** — Full evaluation, robustness–complexity analysis, final report, video, and demo.

## Team — Group 7

- Pranav Jain (AU2420166) — Anchor; WiFlexFormer / DAT + TTA implementation; SOTA positioning
- Paarth Jawaharani (AU2420056) — Data pipeline; Widar3.0-G6D preparation; baseline reproduction; split construction
- Kritika Lunkad (AU2320067) — Evaluation and ablations; benchmark tables and plots; reproducibility and report

Course: ECE 310 — Wireless Communications (Monsoon 2026) · Category: AI Algorithms + Existing Dataset · Instructor: Prof. Dhaval Patel

## Reference

J. Strohmayer, R. Sterzinger, M. Wödlinger, M. Kampel, "DATTA: Domain-Adversarial Test-Time Adaptation for Cross-Domain WiFi-Based Human Activity Recognition," WACV 2026. Code: https://github.com/StrohmayerJ/DATTA · Widar3.0: https://tns.thss.tsinghua.edu.cn/widar3.0/
