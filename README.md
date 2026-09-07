# Robust Wi-Fi CSI-Based Human Activity Recognition Under Cross-Environment and Low-SNR Conditions

This project investigates the robustness of Wi-Fi Channel State Information (CSI)-based Human Activity Recognition (HAR) when the wireless sensing conditions change.

Rather than focusing only on overall classification accuracy, we study how changes in the **indoor environment** and **signal quality (SNR)** affect model performance and whether a lightweight preprocessing/normalization approach can reduce the resulting performance degradation.

## Problem

Wi-Fi CSI contains information about how wireless signals are affected by objects and human movement in the environment. This makes CSI useful for sensing human activities without requiring cameras or wearable sensors.

However, a model trained under one set of wireless conditions may not perform equally well when the conditions change.

In particular:

- Changes in room layout and surroundings can alter the wireless propagation environment and multipath characteristics.
- Lower signal-to-noise ratio (SNR) can make CSI measurements more difficult to interpret.
- A model that performs well under standard train/test conditions may therefore experience a significant performance drop under previously unseen conditions.

This project aims to **quantify these generalization gaps and investigate a lightweight method for improving robustness**.

## Research Question

> How does cross-environment variation and reduced SNR affect Wi-Fi CSI-based human activity recognition, and can lightweight feature normalization recover part of the resulting performance loss without full model retraining?

## Objectives

The project will:

1. Develop a reproducible Wi-Fi CSI-based Human Activity Recognition pipeline.
2. Establish traditional ML and neural-network baselines.
3. Evaluate model performance under standard and unseen-environment conditions.
4. Study the effect of controlled SNR degradation on recognition performance.
5. Investigate a lightweight normalization/preprocessing approach for improving robustness.
6. Benchmark models using both sensing performance and model complexity.
7. Analyze failure cases and the conditions under which models generalize or fail.

## Dataset

We plan to use the **CSI-Bench** Wi-Fi sensing benchmark dataset.

The dataset and its experimental protocols will be documented in detail as the project progresses, including:

- Dataset source
- Sensing task and activity classes
- Number of samples
- CSI representation
- Train/validation/test splits
- Environment information
- User information
- Device information
- Experimental conditions

> Dataset configuration and final experimental splits will be documented after the initial dataset analysis.

## Planned Methodology

The planned experimental pipeline is:

```text
Wi-Fi CSI Dataset
       ↓
Data Preprocessing
       ↓
CSI Representation
       ↓
Traditional ML Baseline
       ↓
Neural Network Baseline
       ↓
Cross-Environment Evaluation
       ↓
Controlled SNR Robustness Evaluation
       ↓
Lightweight Normalization
       ↓
Benchmark + Ablation Analysis
