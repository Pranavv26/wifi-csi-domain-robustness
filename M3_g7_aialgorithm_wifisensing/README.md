# M3: Experimental Validation of DATTA on Widar3.0-G6D

Group 7 · ECE310 Wireless Communications · AI Algorithms + Benchmark Dataset

## Goal
Extend the M2 WiFlexFormer baseline with the DATTA components and measure what each one contributes to cross-domain performance on Widar3.0-G6D.

## Plan
- Evaluation protocol: split the held-out room-1 data into Val_TTA (10%) and Test (90%)
- CSI augmentation
- Domain-adversarial training (gradient reversal layer, domain head, confidence-control constraint)
- Test-time adaptation with random weight resetting
- Ablation: W, W + aug, W_DAT + aug, W_DAT + aug + TTA, DATTA
- Error analysis: per-class recall, confusion matrices, per-domain results

## Structure
- `Report/`: M3 report (PDF)

## Status
In progress. Results and run instructions will be added as experiments complete.

## Team
- Pranav Jain (AU2420166)
- Paarth Jawaharani (AU2420056)
- Kritika Lunkad (AU2320067)
