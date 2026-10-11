# Results

## Plain WiFlexFormer Baseline

The baseline was trained for 15 epochs on CPU using the Widar3.0-G6D dataset. The best checkpoint was selected using validation macro-F1 and evaluated on the held-out test set.

| Metric | Result |
|---|---:|
| Best validation macro-F1 | 48.81% |
| Final test accuracy | 43.34% |
| Final test macro-F1 | 45.96% |

## Files

- `final_test_metrics.json` — Final test evaluation metrics.
- `run_config.json` — Experiment configuration.
- `training_history.csv` — Training loss and validation metrics by epoch.
- `best_validation_macro_f1.pt` — Checkpoint selected using validation macro-F1.

## Scope

These results represent one baseline run. They do not establish the performance of the planned cross-environment, SNR robustness, or domain-adaptation experiments.
