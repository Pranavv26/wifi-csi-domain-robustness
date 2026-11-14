"""Re-evaluate saved state_dict checkpoints (W, W+aug, W_DAT) on the 30,749-sample Test split and
save per-sample predictions + domain labels, in the same JSON layout as DATTA_local.py output."""
import json
import os
import sys
import types

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

import datasets as data
from models.wiflexformer_dat import WiFlexFormerDAT

R = r"C:\Users\Khushi Jain\Desktop\Dataset Wider\results of 3 seeds"
RUNS = {}
for cfg in ["W", "W+aug", "W_DAT"]:
    RUNS[(cfg, 1)] = rf"results_final\{cfg}_s1\best_val_loss.pt"
    RUNS[(cfg, 2)] = rf"{R}\results_seed2\{cfg}_s2\best_val_loss.pt"
    RUNS[(cfg, 3)] = rf"{R}\results_seed3\{cfg}_s3\best_val_loss.pt"

opt = types.SimpleNamespace(ws=220, classes=6, domains=7)
test_set = data.Widar3g6d("data/widar3g6d/", augPath="", opt=opt, mode="TEST")
test_set.csiComplex = None
test_idx, _ = train_test_split(list(range(len(test_set))), test_size=0.1, shuffle=True, random_state=42)
loader = DataLoader(Subset(test_set, test_idx), batch_size=128, shuffle=False, num_workers=0)
print("Test samples:", len(test_idx))

os.makedirs("tta_results", exist_ok=True)
for (cfg, seed), path in RUNS.items():
    out = f"tta_results/base_{cfg}_s{seed}.json"
    if os.path.exists(out):
        print("skip", out)
        continue
    model = WiFlexFormerDAT(grl_lambda=8.0)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    P, C, D = [], [], []
    with torch.no_grad():
        for x, c, d, _, _ in loader:
            P.append(model(x.float())["activity_recognizer_logits"].argmax(1))
            C.append(c.long())
            D.append(d.long())
    P, C, D = torch.cat(P), torch.cat(C), torch.cat(D)
    cm = torch.zeros(6, 6)
    for t, p in zip(C, P):
        cm[t, p] += 1
    rec = cm.diag() / cm.sum(1)
    prec = cm.diag() / cm.sum(0)
    r, p_ = rec.mean().item(), prec.mean().item()
    res = {"config": cfg, "seed": seed, "acc": (cm.diag().sum() / cm.sum()).item(),
           "precision": p_, "recall": r, "f1": 2 * p_ * r / (p_ + r), "n": int(cm.sum().item()),
           "confusion_matrix": cm.int().tolist(), "pred": P.tolist(), "true": C.tolist(), "domain": D.tolist()}
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"{cfg} s{seed}: F1 {res['f1']*100:.2f} Acc {res['acc']*100:.2f}", flush=True)
