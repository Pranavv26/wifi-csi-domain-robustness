import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

import datasets as data
from models.wiflexformer_dat import WiFlexFormerDAT

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def confusion(logits, target, n):
    pred = logits.argmax(dim=1)
    idx = target.long() * n + pred
    return torch.bincount(idx, minlength=n * n).reshape(n, n).cpu().float()


def scores(cm):
    rec = cm.diag() / cm.sum(1)
    prec = cm.diag() / cm.sum(0)
    r = rec[~torch.isnan(rec)].mean().item()
    p = prec[~torch.isnan(prec)].mean().item()
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    acc = (cm.diag().sum() / cm.sum()).item()
    return acc, p, r, f1


def compute_loss(model, pred, c, d, config):
    if config == "W_DAT":
        return model.loss_dat(pred, {"activity": c, "domain": d})
    return nn.CrossEntropyLoss()(pred["activity_recognizer_logits"], c)


def run_epoch(model, loader, device, config, n_cls, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    total, batches = 0.0, 0
    cm = torch.zeros(n_cls, n_cls)
    with torch.set_grad_enabled(train):
        for x, c, d, _, _ in tqdm(loader, leave=False, desc="train" if train else "val"):
            x, c, d = x.to(device).float(), c.to(device).long(), d.to(device).long()
            pred = model(x)
            loss = compute_loss(model, pred, c, d, config)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            total += loss.item()
            batches += 1
            cm += confusion(pred["activity_recognizer_logits"].detach(), c, n_cls)
    return total / max(batches, 1), scores(cm)


def evaluate(model, loader, device, n_cls):
    model.eval()
    cm = torch.zeros(n_cls, n_cls)
    n, t_total = 0, 0.0
    with torch.no_grad():
        for x, c, _, _, _ in tqdm(loader, leave=False, desc="test"):
            x, c = x.to(device).float(), c.to(device).long()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(x)["activity_recognizer_logits"]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_total += time.perf_counter() - t0
            n += x.shape[0]
            cm += confusion(logits, c, n_cls)
    acc, p, r, f1 = scores(cm)
    return {"acc": acc, "precision": p, "recall": r, "f1": f1,
            "ms_per_sample": 1000 * t_total / max(n, 1), "n_test": n,
            "confusion_matrix": cm.int().tolist()}


def train_one(opt, seed, device, test_loader):
    run_name = f"{opt.config}_s{seed}"
    run_dir = os.path.join(opt.out, run_name)
    os.makedirs(run_dir, exist_ok=True)
    result_path = os.path.join(run_dir, "result.json")
    if os.path.exists(result_path):
        print(f"[{run_name}] already finished, skipping")
        with open(result_path) as f:
            return json.load(f)

    set_seed(seed)
    aug = opt.augment if opt.config in ("W+aug", "W_DAT") else ""
    dataset = data.Widar3g6d(opt.data, augPath=aug, opt=opt, mode="TRAIN")
    dataset.csiComplex = None
    n_val = int(len(dataset) * 0.2)
    g = torch.Generator().manual_seed(42)
    ds_train, ds_val = random_split(dataset, [len(dataset) - n_val, n_val], generator=g)
    dl_train = DataLoader(ds_train, batch_size=opt.bs, shuffle=True, drop_last=True,
                          num_workers=opt.workers, worker_init_fn=worker_init,
                          generator=torch.Generator().manual_seed(seed),
                          persistent_workers=opt.workers > 0)
    dl_val = DataLoader(ds_val, batch_size=128, shuffle=False, num_workers=opt.workers,
                        persistent_workers=opt.workers > 0)

    model = WiFlexFormerDAT(grl_lambda=opt.ld).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.001, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=len(dl_train), T_mult=1, eta_min=opt.lr / 10)

    ckpt_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best_val_loss.pt")
    log_path = os.path.join(run_dir, "log.csv")
    start_epoch, best_val, best_epoch = 0, float("inf"), -1
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch, best_val, best_epoch = ck["epoch"] + 1, ck["best_val"], ck["best_epoch"]
        print(f"[{run_name}] resuming from epoch {start_epoch}")
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_f1", "val_loss", "val_f1", "sec"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{run_name}] params {n_params/1e3:.1f}K | train {len(ds_train)} | val {len(ds_val)} | aug={'on' if aug else 'off'}")

    for epoch in range(start_epoch, opt.epochs):
        t0 = time.time()
        if opt.config == "W_DAT":
            model.grl_lambda = (2.0 / (1.0 + np.exp(-10 * epoch / opt.epochs)) - 1) * opt.ld
        tr_loss, tr_s = run_epoch(model, dl_train, device, opt.config, opt.classes, optimizer, scheduler)
        va_loss, va_s = run_epoch(model, dl_val, device, opt.config, opt.classes)
        if va_loss < best_val:
            best_val, best_epoch = va_loss, epoch
            torch.save(model.state_dict(), best_path)
        sec = time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{tr_s[3]:.4f}", f"{va_loss:.4f}", f"{va_s[3]:.4f}", f"{sec:.1f}"])
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch,
                    "best_val": best_val, "best_epoch": best_epoch}, ckpt_path)
        print(f"[{run_name}] ep {epoch+1}/{opt.epochs} | train loss {tr_loss:.3f} F1 {tr_s[3]*100:.1f} | "
              f"val loss {va_loss:.3f} F1 {va_s[3]*100:.1f} | {sec:.0f}s")

    model.load_state_dict(torch.load(best_path, map_location=device))
    res = evaluate(model, test_loader, device, opt.classes)
    res.update({"config": opt.config, "seed": seed, "epochs": opt.epochs, "best_epoch": best_epoch + 1,
                "best_val_loss": best_val, "params": n_params, "bs": opt.bs, "lr": opt.lr})
    with open(result_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{run_name}] TEST | Acc {res['acc']*100:.2f} | P {res['precision']*100:.2f} | "
          f"R {res['recall']*100:.2f} | F1 {res['f1']*100:.2f} | {res['ms_per_sample']:.3f} ms/sample")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["W", "W+aug", "W_DAT"], required=True)
    ap.add_argument("--data", default="data/widar3g6d/")
    ap.add_argument("--augment", default="aug/default.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.00005)
    ap.add_argument("--ld", type=float, default=8.0)
    ap.add_argument("--classes", type=int, default=6)
    ap.add_argument("--domains", type=int, default=7)
    ap.add_argument("--ws", type=int, default=220)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results")
    opt = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    test_set = data.Widar3g6d(opt.data, augPath="", opt=opt, mode="TEST")
    test_set.csiComplex = None
    test_idx, _ = train_test_split(list(range(len(test_set))), test_size=0.1, shuffle=True, random_state=42)
    test_loader = DataLoader(Subset(test_set, test_idx), batch_size=128, shuffle=False, num_workers=opt.workers)

    results = [train_one(opt, s, device, test_loader) for s in opt.seeds]

    f1 = np.array([r["f1"] for r in results]) * 100
    acc = np.array([r["acc"] for r in results]) * 100
    summary = {"config": opt.config, "seeds": opt.seeds, "epochs": opt.epochs,
               "f1_mean": f1.mean(), "f1_std": f1.std(), "acc_mean": acc.mean(), "acc_std": acc.std(),
               "per_seed_f1": f1.round(2).tolist()}
    with open(os.path.join(opt.out, f"summary_{opt.config}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n==== {opt.config} over seeds {opt.seeds}: F1 {f1.mean():.2f} ± {f1.std():.2f} | Acc {acc.mean():.2f} ± {acc.std():.2f}")


if __name__ == "__main__":
    main()
