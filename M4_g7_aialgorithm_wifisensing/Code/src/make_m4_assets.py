"""Builds the M4 tables (CSV + LaTeX snippets) and figures from the raw result files.

Run:  python make_m4_assets.py            (paths below are relative to the project folder)
Only reads result files; nothing is trained here.
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = r"C:\Users\Khushi Jain\Desktop\Wifi sensing project"
DATTA = r"C:\Users\Khushi Jain\Downloads\DATTA-main"
M3 = os.path.join(ROOT, "M3_g7_aialgorithm_wifisensing", "Results", "raw", "training_runs")
R15 = os.path.join(DATTA, "results_15ep")
R30 = {2: os.path.join(ROOT, "results_30ep_seed2_raw", "results_30ep"),
       3: os.path.join(ROOT, "results_30ep_kritika_raw", "results_30ep")}
TTA = os.path.join(DATTA, "tta_results")
OUT = os.path.join(ROOT, "M4_g7_aialgorithm_wifisensing", "Results")
FIG = os.path.join(ROOT, "M4_report_latex", "figs")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

CONFIGS = ["W", "W+aug", "W_DAT"]
LABEL = {"W": "W", "W+aug": "W + aug", "W_DAT": "W_DAT + aug"}
COLOR = {"W": "#7f7f7f", "W+aug": "#26548a", "W_DAT": "#d9822b"}
CLASSES = ["Push&Pull", "Sweep", "Clap", "Slide", "Draw-circle", "Draw-zigzag"]


def load(path):
    with open(path) as f:
        return json.load(f)


def f1_of_cm(cm):
    cm = np.array(cm, dtype=float)
    rec = np.diag(cm) / cm.sum(1)
    prec = np.diag(cm) / cm.sum(0)
    r, p = np.nanmean(rec), np.nanmean(prec)
    return 100 * 2 * p * r / (p + r), 100 * np.diag(cm).sum() / cm.sum(), 100 * rec


# ---------------------------------------------------------------- training runs
# F1[(config, epochs, seed)] = (f1, acc) in %, as measured when the run finished (training-time evaluation)
RES = {}
for c in CONFIGS:
    for s in (1, 2, 3):
        r = load(os.path.join(M3, f"{c}_s{s}", "result.json"))
        RES[(c, 8, s)] = r
    r = load(os.path.join(R15, f"{c}_s1", "result.json"))
    RES[(c, 15, 1)] = r
    for s in (2, 3):
        RES[(c, 30, s)] = load(os.path.join(R30[s], f"{c}_s{s}", "result.json"))


def f1(c, e, s):
    return 100 * RES[(c, e, s)]["f1"]


def acc(c, e, s):
    return 100 * RES[(c, e, s)]["acc"]


# ---------------------------------------------------------------- table: all runs
rows = []
for (c, e, s), r in sorted(RES.items(), key=lambda k: (CONFIGS.index(k[0][0]), k[0][1], k[0][2])):
    rows.append([LABEL[c], e, s, f"{100*r['f1']:.2f}", f"{100*r['acc']:.2f}",
                 f"{100*r['precision']:.2f}", f"{100*r['recall']:.2f}", r["best_epoch"]])
with open(os.path.join(OUT, "all_training_runs.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["config", "epochs", "seed", "macro_f1", "acc", "precision", "recall", "best_epoch"])
    w.writerows(rows)

# ---------------------------------------------------------------- paired effect of longer training
pairs = []
for c in CONFIGS:
    for s, e2 in ((1, 15), (2, 30), (3, 30)):
        pairs.append((c, s, e2, f1(c, 8, s), f1(c, e2, s)))
with open(os.path.join(OUT, "training_length_pairs.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["config", "seed", "longer_epochs", "f1_8ep", "f1_longer", "delta"])
    for c, s, e2, a, b in pairs:
        w.writerow([c, s, e2, f"{a:.2f}", f"{b:.2f}", f"{b-a:+.2f}"])

summary = {}
for c in CONFIGS:
    d30 = [f1(c, 30, s) for s in (2, 3)]
    d8_23 = [f1(c, 8, s) for s in (2, 3)]
    summary[c] = {"f1_8_all3": np.mean([f1(c, 8, s) for s in (1, 2, 3)]),
                  "f1_8_s23": np.mean(d8_23), "f1_30_s23": np.mean(d30),
                  "f1_8_s1": f1(c, 8, 1), "f1_15_s1": f1(c, 15, 1),
                  "acc_8_s23": np.mean([acc(c, 8, s) for s in (2, 3)]),
                  "acc_30_s23": np.mean([acc(c, 30, s) for s in (2, 3)])}
with open(os.path.join(OUT, "training_length_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)

# ---------------------------------------------------------------- figure 1: F1 vs training length
fig, ax = plt.subplots(figsize=(6.4, 3.5))
xs = {8: 0, 15: 1, 30: 2}
markers = {1: "o", 2: "s", 3: "^"}
for c in CONFIGS:
    for s, e2 in ((1, 15), (2, 30), (3, 30)):
        ax.plot([xs[8], xs[e2]], [f1(c, 8, s), f1(c, e2, s)], color=COLOR[c], alpha=0.8,
                marker=markers[s], ms=5, lw=1.2)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["8 epochs", "15 epochs", "30 epochs"])
ax.set_ylabel("Test macro-F1 (%)")
ax.set_xlim(-0.25, 2.25)
ax.grid(alpha=0.25)
h1 = [plt.Line2D([0], [0], color=COLOR[c], lw=2, label=LABEL[c]) for c in CONFIGS]
h2 = [plt.Line2D([0], [0], color="k", marker=markers[s], lw=0, label=f"seed {s}") for s in (1, 2, 3)]
ax.legend(handles=h1 + h2, fontsize=8, ncol=2, loc="lower center", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "f1_vs_epochs.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- figure 2: 30-epoch validation curves
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for c in CONFIGS:
    curves = []
    for s in (2, 3):
        rows_ = list(csv.DictReader(open(os.path.join(R30[s], f"{c}_s{s}", "log.csv"))))
        curves.append([100 * float(r["val_f1"]) for r in rows_])
    n = min(len(x) for x in curves)
    m = np.mean([x[:n] for x in curves], axis=0)
    axs[0].plot(range(1, n + 1), m, color=COLOR[c], label=LABEL[c], lw=1.4)
    curves = []
    for s in (2, 3):
        rows_ = list(csv.DictReader(open(os.path.join(R30[s], f"{c}_s{s}", "log.csv"))))
        curves.append([float(r["val_loss"]) for r in rows_])
    m = np.mean([x[:n] for x in curves], axis=0)
    if c != "W_DAT":  # DAT loss contains extra terms, not comparable
        axs[1].plot(range(1, n + 1), m, color=COLOR[c], label=LABEL[c], lw=1.4)
axs[0].axvline(8, color="k", ls=":", lw=0.8); axs[1].axvline(8, color="k", ls=":", lw=0.8)
axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Source-validation macro-F1 (%)")
axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Source-validation loss")
axs[0].legend(fontsize=7, frameon=False); axs[0].grid(alpha=0.25); axs[1].grid(alpha=0.25)
axs[1].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "val_curves_30ep.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- sec per epoch
sec = {}
for tag, path in [("seed1_15ep_resumed", os.path.join(R15, "W_DAT_s1", "log.csv"))] + \
                 [(f"seed{s}_30ep", os.path.join(R30[s], "W_DAT_s%d" % s, "log.csv")) for s in (2, 3)]:
    rows_ = list(csv.DictReader(open(path)))
    sec[tag] = float(np.median([float(r["sec"]) for r in rows_]))
with open(os.path.join(OUT, "sec_per_epoch.json"), "w") as fh:
    json.dump(sec, fh, indent=2)

# ---------------------------------------------------------------- adaptation results
ADAPT = {}  # key: (tag, kind) -> dict
for tag, names in {"15ep_s1": ("test15_tta_e15_1", "test15_datta_e15_1"),
                   "30ep_s3": ("test30_tta_e30s3_1", "test30_datta_e30s3_1"),
                   "30ep_s2": ("test30_tta_e30s2_1", "test30_datta_e30s2_1")}.items():
    for kind, n in zip(("TTA", "DATTA"), names):
        p = os.path.join(TTA, n + ".json")
        if os.path.exists(p):
            ADAPT[(tag, kind)] = load(p)
base_json = {"15ep_s1": RES[("W_DAT", 15, 1)], "30ep_s3": RES[("W_DAT", 30, 3)], "30ep_s2": RES[("W_DAT", 30, 2)]}

rows = []
for tag in ("15ep_s1", "30ep_s2", "30ep_s3"):
    b = base_json[tag]
    bf, ba, _ = f1_of_cm(b["confusion_matrix"])
    line = [tag, f"{bf:.2f}"]
    for kind in ("TTA", "DATTA"):
        a = ADAPT.get((tag, kind))
        line.append(f"{100*a['f1']:.2f}" if a else "NA")
    rows.append(line)
with open(os.path.join(OUT, "adaptation_results.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["run", "W_DAT_no_adapt_f1", "TTA_f1", "DATTA_f1"])
    w.writerows(rows)
print(open(os.path.join(OUT, "adaptation_results.csv")).read())

# per-class recall at 30 epochs (seeds with adaptation available)
tags = [t for t in ("30ep_s2", "30ep_s3") if (t, "TTA") in ADAPT and (t, "DATTA") in ADAPT]
if tags:
    rec = {"W_DAT": [], "TTA": [], "DATTA": []}
    for t in tags:
        rec["W_DAT"].append(f1_of_cm(base_json[t]["confusion_matrix"])[2])
        rec["TTA"].append(f1_of_cm(ADAPT[(t, "TTA")]["confusion_matrix"])[2])
        rec["DATTA"].append(f1_of_cm(ADAPT[(t, "DATTA")]["confusion_matrix"])[2])
    rec = {k: np.mean(v, axis=0) for k, v in rec.items()}
    with open(os.path.join(OUT, "per_class_recall_30ep.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "W_DAT", "TTA", "DATTA"])
        for i, cl in enumerate(CLASSES):
            w.writerow([cl] + [f"{rec[k][i]:.2f}" for k in ("W_DAT", "TTA", "DATTA")])
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    wd = 0.26
    for j, (k, col) in enumerate([("W_DAT", COLOR["W_DAT"]), ("TTA", "#4c9f70"), ("DATTA", "#26548a")]):
        ax.bar(np.arange(6) + (j - 1) * wd, rec[k], wd, label={"W_DAT": "W_DAT + aug", "TTA": "+ TTA", "DATTA": "DATTA (full)"}[k], color=col)
    ax.set_xticks(range(6)); ax.set_xticklabels(CLASSES, fontsize=8)
    ax.set_ylabel("Recall (%)"); ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
    ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "per_class_30ep.png"), dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- print the numbers that go into the report
print("== per run (F1 / acc) ==")
for c in CONFIGS:
    for e, ss in ((8, (1, 2, 3)), (15, (1,)), (30, (2, 3))):
        print(c, e, [(s, round(f1(c, e, s), 2), round(acc(c, e, s), 2)) for s in ss],
              "mean F1 %.2f" % np.mean([f1(c, e, s) for s in ss]))
print("== pairs ==")
for c, s, e2, a, b in pairs:
    print(c, "seed", s, e2, f"{a:.2f}->{b:.2f} ({b-a:+.2f})")
print("== summary ==")
print(json.dumps(summary, indent=1))
print(json.dumps(sec, indent=1))
