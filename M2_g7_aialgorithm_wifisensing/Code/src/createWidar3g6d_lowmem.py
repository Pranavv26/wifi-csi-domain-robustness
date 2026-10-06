"""
createWidar3g6d_lowmem.py

Low-memory drop-in replacement for DATTA's utils/createWidar3g6d.py
(Strohmayer et al., WACV 2026; github.com/StrohmayerJ/DATTA).

Same file selection, same downsampling (1000 Hz -> 100 Hz), same per-sample
normalisation, same 120-220 length filter, same labels/domains, and the same
output cache files and dictionary keys, so DATTA's datasets.py loads them
unchanged.

The only change is how zero-padding is done. The original converts every
sample to nested Python lists (.tolist()) before building the padded array,
which needs ~16 GB RAM. Here samples are copied straight into a pre-allocated
complex64 NumPy array (the values are already complex64, so nothing is lost),
and each per-sample array is freed as soon as it is copied. Peak RAM is about
3-4 GB, so it runs on an 8 GB laptop.

Usage (from the DATTA repo root, same as the original):
    python3 utils/createWidar3g6d_lowmem.py --mode TRAIN
    python3 utils/createWidar3g6d_lowmem.py --mode TEST
"""

import argparse
import gc
import os
import pickle
import warnings

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

SELECTED_LABELS = [1, 2, 3, 4, 5, 6]
SELECTED_RX = [1, 2, 3, 4, 5, 6]
MIN_LEN, MAX_LEN = 120, 220

# Domain ids exactly as in the original script.
#   TEST : room 1, users 5,10,11,12,13,14,15,16,17 -> domains 0..8
#   TRAIN: room 2 users 1,2,6 -> 9,10,11 ; room 3 users 3,7,8,9 -> 12..15
ROOM1_USERS = [5, 10, 11, 12, 13, 14, 15, 16]          # anything else in room 1 -> 8
ROOM2_USERS = {1: 9, 2: 10}                             # anything else in room 2 -> 11
ROOM3_USERS = {3: 12, 7: 13, 8: 14}                     # anything else in room 3 -> 15


def domain_label(env, user):
    if env == 1:
        return ROOM1_USERS.index(user) if user in ROOM1_USERS else 8
    if env == 2:
        return ROOM2_USERS.get(user, 11)
    return ROOM3_USERS.get(user, 15)


def environment_of(path, mode):
    """Return the room id for a file, or None if the file belongs to the other split."""
    if "20181130" in path:
        return None if mode == "TRAIN" else 1
    if "20181204" in path or "20181209" in path:
        return None if mode == "TEST" else 2
    return None if mode == "TEST" else 3


def create_split(opt):
    import csiread  # imported here so the module can be inspected without it

    if opt.mode not in ("TRAIN", "TEST"):
        print("Invalid mode. Please select TRAIN or TEST.")
        return

    out_name = ("widar3-g6_csi_domain_train_cache.pkl" if opt.mode == "TRAIN"
                else "widar3-g6_csi_domain_test_cache.pkl")
    data_cache = os.path.join(opt.data, out_name)

    all_files = []
    for root, _, files in os.walk(opt.data):
        for name in files:
            path = os.path.join(root, name)
            if "cache" in path:
                continue
            all_files.append((root, name))
    print(f"Total files to process: {len(all_files)}")

    samples, acts, envs, users, doms, lengths = [], [], [], [], [], []
    n_sub = None
    T_MAX = 0

    for root, name in tqdm(all_files, desc="Processing CSI files", unit="file"):
        path = os.path.join(root, name)
        env = environment_of(path, opt.mode)
        if env is None:
            continue

        # Parse labels from the file name BEFORE reading the CSI, so files we
        # would discard anyway are skipped cheaply. (Same result as the original,
        # which parsed after reading.)
        try:
            act = int(name.split("-")[1])
            rx = int(name.split("-")[5].split(".")[0][1:])
            user = int(name.split("-")[0][4:])
        except (IndexError, ValueError):
            continue
        if act not in SELECTED_LABELS or rx not in SELECTED_RX:
            continue

        csidata = csiread.Intel(path, nrxnum=3, ntxnum=1, pl_size=10, if_report=False)
        csidata.read()
        csi = csidata.get_scaled_csi()[:, :, 0, 0]    # first antenna
        csi = np.transpose(csi, (1, 0))               # (subcarriers, time)
        csi = csi[:, 0::10]                           # 1000 Hz -> 100 Hz
        del csidata

        if csi.shape[1] == 0:
            continue
        max_abs = np.max(np.abs(csi))
        if max_abs == 0:
            continue
        csi = (csi / max_abs).astype(np.complex64)

        length = csi.shape[1]
        if length < MIN_LEN or length > MAX_LEN:
            continue

        n_sub = csi.shape[0] if n_sub is None else n_sub
        T_MAX = max(T_MAX, length)
        samples.append(np.ascontiguousarray(csi))
        acts.append(act)
        envs.append(env)
        users.append(user)
        doms.append(domain_label(env, user))
        lengths.append(length)

    if not samples:
        print("No samples found. Check that the unzipped Widar3.0 folders are inside", opt.data)
        return

    # Zero-pad into one pre-allocated array, freeing each sample after copying.
    N = len(samples)
    csiComplex = np.zeros((N, n_sub, T_MAX), dtype=np.complex64)
    for i in range(N):
        s = samples[i]
        csiComplex[i, :, : s.shape[1]] = s
        samples[i] = None
    del samples
    gc.collect()

    a = np.array(acts, dtype=np.int8)
    e = np.array(envs, dtype=np.int8)
    u = np.array(users, dtype=np.int8)
    d = np.array(doms, dtype=np.int8)

    data_to_save = {
        "csiComplex": csiComplex,
        "activities": a,
        "environments": e,
        "users": u,
        "domains": d,
        "T_MAX": T_MAX,
    }

    for v in np.unique(a):
        print("Activity:", v, "Number of samples:", int((a == v).sum()))
    for v in np.unique(e):
        print("Environment:", v, "Number of samples:", int((e == v).sum()))
    for v in np.unique(u):
        print("User:", v, "Number of samples:", int((u == v).sum()))
    for v in np.unique(d):
        print("Domain:", v, "Number of samples:", int((d == v).sum()))
    print(f"{opt.mode} split number of samples: {N}, with a max length of {T_MAX}")
    print("Array shape:", csiComplex.shape, "dtype:", csiComplex.dtype,
          f"size: {csiComplex.nbytes / 1e9:.2f} GB")
    print("Unique activities:", np.unique(a), "Number of activities:", len(np.unique(a)))
    print("Unique environments:", np.unique(e), "Number of environments:", len(np.unique(e)))
    print("Unique users:", np.unique(u), "Number of users:", len(np.unique(u)))
    print("Unique domains:", np.unique(d), "Number of domains:", len(np.unique(d)))
    print("Mean sample length:", np.mean(lengths))
    print("Std sample length:", np.std(lengths))
    print("Median sample length:", np.median(lengths))
    print("Min sample length:", np.min(lengths))
    print("Max sample length:", np.max(lengths))
    print("Activity distribution:", [float((a == v).sum() / len(a)) for v in np.unique(a)])

    with open(data_cache, "wb") as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved:", data_cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/widar3g6d", help="Location of Widar3.0 data")
    parser.add_argument("--mode", default="TRAIN", help="Split type: TRAIN or TEST")
    create_split(parser.parse_args())
