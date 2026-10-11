"""Dataset utilities for the plain WiFlexFormer Widar3.0-G6D baseline.

The Widar cache holds complex CSI as [samples, 52, 220].  This dataset converts
one sample to amplitude only when it is requested, avoiding a second full copy
of the cache in RAM.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


EXPECTED_SUBCARRIERS = 52
EXPECTED_WINDOW_SIZE = 220
EXPECTED_NUM_CLASSES = 6


def load_widar_cache(cache_path: str | Path) -> Dict[str, np.ndarray]:
    """Load and validate one cache created by utils/createWidar3g6d.py."""
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Widar cache was not found: {cache_path}")

    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)

    required = {"csiComplex", "activities", "environments", "users", "domains", "T_MAX"}
    missing = required.difference(cache)
    if missing:
        raise KeyError(f"{cache_path} is missing cache keys: {sorted(missing)}")

    csi = np.asarray(cache["csiComplex"])
    activities = np.asarray(cache["activities"])
    if csi.ndim != 3:
        raise ValueError(f"Expected CSI [N, subcarriers, 220], got shape {csi.shape}.")
    if csi.shape[1] < 1 or csi.shape[2] != EXPECTED_WINDOW_SIZE:
        raise ValueError(
            "This plain baseline requires CSI shaped [N, subcarriers, 220], but got "
            f"{csi.shape}."
        )
    if csi.shape[0] != activities.shape[0]:
        raise ValueError("csiComplex and activities have different sample counts.")
    if not np.iscomplexobj(csi):
        raise TypeError(f"Expected complex CSI, got dtype {csi.dtype}.")

    labels = activities.astype(np.int64, copy=False) - 1  # cache labels 1..6 -> CrossEntropy labels 0..5
    unique_labels = np.unique(labels)
    expected_labels = np.arange(EXPECTED_NUM_CLASSES)
    if not np.array_equal(unique_labels, expected_labels):
        raise ValueError(
            "Expected every activity label 1..6 in this cache; found original labels "
            f"{(unique_labels + 1).tolist()}."
        )

    cache["csiComplex"] = csi
    cache["labels_zero_based"] = labels
    return cache


class WidarAmplitudeDataset(Dataset):
    """Indexable amplitude examples with output shape [1, 52, 220]."""

    def __init__(self, cache: Dict[str, np.ndarray], indices: Optional[Sequence[int]] = None):
        self.csi = cache["csiComplex"]
        self.labels = cache["labels_zero_based"]
        self.indices = np.arange(len(self.labels), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)

        if self.indices.ndim != 1 or len(self.indices) == 0:
            raise ValueError("indices must be a non-empty one-dimensional sequence.")
        if self.indices.min() < 0 or self.indices.max() >= len(self.labels):
            raise IndexError("An index is outside the cache sample range.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        # np.abs creates only this sample's float32 amplitude array, not the whole cache.
        amplitude = np.abs(self.csi[index]).astype(np.float32, copy=False)
        features = torch.from_numpy(amplitude).unsqueeze(0)  # [1, 52, 220]
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return features, label
