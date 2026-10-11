"""Train a plain WiFlexFormer baseline on pre-built Widar3.0-G6D caches.

Validation is an 80/20 stratified split of the training cache.  The separate
test cache is deliberately untouched until the final single evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.wiflexformer import WiFlexFormer
from widar_baseline_dataset import (
    EXPECTED_NUM_CLASSES,
    EXPECTED_WINDOW_SIZE,
    WidarAmplitudeDataset,
    load_widar_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plain WiFlexFormer baseline for Widar3.0-G6D.")
    parser.add_argument("--train-cache", type=Path, required=True, help="Path to widar3-g6_csi_domain_train_cache.pkl")
    parser.add_argument("--test-cache", type=Path, required=True, help="Path to widar3-g6_csi_domain_test_cache.pkl")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/wiflexformer_widar_baseline"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8, help="Start at 8 on a 16 GB CPU-only machine; raise only if RAM permits.")
    parser.add_argument("--num-workers", type=int, default=0, help="0 is the reliable Windows default.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=0, help="0 leaves PyTorch's thread count unchanged.")
    parser.add_argument("--sanity-check", action="store_true", help="Validate one batch and one model forward pass, then exit.")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs and batch-size must be at least 1; num-workers must be at least 0.")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("val-fraction must be strictly between 0 and 1.")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CPU execution is deterministic for this model with these settings.
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # CPU-only baseline
        persistent_workers=num_workers > 0,
        generator=generator,
    )


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, description: str) -> Dict[str, float]:
    model.eval()
    all_targets, all_predictions = [], []
    for features, targets in tqdm(loader, desc=description, leave=False):
        logits = model(features.to(device, non_blocking=False))
        all_targets.append(targets.numpy())
        all_predictions.append(logits.argmax(dim=1).cpu().numpy())

    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }


def train_one_epoch(model, loader, optimizer, criterion, device, epoch: int) -> float:
    model.train()
    running_loss = 0.0
    for features, targets in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * targets.size(0)
    return running_loss / len(loader.dataset)


def build_model(input_dim: int) -> WiFlexFormer:
    return WiFlexFormer(
        input_dim=input_dim,
        num_channels=1,
        window_size=EXPECTED_WINDOW_SIZE,
        num_classes=EXPECTED_NUM_CLASSES,
    )


def main() -> None:
    args = parse_args()
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    device = torch.device("cpu")
    print(f"Device: {device}; PyTorch: {torch.__version__}")

    print(f"Loading training cache: {args.train_cache}")
    train_cache = load_widar_cache(args.train_cache)
    print(f"Loading held-out test cache: {args.test_cache}")
    test_cache = load_widar_cache(args.test_cache)
    input_dim = int(train_cache["csiComplex"].shape[1])
    test_input_dim = int(test_cache["csiComplex"].shape[1])
    if test_input_dim != input_dim:
        raise ValueError(
            f"Train and test cache subcarrier counts differ: {input_dim} versus {test_input_dim}."
        )

    indices = np.arange(len(train_cache["labels_zero_based"]))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=args.val_fraction,
        random_state=args.seed,
        shuffle=True,
        stratify=train_cache["labels_zero_based"],
    )
    train_dataset = WidarAmplitudeDataset(train_cache, train_indices)
    val_dataset = WidarAmplitudeDataset(train_cache, val_indices)
    test_dataset = WidarAmplitudeDataset(test_cache)
    print(f"Samples: train={len(train_dataset)}, val={len(val_dataset)}, final_test={len(test_dataset)}")

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, args.seed)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, args.seed + 1)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, args.seed + 2)
    model = build_model(input_dim).to(device)

    features, labels = next(iter(train_loader))
    with torch.inference_mode():
        logits = model(features.to(device))
    expected_feature_shape = (min(args.batch_size, len(train_dataset)), 1, input_dim, EXPECTED_WINDOW_SIZE)
    if tuple(features.shape) != expected_feature_shape or tuple(logits.shape) != (features.shape[0], EXPECTED_NUM_CLASSES):
        raise RuntimeError(f"Sanity-check shape failure: features={tuple(features.shape)}, logits={tuple(logits.shape)}")
    print(f"Sanity check passed: features={tuple(features.shape)}, labels={tuple(labels.shape)}, logits={tuple(logits.shape)}")
    if args.sanity_check:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_validation_macro_f1.pt"
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, handle, indent=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_val_f1 = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics = evaluate(model, val_loader, device, "Validation")
        scheduler.step(val_metrics["macro_f1"])
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_accuracy={val_metrics['accuracy']:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_macro_f1": best_val_f1,
                    "model_kwargs": {"input_dim": input_dim, "num_channels": 1, "window_size": 220, "num_classes": 6},
                },
                checkpoint_path,
            )
            print(f"  Saved best checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, "Final held-out test")
    result = {
        "checkpoint": str(checkpoint_path),
        "best_validation_macro_f1": checkpoint["best_val_macro_f1"],
        "final_test_accuracy": test_metrics["accuracy"],
        "final_test_macro_f1": test_metrics["macro_f1"],
    }
    with (args.output_dir / "final_test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("Final held-out test (evaluated once after model selection):")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    # Required on Windows whenever num_workers is changed above zero.
    main()
