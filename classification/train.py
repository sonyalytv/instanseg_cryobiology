"""
Training script for the cell-type classifier.

Usage (local):
    python -m classification.train \
        --data_path  "C:/Work/.../by_type" \
        --checkpoint "C:/Work/.../my_instanseg" \
        --output_path "./classification_results"

Usage (Kaggle):
    python -m classification.train \
        --data_path  "/kaggle/input/cell-images/by_type" \
        --checkpoint "/kaggle/input/instanseg-ckpt/my_instanseg" \
        --output_path "/kaggle/working/classification_results"
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from classification.config import (
    CLASS_NAMES, NUM_CLASSES, IDX_TO_CLASS,
    DEFAULT_BATCH_SIZE, DEFAULT_NUM_EPOCHS, DEFAULT_LR, DEFAULT_LR_ENCODER,
    DEFAULT_WEIGHT_DECAY, DEFAULT_PATIENCE, DEFAULT_FREEZE_BLOCKS,
    DEFAULT_NUM_WORKERS, DEFAULT_VAL_SPLIT, DEFAULT_RANDOM_SEED,
    DEFAULT_LONG_SIDE, DEFAULT_TILE_SIZE, ENCODER_LAYERS,
)
from classification.dataset import CellTypeDataset, collect_image_paths
from classification.augmentations import TrainTransform, ValTransform
from classification.model import (
    CellTypeClassifier, load_encoder_weights, save_classifier_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a cell-type classifier on top of InstanSeg encoder."
    )
    # ── Paths (the only things that change between local and Kaggle) ─────
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root folder containing class subfolders "
                             "(epithelial/, fibroblasts/, leukocytes/, neuroblasts/)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pre-trained InstanSeg checkpoint "
                             "(folder or .pth file). If None, train from scratch.")
    parser.add_argument("--output_path", type=str, default="./classification_results",
                        help="Where to save model checkpoints, metrics, and plots.")

    # ── Hyperparameters ──────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--lr_encoder", type=float, default=DEFAULT_LR_ENCODER)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--freeze_blocks", type=int, default=DEFAULT_FREEZE_BLOCKS)
    parser.add_argument("--long_side", type=int, default=DEFAULT_LONG_SIDE)
    parser.add_argument("--tile_size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--val_split", type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if omitted)")

    return parser.parse_args()


def _choose_device(requested: str = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalized so they sum to num_classes."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def _build_sampler(labels: list, num_classes: int) -> WeightedRandomSampler:
    """WeightedRandomSampler to oversample minority classes."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    class_weight = 1.0 / (counts + 1e-6)
    sample_weights = [class_weight[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )


def _collate_fn(batch):
    """Custom collate that pads images to the same size within a batch."""
    images, labels = zip(*batch)
    # Find max H, W in this batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded = []
    for img in images:
        _, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        if pad_h > 0 or pad_w > 0:
            img = torch.nn.functional.pad(
                img, (0, pad_w, 0, pad_h), mode="constant", value=0
            )
        padded.append(img)

    return torch.stack(padded), torch.tensor(labels, dtype=torch.long)


def train_one_epoch(model, dataloader, criterion, optimizer, device, clip_grad=5.0):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(all_labels)
    epoch_acc = balanced_accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(all_labels)
    epoch_acc = balanced_accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def main():
    args = parse_args()
    _seed_everything(args.seed)
    device = _choose_device(args.device)
    print(f"Using device: {device}")

    # ── Output directory ─────────────────────────────────────────────────
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Collect data ─────────────────────────────────────────────────────
    print(f"\nCollecting images from: {args.data_path}")
    all_paths, all_labels = collect_image_paths(args.data_path)
    if len(all_paths) == 0:
        print("[ERROR] No images found. Check --data_path.")
        sys.exit(1)

    # ── Stratified split ─────────────────────────────────────────────────
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels,
        test_size=args.val_split,
        stratify=all_labels,
        random_state=args.seed,
    )
    print(f"\nTrain: {len(train_paths)} | Val: {len(val_paths)}")

    # ── Class weights ────────────────────────────────────────────────────
    class_weights = _compute_class_weights(train_labels, NUM_CLASSES).to(device)
    print(f"Class weights: {dict(zip(CLASS_NAMES, class_weights.cpu().numpy().round(3)))}")

    # ── Datasets & loaders ───────────────────────────────────────────────
    train_ds = CellTypeDataset(
        train_paths, train_labels,
        long_side=args.long_side,
        transform=TrainTransform(crop_size=args.tile_size),
    )
    val_ds = CellTypeDataset(
        val_paths, val_labels,
        long_side=args.long_side,
        transform=ValTransform(crop_size=args.tile_size),
    )

    train_sampler = _build_sampler(train_labels, NUM_CLASSES)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = CellTypeClassifier(
        num_classes=NUM_CLASSES,
        layers=ENCODER_LAYERS,
        freeze_blocks=args.freeze_blocks,
    )

    if args.checkpoint:
        model = load_encoder_weights(model, args.checkpoint, device=str(device))

    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ── Optimizer with differential LR ───────────────────────────────────
    encoder_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and n.startswith("encoder")]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and n.startswith("head")]

    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": args.lr_encoder})
    param_groups.append({"params": head_params, "lr": args.lr})

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # ── Loss ─────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Training loop ────────────────────────────────────────────────────
    best_val_acc = -1.0
    no_improvement = 0
    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"Starting training for {args.num_epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_preds, val_labels_arr = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0

        # Log
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{args.num_epochs} | "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | "
              f"{elapsed:.0f}s")

        # Checkpointing
        save_classifier_checkpoint(
            model, optimizer, epoch, val_acc,
            str(output_path / "model_last.pth"),
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improvement = 0
            save_classifier_checkpoint(
                model, optimizer, epoch, val_acc,
                str(output_path / "model_best.pth"),
            )
            print(f"  [+] New best val_acc: {best_val_acc:.4f}")
        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                print(f"\nEarly stopping after {args.patience} epochs "
                      f"without improvement (best val_acc={best_val_acc:.4f})")
                break

    # ── Save metrics ─────────────────────────────────────────────────────
    df = pd.DataFrame(history)
    df.to_csv(output_path / "training_metrics.csv", index=False)

    # ── Plot loss & accuracy curves ──────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["epoch"], history["train_loss"], label="Train")
    ax1.plot(history["epoch"], history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["epoch"], history["train_acc"], label="Train")
    ax2.plot(history["epoch"], history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Balanced Accuracy")
    ax2.set_title("Balanced Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "training_curves.png", dpi=150)
    plt.close()

    # ── Save args ────────────────────────────────────────────────────────
    pd.DataFrame.from_dict(vars(args), orient="index").to_csv(
        output_path / "training_args.csv", header=False
    )

    print(f"\n{'='*60}")
    print(f"Training complete. Best val balanced accuracy: {best_val_acc:.4f}")
    print(f"Results saved to: {output_path.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
