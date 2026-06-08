"""
Evaluation script for the cell-type classifier.

Usage:
    python -m classification.evaluate \
        --data_path  "/path/to/by_type" \
        --checkpoint "/path/to/model_best.pth" \
        --output_path "./eval_results"
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from classification.config import (
    CLASS_NAMES, NUM_CLASSES, IDX_TO_CLASS,
    DEFAULT_LONG_SIDE, DEFAULT_TILE_SIZE,
    DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT, DEFAULT_RANDOM_SEED, ENCODER_LAYERS,
)
from classification.dataset import (
    CellTypeDataset, collect_image_paths, _read_image, _to_3channel_float,
    _resize_long_side,
)
from classification.augmentations import ValTransform, TTATransform
from classification.model import load_classifier_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained cell-type classifier.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root folder with class subfolders")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to classifier checkpoint (.pth)")
    parser.add_argument("--output_path", type=str, default="./eval_results",
                        help="Where to save evaluation outputs")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test", "all"],
                        help="Evaluate on val split, test split, or entire dataset")
    parser.add_argument("--tta", action="store_true", default=False,
                        help="Enable test-time augmentation (4x: original + rotations)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--long_side", type=int, default=DEFAULT_LONG_SIDE)
    parser.add_argument("--tile_size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--val_split", type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--test_split", type=float, default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_vis", type=int, default=24,
                        help="Number of sample images to show in the prediction grid")
    return parser.parse_args()


def _choose_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collate_fn(batch):
    images, labels = zip(*batch)
    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)

    padded = []
    for img in images:
        if img.dim() == 4:
            # TTA: (K, C, H, W)
            k, c, h, w = img.shape
            pad_h, pad_w = max_h - h, max_w - w
            if pad_h > 0 or pad_w > 0:
                img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0)
        else:
            _, h, w = img.shape
            pad_h, pad_w = max_h - h, max_w - w
            if pad_h > 0 or pad_w > 0:
                img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0)
        padded.append(img)

    return torch.stack(padded), torch.tensor(labels, dtype=torch.long)


def _plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=11)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_prediction_grid(eval_paths, eval_labels, preds, probs,
                          long_side, num_vis, save_path):
    """Create a grid of sample images with true/predicted labels.
    Correct predictions get a green title, incorrect get red."""

    n_total = len(eval_paths)
    if n_total == 0:
        return

    # Select samples: mix of correct and incorrect
    correct_mask = np.array(eval_labels) == np.array(preds)
    incorrect_idxs = np.where(~correct_mask)[0]
    correct_idxs = np.where(correct_mask)[0]

    # Prioritise showing all misclassifications, fill rest with correct
    np.random.seed(42)
    selected = []
    if len(incorrect_idxs) > 0:
        n_wrong = min(len(incorrect_idxs), num_vis // 2)
        selected.extend(np.random.choice(incorrect_idxs, n_wrong, replace=False).tolist())
    remaining = num_vis - len(selected)
    if remaining > 0 and len(correct_idxs) > 0:
        n_right = min(len(correct_idxs), remaining)
        selected.extend(np.random.choice(correct_idxs, n_right, replace=False).tolist())

    if not selected:
        return

    ncols = min(4, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for ax_idx, sample_idx in enumerate(selected):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row, col]

        # Load raw image for display
        try:
            img = _read_image(eval_paths[sample_idx])
            img = _to_3channel_float(img)
            img = _resize_long_side(img, long_side)
        except Exception:
            img = np.zeros((long_side, long_side, 3), dtype=np.float32)

        ax.imshow(np.clip(img, 0, 1))
        ax.axis("off")

        true_name = IDX_TO_CLASS[eval_labels[sample_idx]]
        pred_name = IDX_TO_CLASS[preds[sample_idx]]
        conf = probs[sample_idx].max()
        is_correct = eval_labels[sample_idx] == preds[sample_idx]

        title_color = "#2e8b57" if is_correct else "#c0392b"
        marker = "[OK]" if is_correct else "[X]"
        ax.set_title(
            f"{marker} True: {true_name}\nPred: {pred_name} ({conf:.2f})",
            fontsize=9, fontweight="bold", color=title_color,
        )

    # Hide unused axes
    for ax_idx in range(len(selected), nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row, col].axis("off")

    plt.suptitle("Prediction Samples (green=correct, red=incorrect)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Prediction grid saved to: {save_path}")


@torch.no_grad()
def evaluate(model, dataloader, device, use_tta=False):
    """Run inference and return predictions, labels, probabilities, and timing."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_images = 0

    # Warm up GPU
    if device.type == "cuda":
        dummy = torch.randn(1, 3, 256, 256, device=device)
        _ = model(dummy)
        torch.cuda.synchronize()

    t_start = time.time()

    for images, labels in dataloader:
        batch_size = labels.shape[0]
        if use_tta:
            B, K, C, H, W = images.shape
            images = images.view(B * K, C, H, W).to(device)
            logits = model(images)
            logits = logits.view(B, K, -1).mean(dim=1)
        else:
            images = images.to(device)
            logits = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        total_images += batch_size

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.time()

    elapsed = t_end - t_start
    avg_ms = (elapsed / total_images * 1000) if total_images > 0 else 0.0

    return (np.array(all_preds), np.array(all_labels),
            np.array(all_probs), elapsed, avg_ms)


def main():
    args = parse_args()
    device = _choose_device(args.device)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt = load_classifier_checkpoint(
        args.checkpoint, device=str(device), layers=ENCODER_LAYERS, freeze_blocks=0
    )
    print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, "
          f"best metric {ckpt.get('best_metric', '?')}")

    # Collect data
    all_paths, all_labels = collect_image_paths(args.data_path)
    if len(all_paths) == 0:
        print("[ERROR] No images found.")
        sys.exit(1)

    # Use split or entire dataset
    if args.split in ["val", "test"]:
        if args.test_split > 0:
            train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
                all_paths, all_labels, test_size=args.test_split, stratify=all_labels, random_state=args.seed
            )
        else:
            train_val_paths, train_val_labels = all_paths, all_labels
            test_paths, test_labels = [], []

        if args.split == "test":
            eval_paths, eval_labels = test_paths, test_labels
            print(f"\nEvaluating on test split: {len(eval_paths)} images")
        else:
            relative_val_split = args.val_split / (1.0 - args.test_split) if args.test_split < 1 else args.val_split
            _, eval_paths, _, eval_labels = train_test_split(
                train_val_paths, train_val_labels,
                test_size=relative_val_split,
                stratify=train_val_labels,
                random_state=args.seed,
            )
            print(f"\nEvaluating on validation split: {len(eval_paths)} images")
    else:
        eval_paths, eval_labels = all_paths, all_labels
        print(f"\nEvaluating on ALL data: {len(eval_paths)} images")

    # Dataset
    transform = TTATransform(args.tile_size) if args.tta else ValTransform(args.tile_size)
    ds = CellTypeDataset(eval_paths, eval_labels, long_side=args.long_side, transform=transform)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=_collate_fn,
    )

    # Run evaluation with timing
    preds, labels, probs, total_time, avg_ms = evaluate(
        model, loader, device, use_tta=args.tta
    )

    # Metrics
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
    cm = confusion_matrix(labels, preds)

    print(f"\n{'='*60}")
    print(f"Overall accuracy:    {acc:.4f}")
    print(f"Balanced accuracy:   {bal_acc:.4f}")
    print(f"Total inference time: {total_time:.2f}s")
    print(f"Avg time per image:   {avg_ms:.2f} ms")
    print(f"{'='*60}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    # Save outputs
    # Classification report CSV
    report_dict = classification_report(
        labels, preds, target_names=CLASS_NAMES, digits=4, output_dict=True
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        output_path / "classification_report.csv"
    )

    # Confusion matrix plot
    _plot_confusion_matrix(cm, CLASS_NAMES, output_path / "confusion_matrix.png")

    # Prediction grid
    _plot_prediction_grid(
        eval_paths, eval_labels, preds, probs,
        long_side=args.long_side,
        num_vis=args.num_vis,
        save_path=output_path / "prediction_samples.png",
    )

    # Summary (includes timing)
    summary = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "total_inference_time_s": total_time,
        "avg_time_per_image_ms": avg_ms,
        "tta": args.tta,
        "num_samples": len(labels),
        "checkpoint": args.checkpoint,
    }
    pd.DataFrame.from_dict(summary, orient="index").to_csv(
        output_path / "eval_summary.csv", header=False
    )

    print(f"\nResults saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
