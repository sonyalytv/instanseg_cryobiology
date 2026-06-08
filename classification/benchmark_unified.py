"""
Benchmark script for the unified segmentation + classification pipeline.

Compares inference timing across three modes:
  1. Classification only  (encoder + cls head)
  2. Segmentation only    (encoder + seg decoder)
  3. Unified              (encoder once + both heads)

Also generates overlay visualizations combining segmentation contours
with predicted class labels.

Usage:
    python -m classification.benchmark_unified \
        --data_path    "/path/to/by_type" \
        --seg_checkpoint "/path/to/my_instanseg" \
        --cls_checkpoint "/path/to/model_best.pth" \
        --output_path  "./benchmark_results"
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from classification.config import (
    CLASS_NAMES, NUM_CLASSES, IDX_TO_CLASS,
    DEFAULT_LONG_SIDE, DEFAULT_TILE_SIZE,
    DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT, DEFAULT_RANDOM_SEED,
    ENCODER_LAYERS,
)
from classification.dataset import (
    collect_image_paths, _read_image, _to_3channel_float, _resize_long_side,
)
from classification.augmentations import ValTransform
from classification.unified import UnifiedSegCls, load_unified_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the unified seg+cls pipeline."
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root folder with class subfolders")
    parser.add_argument("--seg_checkpoint", type=str, required=True,
                        help="Path to the full InstanSeg segmentation checkpoint "
                             "(folder or .pth)")
    parser.add_argument("--cls_checkpoint", type=str, required=True,
                        help="Path to the trained classifier checkpoint (.pth)")
    parser.add_argument("--output_path", type=str, default="./benchmark_results",
                        help="Where to save benchmark results")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test", "all"])
    parser.add_argument("--long_side", type=int, default=DEFAULT_LONG_SIDE)
    parser.add_argument("--tile_size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--val_split", type=float, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--test_split", type=float, default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_vis", type=int, default=12,
                        help="Number of overlay visualizations to generate")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations before timing")
    return parser.parse_args()


def _choose_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _preprocess_image(path: str, long_side: int, tile_size: int) -> torch.Tensor:
    """Read and preprocess a single image to a (1, 3, H, W) tensor."""
    img = _read_image(path)
    img = _to_3channel_float(img)
    img = _resize_long_side(img, long_side)
    tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
    transform = ValTransform(tile_size)
    tensor = transform(tensor)
    return tensor.unsqueeze(0)  # (1, 3, H, W)


def _seg_to_contours(seg_output: torch.Tensor) -> np.ndarray:
    """Extract boundary mask from a segmentation label map.
    seg_output: (1, C, H, W) where each channel is an integer label map.
    Returns a (H, W) boolean mask of cell boundaries."""
    # Take the first channel (nuclei or cells)
    labels = seg_output[0, 0].numpy().astype(np.int32)
    h, w = labels.shape

    # Simple boundary detection: pixel differs from any neighbour
    boundary = np.zeros((h, w), dtype=bool)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(labels, dy, axis=0), dx, axis=1)
        boundary |= (labels != shifted) & (labels > 0)

    return boundary


@torch.no_grad()
def _benchmark_mode(model, tensors, device, mode, warmup=5):
    """Benchmark a specific forward-pass mode over a list of tensors.

    Args:
        model: UnifiedSegCls model
        tensors: list of (1, 3, H, W) tensors
        device: torch.device
        mode: "cls_only", "seg_only", or "unified"
        warmup: number of warmup iterations

    Returns:
        (total_time_s, avg_ms_per_image)
    """
    # Warmup
    dummy = tensors[0].to(device)
    for _ in range(warmup):
        if mode == "cls_only":
            model.forward_cls_only(dummy)
        elif mode == "seg_only":
            model.forward_seg_only(dummy)
        else:
            model(dummy, run_seg=True, run_cls=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t_start = time.time()
    for t in tensors:
        t_gpu = t.to(device)
        if mode == "cls_only":
            model.forward_cls_only(t_gpu)
        elif mode == "seg_only":
            model.forward_seg_only(t_gpu)
        else:
            model(t_gpu, run_seg=True, run_cls=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.time()

    elapsed = t_end - t_start
    avg_ms = (elapsed / len(tensors) * 1000) if tensors else 0.0
    return elapsed, avg_ms


def _plot_overlay_grid(images_raw, seg_outputs, cls_results, save_path, num_vis):
    """Plot a grid of images with segmentation contours and classification labels."""
    n = min(num_vis, len(images_raw))
    if n == 0:
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx in range(n):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        img = images_raw[idx]
        ax.imshow(np.clip(img, 0, 1))

        # Overlay segmentation contours if available
        if seg_outputs[idx] is not None:
            boundary = _seg_to_contours(seg_outputs[idx])
            # Create red overlay for boundaries
            overlay = np.zeros((*boundary.shape, 4), dtype=np.float32)
            overlay[boundary, 0] = 1.0  # Red channel
            overlay[boundary, 3] = 0.8  # Alpha
            ax.imshow(overlay)

        # Title with classification result
        r = cls_results[idx]
        if r["true_class"] is not None:
            is_correct = r["predicted_class"] == r["true_class"]
            color = "#2e8b57" if is_correct else "#c0392b"
            marker = "[OK]" if is_correct else "[X]"
            title = (f"{marker} True: {r['true_class']}\n"
                     f"Pred: {r['predicted_class']} ({r['confidence']:.2f})")
        else:
            color = "#333333"
            title = f"Pred: {r['predicted_class']} ({r['confidence']:.2f})"

        ax.set_title(title, fontsize=10, fontweight="bold", color=color)
        ax.axis("off")

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    plt.suptitle("Unified Pipeline: Segmentation + Classification",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Overlay grid saved to: {save_path}")


def main():
    args = parse_args()
    device = _choose_device(args.device)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load unified model
    print("Loading unified model...")
    model = load_unified_model(
        seg_checkpoint_path=args.seg_checkpoint,
        cls_checkpoint_path=args.cls_checkpoint,
        device=str(device),
    )

    # Collect and split data
    all_paths, all_labels = collect_image_paths(args.data_path)
    if len(all_paths) == 0:
        print("[ERROR] No images found.")
        sys.exit(1)

    if args.split in ["val", "test"]:
        if args.test_split > 0:
            train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
                all_paths, all_labels, test_size=args.test_split,
                stratify=all_labels, random_state=args.seed,
            )
        else:
            train_val_paths, train_val_labels = all_paths, all_labels
            test_paths, test_labels = [], []

        if args.split == "test":
            eval_paths, eval_labels = test_paths, test_labels
        else:
            rel_val = args.val_split / (1.0 - args.test_split) if args.test_split < 1 else args.val_split
            _, eval_paths, _, eval_labels = train_test_split(
                train_val_paths, train_val_labels,
                test_size=rel_val, stratify=train_val_labels,
                random_state=args.seed,
            )
    else:
        eval_paths, eval_labels = all_paths, all_labels

    print(f"\nBenchmarking on {len(eval_paths)} images (split={args.split})")

    # Preprocess all images
    print("Preprocessing images...")
    tensors = []
    images_raw = []
    for p in eval_paths:
        try:
            img = _read_image(p)
            img = _to_3channel_float(img)
            img = _resize_long_side(img, args.long_side)
            images_raw.append(img)
            t = torch.from_numpy(img).permute(2, 0, 1)
            transform = ValTransform(args.tile_size)
            t = transform(t).unsqueeze(0)
            tensors.append(t)
        except Exception as e:
            print(f"  [SKIP] {p}: {e}")

    if not tensors:
        print("[ERROR] No images could be preprocessed.")
        sys.exit(1)

    # Benchmark three modes
    print(f"\nBenchmarking (warmup={args.warmup} iters)...")
    print("-" * 60)

    cls_total, cls_avg = _benchmark_mode(model, tensors, device, "cls_only", args.warmup)
    print(f"  Classification only:  {cls_avg:8.2f} ms/image  (total {cls_total:.2f}s)")

    seg_total, seg_avg = _benchmark_mode(model, tensors, device, "seg_only", args.warmup)
    print(f"  Segmentation only:    {seg_avg:8.2f} ms/image  (total {seg_total:.2f}s)")

    uni_total, uni_avg = _benchmark_mode(model, tensors, device, "unified", args.warmup)
    print(f"  Unified (seg+cls):    {uni_avg:8.2f} ms/image  (total {uni_total:.2f}s)")

    # Calculate the overhead of running both vs. the slower single mode
    separate_total = cls_avg + seg_avg
    savings = separate_total - uni_avg
    print("-" * 60)
    print(f"  Running separately:   {separate_total:8.2f} ms/image")
    print(f"  Unified savings:      {savings:8.2f} ms/image "
          f"({savings/separate_total*100:.1f}% faster)")

    # Save timing results
    timing_df = pd.DataFrame({
        "mode": ["cls_only", "seg_only", "unified", "separate_sum"],
        "total_time_s": [cls_total, seg_total, uni_total, cls_total + seg_total],
        "avg_ms_per_image": [cls_avg, seg_avg, uni_avg, cls_avg + seg_avg],
        "num_images": [len(tensors)] * 4,
    })
    timing_df.to_csv(output_path / "benchmark_timing.csv", index=False)
    print(f"\n  Timing results saved to: {output_path / 'benchmark_timing.csv'}")

    # Generate overlay visualizations on a subset
    print(f"\nGenerating overlay visualizations ({args.num_vis} samples)...")
    vis_indices = np.random.RandomState(42).choice(
        len(tensors), size=min(args.num_vis, len(tensors)), replace=False
    )

    vis_images = []
    vis_segs = []
    vis_cls = []

    for idx in vis_indices:
        t_gpu = tensors[idx].to(device)
        seg_out, cls_logits = model(t_gpu, run_seg=True, run_cls=True)

        # Process segmentation (run through InstanSeg postprocessing if available)
        seg_cpu = seg_out.cpu() if seg_out is not None else None

        # Process classification
        probs = torch.softmax(cls_logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(probs.argmax())

        true_label = eval_labels[idx] if idx < len(eval_labels) else None
        true_class = IDX_TO_CLASS[true_label] if true_label is not None else None

        vis_images.append(images_raw[idx])
        vis_segs.append(seg_cpu)
        vis_cls.append({
            "predicted_class": IDX_TO_CLASS[pred_idx],
            "confidence": float(probs[pred_idx]),
            "true_class": true_class,
        })

    _plot_overlay_grid(vis_images, vis_segs, vis_cls,
                       save_path=output_path / "unified_samples.png",
                       num_vis=args.num_vis)

    # Plot timing bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    modes = ["Classification\nOnly", "Segmentation\nOnly", "Unified\n(Seg+Cls)"]
    times = [cls_avg, seg_avg, uni_avg]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    bars = ax.bar(modes, times, color=colors, edgecolor="white", linewidth=1.5)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{t:.1f} ms", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Average Time per Image (ms)", fontsize=12)
    ax.set_title("Inference Speed Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "benchmark_chart.png", dpi=150)
    plt.close()
    print(f"  Benchmark chart saved to: {output_path / 'benchmark_chart.png'}")

    print(f"\nAll results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
