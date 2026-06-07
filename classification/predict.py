"""
Single-image or batch inference for the cell-type classifier.

Usage:
    # Single image
    python -m classification.predict \
        --image "/path/to/image.tif" \
        --checkpoint "/path/to/model_best.pth"

    # Folder of images
    python -m classification.predict \
        --image "/path/to/folder/" \
        --checkpoint "/path/to/model_best.pth"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from classification.config import (
    CLASS_NAMES, NUM_CLASSES, IDX_TO_CLASS,
    SUPPORTED_EXTENSIONS, DEFAULT_LONG_SIDE, DEFAULT_TILE_SIZE,
    ENCODER_LAYERS,
)
from classification.dataset import _read_image, _to_3channel_float, _resize_long_side
from classification.augmentations import ValTransform, TTATransform
from classification.model import load_classifier_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict cell type for a single image or folder of images."
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to an image file or folder of images")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained classifier checkpoint (.pth)")
    parser.add_argument("--tta", action="store_true", default=False,
                        help="Enable test-time augmentation")
    parser.add_argument("--long_side", type=int, default=DEFAULT_LONG_SIDE)
    parser.add_argument("--tile_size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _choose_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_single(
    model,
    image_path: str,
    device: torch.device,
    long_side: int = DEFAULT_LONG_SIDE,
    tile_size: int = DEFAULT_TILE_SIZE,
    use_tta: bool = False,
) -> dict:
    """Predict cell type for a single image.

    Returns:
        dict with keys: "path", "predicted_class", "confidence", "probabilities"
    """
    # Read & preprocess
    img = _read_image(image_path)
    img = _to_3channel_float(img)
    img = _resize_long_side(img, long_side)
    tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

    if use_tta:
        transform = TTATransform(tile_size)
        tensor = transform(tensor)  # (4, 3, H, W)
        tensor = tensor.to(device)
        logits = model(tensor)  # (4, num_classes)
        logits = logits.mean(dim=0, keepdim=True)  # (1, num_classes)
    else:
        transform = ValTransform(tile_size)
        tensor = transform(tensor).unsqueeze(0).to(device)  # (1, 3, H, W)
        logits = model(tensor)  # (1, num_classes)

    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(probs.argmax())

    return {
        "path": str(image_path),
        "predicted_class": IDX_TO_CLASS[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    }


def main():
    args = parse_args()
    device = _choose_device(args.device)

    # Load model
    model, ckpt = load_classifier_checkpoint(
        args.checkpoint, device=str(device), layers=ENCODER_LAYERS, freeze_blocks=0
    )
    print(f"Loaded model from epoch {ckpt.get('epoch', '?')}")

    # Collect image paths
    image_path = Path(args.image)
    if image_path.is_file():
        paths = [image_path]
    elif image_path.is_dir():
        paths = sorted([
            f for f in image_path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ])
        print(f"Found {len(paths)} images in {image_path}")
    else:
        print(f"[ERROR] Path not found: {args.image}")
        sys.exit(1)

    if not paths:
        print("[ERROR] No supported images found.")
        sys.exit(1)

    # Predict
    print(f"\n{'─'*60}")
    for p in paths:
        result = predict_single(
            model, str(p), device,
            long_side=args.long_side,
            tile_size=args.tile_size,
            use_tta=args.tta,
        )
        fname = Path(result["path"]).name
        pred = result["predicted_class"]
        conf = result["confidence"]
        probs_str = "  ".join(
            f"{cls}: {prob:.3f}"
            for cls, prob in result["probabilities"].items()
        )
        print(f"{fname:40s} → {pred:15s} ({conf:.3f})   [{probs_str}]")

    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
