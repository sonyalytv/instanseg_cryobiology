"""
PyTorch Dataset for cell-type classification.

Reads images from class-labelled subfolders:
    data_root/
        epithelial/
        fibroblasts/
        leukocytes/
        neuroblasts/

Skips unsupported formats (.zvi).
Supports: .jpg .jpeg .png .tif .tiff .bmp
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import tifffile

from classification.config import (
    CLASS_NAMES, CLASS_TO_IDX, SUPPORTED_EXTENSIONS,
    DEFAULT_LONG_SIDE, MODEL_INPUT_CHANNELS,
)


def _read_image(path: str) -> np.ndarray:
    """Read an image file and return it as a numpy array (H, W, C) or (H, W).
    Uses tifffile for .tif/.tiff, PIL for everything else."""
    ext = Path(path).suffix.lower()
    if ext in (".tif", ".tiff"):
        img = tifffile.imread(path)
    else:
        img = np.array(Image.open(path))
    return img


def _to_3channel_float(img: np.ndarray) -> np.ndarray:
    """Ensure image is (H, W, 3) float32 in [0, 1]."""
    # Handle different dtypes
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.float32) / img.max() if img.max() > 0 else img.astype(np.float32)
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / img.max()

    # Handle channel dimension
    if img.ndim == 2:
        # Grayscale → repeat to 3 channels
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        if img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[-1]:
            # Channel-first -> channel-last
            img = np.transpose(img, (1, 2, 0))
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]  # Drop alpha
        elif img.shape[-1] != 3:
            # Multi-channel fluorescence: take first 3 or pad
            if img.shape[-1] > 3:
                img = img[..., :3]
            else:
                pad = np.zeros((*img.shape[:2], 3 - img.shape[-1]), dtype=img.dtype)
                img = np.concatenate([img, pad], axis=-1)

    return img.astype(np.float32)


def _resize_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    """Resize so the longest side equals `long_side`, preserving aspect ratio.
    Input/output: (H, W, C) float32."""
    h, w = img.shape[:2]
    if max(h, w) == long_side:
        return img
    if h >= w:
        new_h = long_side
        new_w = int(round(w * long_side / h))
    else:
        new_w = long_side
        new_h = int(round(h * long_side / w))

    pil_img = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    return np.array(pil_img).astype(np.float32) / 255.0


def collect_image_paths(data_root: str) -> Tuple[List[str], List[int]]:
    """Walk class subfolders and collect (path, class_idx) pairs.
    Skips unsupported extensions."""
    paths, labels = [], []
    data_root = Path(data_root)

    for class_name in CLASS_NAMES:
        class_dir = data_root / class_name
        if not class_dir.is_dir():
            print(f"[WARNING] Class folder not found: {class_dir}")
            continue
        class_idx = CLASS_TO_IDX[class_name]
        count = 0
        for fpath in sorted(class_dir.rglob("*")):
            if fpath.is_file() and fpath.suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(str(fpath))
                labels.append(class_idx)
                count += 1
        print(f"  {class_name}: {count} images")

    print(f"Total: {len(paths)} images across {len(CLASS_NAMES)} classes")
    return paths, labels


class CellTypeDataset(Dataset):
    """Dataset for cell-type classification from folder structure."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        long_side: int = DEFAULT_LONG_SIDE,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            image_paths: list of absolute paths to image files.
            labels: list of integer class labels (same length).
            long_side: resize longest side to this value.
            transform: optional callable (image_tensor) -> image_tensor
                       for augmentation. Receives (C, H, W) float tensor.
        """
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels = labels
        self.long_side = long_side
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        label = self.labels[idx]

        # Read and preprocess
        try:
            img = _read_image(path)
            img = _to_3channel_float(img)
            img = _resize_long_side(img, self.long_side)
        except Exception as e:
            print(f"[ERROR] Failed to read {path}: {e}. Returning black image.")
            img = np.zeros((self.long_side, self.long_side, 3), dtype=np.float32)

        # HWC -> CHW tensor
        tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

        # Apply augmentations
        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label
