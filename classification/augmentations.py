"""
Augmentation transforms for cell-type classification.

Designed for microscopy images. Uses torchvision functional transforms
so everything stays as plain torch tensors (no PIL round-trips during training).
"""

import random
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop, Resize, InterpolationMode
from classification.config import DEFAULT_TILE_SIZE


class TrainTransform:
    """Training augmentations: random crop, flips, rotation, colour jitter, noise.
    Input / output: (C, H, W) float32 tensor in [0, 1]."""

    def __init__(self, crop_size: int = DEFAULT_TILE_SIZE):
        self.crop_size = crop_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape

        # ── Pad if smaller than crop size ────────────────────────────
        pad_h = max(0, self.crop_size - h)
        pad_w = max(0, self.crop_size - w)
        if pad_h > 0 or pad_w > 0:
            img = torch.nn.functional.pad(
                img,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                mode="reflect",
            )

        # ── Random crop ──────────────────────────────────────────────
        _, h, w = img.shape
        if h > self.crop_size or w > self.crop_size:
            i, j, th, tw = RandomCrop.get_params(
                img, output_size=(self.crop_size, self.crop_size)
            )
            img = img[:, i : i + th, j : j + tw]

        # ── Random horizontal & vertical flips ───────────────────────
        if random.random() > 0.5:
            img = TF.hflip(img)
        if random.random() > 0.5:
            img = TF.vflip(img)

        # ── Random 90° rotation ──────────────────────────────────────
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            img = TF.rotate(img, angle)

        # ── Brightness / contrast jitter ─────────────────────────────
        if random.random() > 0.3:
            factor = 1.0 + (random.random() - 0.5) * 0.4  # [0.8, 1.2]
            img = torch.clamp(img * factor, 0.0, 1.0)

        # ── Gaussian noise ───────────────────────────────────────────
        if random.random() > 0.5:
            noise = torch.randn_like(img) * 0.02
            img = torch.clamp(img + noise, 0.0, 1.0)

        # ── Percentile normalization (per-channel) ───────────────────
        img = _percentile_normalize(img)

        return img


class ValTransform:
    """Validation / test transforms: deterministic centre crop + normalize.
    Input / output: (C, H, W) float32 tensor in [0, 1]."""

    def __init__(self, crop_size: int = DEFAULT_TILE_SIZE):
        self.crop_size = crop_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape

        # ── Pad if smaller than crop size ────────────────────────────
        pad_h = max(0, self.crop_size - h)
        pad_w = max(0, self.crop_size - w)
        if pad_h > 0 or pad_w > 0:
            img = torch.nn.functional.pad(
                img,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                mode="reflect",
            )

        # ── Centre crop ──────────────────────────────────────────────
        _, h, w = img.shape
        if h > self.crop_size or w > self.crop_size:
            img = TF.center_crop(img, [self.crop_size, self.crop_size])

        # ── Percentile normalization ─────────────────────────────────
        img = _percentile_normalize(img)

        return img


class TTATransform:
    """Test-time augmentation: returns a batch of 4 variants
    (original + 3 rotations). Average predictions over them."""

    def __init__(self, crop_size: int = DEFAULT_TILE_SIZE):
        self.val = ValTransform(crop_size)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        base = self.val(img)
        variants = [base]
        for angle in [90, 180, 270]:
            variants.append(TF.rotate(base, angle))
        return torch.stack(variants)  # (4, C, H, W)


def _percentile_normalize(
    img: torch.Tensor,
    lower: float = 0.01,
    upper: float = 0.99,
) -> torch.Tensor:
    """Per-channel percentile normalization to [0, 1]."""
    for c in range(img.shape[0]):
        ch = img[c]
        lo = torch.quantile(ch, lower)
        hi = torch.quantile(ch, upper)
        if hi - lo > 1e-6:
            img[c] = (ch - lo) / (hi - lo)
        else:
            img[c] = ch - lo
    return torch.clamp(img, 0.0, 1.0)
