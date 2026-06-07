"""
CellTypeClassifier — classification head on top of InstanSeg's shared encoder.

Architecture:
    InstanSeg_UNet Encoder (partially frozen)
            ↓
    Bottleneck features  [B, 256, H/8, W/8]
            ↓
    Global Average Pooling  →  [B, 256]
            ↓
    Dropout(0.3)
            ↓
    FC(256, 128)  +  ReLU  +  Dropout(0.2)
            ↓
    FC(128, num_classes)  →  logits
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from classification.config import (
    NUM_CLASSES, ENCODER_LAYERS, BOTTLENECK_DIM, DEFAULT_FREEZE_BLOCKS,
)


class ClassificationHead(nn.Module):
    """Lightweight head: GAP → FC → FC → logits."""

    def __init__(self, in_features: int = BOTTLENECK_DIM,
                 num_classes: int = NUM_CLASSES,
                 dropout: float = 0.3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.66),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → logits: (B, num_classes)"""
        x = self.gap(x).flatten(1)   # (B, C)
        return self.classifier(x)


class CellTypeClassifier(nn.Module):
    """Full classifier: InstanSeg encoder + classification head."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        layers: list = None,
        norm: str = "BATCH",
        freeze_blocks: int = DEFAULT_FREEZE_BLOCKS,
        dropout: float = 0.3,
    ):
        super().__init__()
        if layers is None:
            layers = ENCODER_LAYERS

        # Build a fresh InstanSeg_UNet just for its encoder
        from instanseg.utils.models.InstanSeg_UNet import InstanSeg_UNet

        # We need some out_channels to instantiate the UNet,
        # but we won't use the decoder at all.
        dummy_out = [[1]]
        self._unet = InstanSeg_UNet(
            in_channels=3,
            out_channels=dummy_out,
            layers=np.array(layers)[::-1],
            norm=norm,
        )
        # We only keep the encoder
        self.encoder = self._unet.encoder
        del self._unet.decoders
        del self._unet

        self.head = ClassificationHead(
            in_features=layers[-1],   # bottleneck dim (256)
            num_classes=num_classes,
            dropout=dropout,
        )

        # Freeze encoder blocks
        self._freeze_blocks(freeze_blocks)

    def _freeze_blocks(self, n_blocks: int):
        """Freeze the first `n_blocks` encoder blocks."""
        for i, block in enumerate(self.encoder):
            if i < n_blocks:
                for param in block.parameters():
                    param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze everything (for fine-tuning the full model)."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → logits: (B, num_classes)"""
        # Run through encoder blocks
        for block in self.encoder:
            x = block(x)
        return self.head(x)


# Checkpoint loading utilities

def load_encoder_weights(
    model: CellTypeClassifier,
    checkpoint_path: str,
    device: str = "cpu",
    strict: bool = False,
) -> CellTypeClassifier:
    """Load encoder weights from a pre-trained InstanSeg segmentation checkpoint.

    The checkpoint is expected to have:
        checkpoint["model_state_dict"] with keys like "encoder.0.conv0.0.weight", ...

    Only encoder.* keys are loaded; decoder/pixel_classifier keys are ignored.
    """
    ckpt_path = Path(checkpoint_path)

    # Support both a folder (look for model_weights_best.pth) and a direct .pth file
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "model_weights_best.pth"

    print(f"Loading encoder weights from: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Filter to encoder keys only
    encoder_state = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix if present (DataParallel)
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key.startswith("encoder."):
            encoder_state[clean_key] = value

    # Load into model
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)

    # Filter out expected missing keys (head.*)
    real_missing = [k for k in missing if not k.startswith("head.")]
    if real_missing:
        print(f"[WARNING] Missing encoder keys: {real_missing}")
    loaded_count = len(encoder_state)
    print(f"Loaded {loaded_count} encoder parameters (ignored decoder/head keys)")

    return model


def save_classifier_checkpoint(
    model: CellTypeClassifier,
    optimizer,
    epoch: int,
    best_metric: float,
    save_path: str,
):
    """Save a classifier checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }, save_path)


def load_classifier_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    num_classes: int = NUM_CLASSES,
    layers: list = None,
    freeze_blocks: int = 0,
) -> tuple:
    """Load a full classifier checkpoint (for evaluation / inference)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CellTypeClassifier(
        num_classes=num_classes,
        layers=layers,
        freeze_blocks=freeze_blocks,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
