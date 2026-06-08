"""
Unified segmentation + classification pipeline.

Runs a single forward pass through the shared InstanSeg encoder,
then feeds the bottleneck features into two heads in parallel:
  1. The segmentation decoder (original InstanSeg UNet decoder)
  2. The classification head (GAP + FC)

No retraining needed: it loads the full segmentation checkpoint
and the trained classification head checkpoint separately.
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from classification.config import (
    CLASS_NAMES, NUM_CLASSES, IDX_TO_CLASS,
    ENCODER_LAYERS, BOTTLENECK_DIM, DEFAULT_LONG_SIDE,
)
from classification.model import ClassificationHead


class UnifiedSegCls(nn.Module):
    """Unified model: shared encoder -> segmentation decoder + classification head.

    Architecture:
        Input Image
            |
        [Shared Encoder]  -- single forward pass
            |           \\
            |            \\
        [Seg Decoder]   [Cls Head]
            |               |
        Seg Masks       Class Logits
    """

    def __init__(
        self,
        instanseg_unet: nn.Module,
        cls_head: ClassificationHead,
    ):
        super().__init__()
        self.encoder = instanseg_unet.encoder
        self.decoders = instanseg_unet.decoders
        self.cls_head = cls_head

    def forward(
        self, x: torch.Tensor,
        run_seg: bool = True,
        run_cls: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: (B, C, H, W) input images
            run_seg: whether to run the segmentation decoder
            run_cls: whether to run the classification head

        Returns:
            seg_output: (B, seg_channels, H, W) or None
            cls_logits: (B, num_classes) or None
        """
        # Shared encoder forward pass
        skips = []
        for n, layer in enumerate(self.encoder):
            x = layer(x)
            if n < len(self.encoder) - 1:
                skips.append(x)

        bottleneck = x  # (B, 256, H/8, W/8)

        # Segmentation decoder
        seg_output = None
        if run_seg:
            seg_output = torch.cat(
                [decoder(bottleneck, skips) for decoder in self.decoders],
                dim=1,
            )

        # Classification head
        cls_logits = None
        if run_cls:
            cls_logits = self.cls_head(bottleneck)

        return seg_output, cls_logits

    def forward_cls_only(self, x: torch.Tensor) -> torch.Tensor:
        """Classification-only forward pass (encoder + cls head)."""
        for block in self.encoder:
            x = block(x)
        return self.cls_head(x)

    def forward_seg_only(self, x: torch.Tensor) -> torch.Tensor:
        """Segmentation-only forward pass (encoder + seg decoder)."""
        skips = []
        for n, layer in enumerate(self.encoder):
            x = layer(x)
            if n < len(self.encoder) - 1:
                skips.append(x)
        return torch.cat(
            [decoder(x, skips) for decoder in self.decoders],
            dim=1,
        )


def load_unified_model(
    seg_checkpoint_path: str,
    cls_checkpoint_path: str,
    device: str = "cpu",
    num_classes: int = NUM_CLASSES,
    layers: list = None,
) -> UnifiedSegCls:
    """Load the unified model from two separate checkpoints.

    Args:
        seg_checkpoint_path: Path to the full InstanSeg segmentation checkpoint
                             (folder or .pth file containing the full UNet weights).
        cls_checkpoint_path: Path to the trained classifier checkpoint (.pth)
                             containing model_state_dict with encoder.* and head.* keys.
        device: Target device.
        num_classes: Number of classification classes.
        layers: Encoder layer sizes.

    Returns:
        UnifiedSegCls model with both heads loaded, on the specified device.
    """
    if layers is None:
        layers = ENCODER_LAYERS

    # 1. Build and load the full InstanSeg UNet (encoder + seg decoder)
    from instanseg.utils.models.InstanSeg_UNet import InstanSeg_UNet

    seg_path = Path(seg_checkpoint_path)
    if seg_path.is_dir():
        seg_path = seg_path / "model_weights_best.pth"

    seg_ckpt = torch.load(str(seg_path), map_location=device, weights_only=False)
    seg_state = seg_ckpt["model_state_dict"]

    # Determine decoder output channels from the state dict
    # Find all final_block keys to infer out_channels structure
    decoder_keys = [k for k in seg_state.keys() if "decoders" in k]
    # Count number of decoders
    decoder_indices = set()
    for k in decoder_keys:
        # e.g. "decoders.0.decoder.0.conv0.0.weight"
        parts = k.split(".")
        if len(parts) > 1:
            try:
                decoder_indices.add(int(parts[1]))
            except ValueError:
                pass

    n_decoders = max(decoder_indices) + 1 if decoder_indices else 1

    # For each decoder, find final_block output channels
    out_channels = []
    for d_idx in range(n_decoders):
        fb_keys = [k for k in seg_state.keys()
                    if k.startswith(f"decoders.{d_idx}.final_block.") and k.endswith(".0.weight")]
        block_outs = []
        for fb_key in sorted(fb_keys):
            block_outs.append(seg_state[fb_key].shape[0])
        out_channels.append(block_outs if block_outs else [1])

    unet = InstanSeg_UNet(
        in_channels=3,
        out_channels=out_channels,
        layers=np.array(layers)[::-1],
        norm="BATCH",
    )

    # Load the full UNet state dict
    clean_state = {}
    for key, value in seg_state.items():
        clean_key = key[7:] if key.startswith("module.") else key
        clean_state[clean_key] = value

    unet.load_state_dict(clean_state, strict=False)
    print(f"Loaded segmentation UNet from: {seg_path}")

    # 2. Build and load the classification head
    cls_ckpt = torch.load(str(cls_checkpoint_path), map_location=device, weights_only=False)
    cls_state = cls_ckpt["model_state_dict"]

    cls_head = ClassificationHead(
        in_features=layers[-1],
        num_classes=num_classes,
    )

    # Extract head.* keys from classifier checkpoint
    head_state = {}
    for key, value in cls_state.items():
        if key.startswith("head."):
            # Remove 'head.' prefix since we're loading into the head directly
            head_state[key[5:]] = value

    cls_head.load_state_dict(head_state)
    print(f"Loaded classification head from: {cls_checkpoint_path}")

    # 3. Assemble unified model
    model = UnifiedSegCls(unet, cls_head)
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict_unified(
    model: UnifiedSegCls,
    image_tensor: torch.Tensor,
    device: torch.device,
    run_seg: bool = True,
    run_cls: bool = True,
) -> dict:
    """Run unified inference on a single preprocessed image tensor.

    Args:
        model: Loaded UnifiedSegCls model.
        image_tensor: (1, 3, H, W) preprocessed tensor.
        device: Target device.
        run_seg: Whether to produce segmentation output.
        run_cls: Whether to produce classification output.

    Returns:
        dict with keys:
            - "seg_output": (1, seg_channels, H, W) tensor or None
            - "predicted_class": str or None
            - "confidence": float or None
            - "probabilities": dict {class_name: prob} or None
            - "time_ms": float (wall-clock time for the forward pass)
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    seg_out, cls_logits = model(image_tensor, run_seg=run_seg, run_cls=run_cls)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    result = {"time_ms": (t1 - t0) * 1000}

    if seg_out is not None:
        result["seg_output"] = seg_out.cpu()
    else:
        result["seg_output"] = None

    if cls_logits is not None:
        probs = torch.softmax(cls_logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(probs.argmax())
        result["predicted_class"] = IDX_TO_CLASS[pred_idx]
        result["confidence"] = float(probs[pred_idx])
        result["probabilities"] = {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        }
    else:
        result["predicted_class"] = None
        result["confidence"] = None
        result["probabilities"] = None

    return result
