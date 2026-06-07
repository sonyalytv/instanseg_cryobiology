"""
Central configuration for the cell-type classification pipeline.
All paths are passed via CLI — this file only stores class metadata
and default hyperparameters so the code is Kaggle-portable.
"""

# Class definitions (alphabetical order)
CLASS_NAMES = ["epithelial", "fibroblasts", "leukocytes", "neuroblasts"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Image preprocessing
DEFAULT_LONG_SIDE = 512        # Resize so the long side equals this
DEFAULT_TILE_SIZE = 256        # Crop size fed to the model during training
MODEL_INPUT_CHANNELS = 3       # InstanSeg encoder expects 3-channel input

# Encoder architecture (must match the checkpoint)
ENCODER_LAYERS = [32, 64, 128, 256]
BOTTLENECK_DIM = ENCODER_LAYERS[-1]   # 256

# Training defaults
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_EPOCHS = 100
DEFAULT_LR = 1e-4
DEFAULT_LR_ENCODER = 1e-5     # Differential LR for unfrozen encoder blocks
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 15         # Early stopping patience (epochs)
DEFAULT_FREEZE_BLOCKS = 3     # Freeze encoder blocks 0..2, unfreeze block 3
DEFAULT_NUM_WORKERS = 4
DEFAULT_VAL_SPLIT = 0.15      # Fraction for stratified validation split
DEFAULT_TEST_SPLIT = 0.15     # Fraction for stratified test split
DEFAULT_RANDOM_SEED = 42
