"""
config.py:
Centralized configuration and constant definitions for IntSeqBERT.
Eliminates magic numbers and ensures consistency across modules.
"""

# ==========================================
# 1. File & Directory Configuration
# ==========================================
# Root directory for all data
DATA_ROOT = "data/oeis"

# Standard filenames and directory names
JSONL_FILENAME = "data.jsonl"
FEATURES_DIR_NAME = "features"
SPLIT_DIR_NAME = "splits"

# Split Types (Corresponds to subdirectory names under splits/)
# Example path: data/oeis/splits/strict/train.txt
SPLIT_STRICT = "strict"
SPLIT_EASY = "easy"
SPLIT_CLEAN = "clean"
SPLIT_ALL = "all"

# ==========================================
# 2. Numeric Parameters (Training/Split)
# ==========================================
# Global random seed for reproducibility
SEED = 42

# Default dataset split ratios
VAL_RATIO = 0.05
TEST_RATIO = 0.05

# ==========================================
# 3. Model Structure & Constants
# ==========================================
# Moduli used for the modular arithmetic stream (2 to 101)
MOD_RANGE = list(range(2, 102))

# Default hidden dimension for Transformer models (Encoder/Decoder)
D_MODEL_DEFAULT = 512

# Maximum sequence length for input features
MAX_SEQUENCE_LENGTH = 128

# Minimum sequence length to process. 
# Sequences shorter than this will be skipped during feature extraction.
MIN_SEQUENCE_LENGTH = 10

# ==========================================
# 4. Data Keys (for .pt files)
# ==========================================
KEY_OEIS_ID = "oeis_id"
KEY_MAG_FEATURES = "mag_features"
KEY_MOD_FEATURES = "mod_features"
KEY_MOD_INTEGERS = "mod_integers"
KEY_TARGETS = "targets"

# ==========================================
# 5. Input/Output Dimensions
# ==========================================
# Magnitude Stream Dimensions
# Raw input: [log_abs, sign+, sign-, sign0]
MAG_RAW_DIM = 4
# Model input (Expanded): [log_abs, sign+, sign-, sign0, is_masked]
MAG_EXTENDED_DIM = 5

# Modulo Stream Dimensions
# Calculated dynamically: len(MOD_RANGE) -> 100
NUM_MODULI = len(MOD_RANGE)
# Input features: (sin, cos) pair for each modulus -> 200
MOD_FEATURE_DIM = NUM_MODULI * 2

# ==========================================
# 6. Training / Collator Constants
# ==========================================
# Masking probability for dynamic masking (BERT standard is 0.15)
MASK_PROB = 0.15

# Value to ignore in CrossEntropyLoss (PyTorch standard)
IGNORE_INDEX = -100

# Padding value for floating point feature tensors
PAD_VALUE_FEATURE = 0.0

# ==========================================
# 7. Model Hyperparameters
# ==========================================
# Default model architecture parameters
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 6
DROPOUT = 0.1

# Feedforward layer hidden dim multiplier (d_model * this value)
FEEDFORWARD_MULTIPLIER = 4

# Sign classification classes
NUM_SIGN_CLASSES = 3
SIGN_POSITIVE = 0  # sign+ column in features
SIGN_NEGATIVE = 1  # sign- column in features
SIGN_ZERO = 2      # sign0 column in features

# Sinusoidal positional encoding base frequency
# Controls wavelength range: from 2π to 2π * (base^1) at the last dimension
POSITIONAL_ENCODING_BASE = 10000.0

# Fixed loss weights to prevent task collapse
# Mag : Sign : Mod = 1.0 : 1.0 : 2.0
LOSS_WEIGHT_MAG = 1.0
LOSS_WEIGHT_SIGN = 1.0
LOSS_WEIGHT_MOD = 2.0

# Representative moduli for console output (subset of MOD_RANGE)
# Used for logging during training to avoid printing all 100 mod accuracies
REPRESENTATIVE_MODS = [2, 3, 5, 7, 10, 100, 101]

# ==========================================
# 8. Training Hyperparameters
# ==========================================
# Magnitude accuracy threshold (log10 scale)
# 0.5 means predictions within sqrt(10) ≈ 3.16x of actual value
MAG_ACC_THRESHOLD = 0.5

# AdamW optimizer betas
ADAMW_BETAS = (0.9, 0.98)

# Gradient clipping norm
GRAD_CLIP_NORM = 1.0

# Default training parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_PATIENCE = 5
DEFAULT_NUM_WORKERS = 4
DEFAULT_EPOCHS = 20