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