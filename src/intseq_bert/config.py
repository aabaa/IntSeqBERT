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

# Default dataset split ratios (80:10:10 split)
VAL_RATIO = 0.10
TEST_RATIO = 0.10

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
# Using a sentinel value unlikely to appear in data (10^-9999).
PAD_VALUE_FEATURE = -9999.0

# ==========================================
# 7. Model Hyperparameters
# ==========================================
# Default model architecture parameters
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 8
DROPOUT = 0.2  # v3: Increased from 0.1 for regularization

# Feedforward layer hidden dim multiplier (d_model * this value)
FEEDFORWARD_MULTIPLIER = 4

# --- Model Architecture Config (v3) ---
# Input Projection Type: 'linear' or 'mlp'
# 'linear': Single Linear layer (5 -> d_model)
# 'mlp': Linear -> GELU -> Linear (v3 default, more expressive)
INPUT_PROJ_TYPE = 'mlp'

# Extra Dropout: Apply dropout to streams before FiLM fusion
USE_PRE_FILM_DROPOUT = True

# --- Loss Configuration (v3) ---
# Magnitude loss type: 'huber', 'mse', or 'l1'
MAG_LOSS_TYPE = 'huber'

# Whether to use Heteroscedastic Loss (Uncertainty Estimation)
# True: Gaussian NLL with learned variance (pred_log_var used)
# False: Simple deterministic loss (pred_log_var ignored)
USE_HETEROSCEDASTIC_LOSS = False

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

# ==========================================
# 9. Analysis Constants
# ==========================================
# Minimum samples for statistically reliable analysis
MIN_RELIABLE_SAMPLES = 30  # For scale-wise bucket analysis
MIN_TAG_SAMPLES = 10       # For tag-stratified analysis

# Log-linearity analysis (Growth Type)
# R² threshold to classify a sequence as "log-linear" (exponential growth)
LOG_LINEARITY_R2_THRESHOLD = 0.95

# Bootstrap CI parameters
BOOTSTRAP_SAMPLES_DEFAULT = 1000
CI_LEVEL_DEFAULT = 0.95

# Worst-K analysis
WORST_K_DEFAULT = 100

# Calibration binning
CALIBRATION_BINS_DEFAULT = 10

# Log variance clipping for numerical stability
LOG_VAR_CLIP_MIN = -10
LOG_VAR_CLIP_MAX = 10

# Tolerance accuracy thresholds (log10 scale)
# 0.5 -> ±3.16x, 0.1 -> ±1.26x, 0.05 -> ±1.12x
TOLERANCE_THRESHOLDS = [0.5, 0.1, 0.05]

# Magnitude scale buckets [low, high, name] (log10 scale)
MAGNITUDE_BUCKETS = [
    (0, 2, "Small"),       # 1 ~ 100
    (2, 5, "Medium"),      # 100 ~ 100,000
    (5, 20, "Large"),      # 10^5 ~ 10^20
    (20, 50, "Huge"),      # 10^20 ~ 10^50
    (50, float('inf'), "Astronomical"),  # 10^50+
]

# Base-10 related moduli (excluded from non_base10_acc calculation)
BASE10_RELATED_MODS = frozenset({10, 20, 50, 100})

# Numerical stability constants
EPSILON = 1e-6

# Plot defaults
SCATTER_SAMPLE_SIZE = 10000  # Max points for scatter plots
HISTOGRAM_BINS = 50

# ==========================================
# 10. Solver Constants
# ==========================================
# Mode switching thresholds
SOLVER_DENSE_THRESHOLD = 1_000_000           # Mode A (dense) → Mode AB (sieve)
SOLVER_SIEVE_THRESHOLD = 100_000_000_000_000  # Mode AB → Mode B (CRT), 10^14

# Anchored Sieve parameters
SOLVER_SIEVE_TARGET = 100_000   # Target candidate count after sieving
SOLVER_MAX_ANCHORS = 20         # Maximum number of anchor moduli

# Beam Search parameters
SOLVER_BEAM_WIDTH = 10          # Beam width for CRT candidate generation
SOLVER_TOP_K_DEFAULT = 5        # Default number of candidates to return

# Candidate enumeration safety limits (OOM prevention)
SOLVER_MAX_ENUM_CANDIDATES = 200_000  # Hard limit on enumerated candidates per solve
SOLVER_BEAM_SKIP_THRESHOLD = 10_000_000  # Skip beam if it would generate > this many candidates

# Solver Scoring Weights
SOLVER_MAG_WEIGHT = 1.0  # Magnitude score weight
SOLVER_MOD_WEIGHT = 0.3  # Modulo score weight (Discounted due to redundant moduli like 2, 4, 8)

# ==========================================
# 11. Vanilla Transformer Constants
# ==========================================
VANILLA_VOCAB_SIZE = 30_003     # Token vocabulary size (covers 0-29,999 + special tokens)
VANILLA_PAD_TOKEN_ID = 0        # Padding token ID
VANILLA_UNK_TOKEN_ID = 2        # Unknown token ID
VANILLA_MASK_TOKEN_ID = 1       # Mask token ID
VANILLA_SPECIAL_TOKENS_OFFSET = 3  # Number of special tokens (PAD, MASK, UNK); integers start at this ID