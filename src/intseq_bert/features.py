import math
from typing import List
from sympy import integer_nthroot
import numpy as np

# Import the utility module as a namespace
from . import utils

# ==========================================
# 1. Analytic Features
# ==========================================

def log_magnitude(seq: List[int]) -> List[float]:
    """
    Computes magnitude: 1 + log(|x|) for x != 0, else 0.0.
    This preserves linearity for exponential sequences while strictly separating 
    |x|=1 (val=1.0) from x=0 (val=0.0).
    """
    # Modified logic: 1 + log(abs(x))
    return [1.0 + math.log(abs(x)) if x != 0 else 0.0 for x in seq]

def sign(seq: List[int]) -> List[float]:
    """Computes sign of x: 1.0, -1.0, or 0.0."""
    return [1.0 if x > 0 else (-1.0 if x < 0 else 0.0) for x in seq]

def diff1(seq: List[int]) -> List[float]:
    """Computes 1st order difference of Log Magnitude."""
    logs = log_magnitude(seq)
    diffs = [0.0] * len(seq)
    for i in range(1, len(seq)):
        diffs[i] = logs[i] - logs[i-1]
    return diffs

def diff2(seq: List[int]) -> List[float]:
    """Computes 2nd order difference of Log Magnitude."""
    d1 = diff1(seq)
    diffs = [0.0] * len(seq)
    for i in range(1, len(seq)):
        diffs[i] = d1[i] - d1[i-1]
    return diffs

def direction(seq: List[int]) -> List[float]:
    """Computes direction of raw value change: 1.0, -1.0, 0.0."""
    dirs = [0.0] * len(seq)
    for i in range(1, len(seq)):
        diff = seq[i] - seq[i-1]
        if diff > 0:
            dirs[i] = 1.0
        elif diff < 0:
            dirs[i] = -1.0
    return dirs

def log_raw_diff(seq: List[int]) -> List[float]:
    """
    Computes sparsity: 1 + log(|diff|) for diff != 0, else 0.0.
    Distinguishes 'no change' (0.0) from 'step size 1' (1.0).
    """
    diffs = [0.0] * len(seq)
    for i in range(1, len(seq)):
        raw_diff = abs(seq[i] - seq[i-1])
        if raw_diff != 0:
            diffs[i] = 1.0 + math.log(raw_diff)
        else:
            diffs[i] = 0.0
    return diffs

# ==========================================
# 2. Algebraic Features (Atomic)
# ==========================================

def mod_sin(seq: List[int], m: int) -> List[float]:
    """Computes sin(2*pi * (x % m) / m)."""
    res = []
    scale = 2 * math.pi / m
    for x in seq:
        res.append(math.sin((x % m) * scale))
    return res

def mod_cos(seq: List[int], m: int) -> List[float]:
    """Computes cos(2*pi * (x % m) / m)."""
    res = []
    scale = 2 * math.pi / m
    for x in seq:
        res.append(math.cos((x % m) * scale))
    return res

# ==========================================
# 3. Number Theoretic Features (Atomic)
# ==========================================

def valuation(seq: List[int], p: int) -> List[float]:
    """Computes log(1 + v_p(x))."""
    res = []
    for x in seq:
        v = utils.valuation(x, p)
        res.append(math.log1p(v))
    return res

def is_zero(seq: List[int]) -> List[float]:
    return [1.0 if x == 0 else 0.0 for x in seq]

def is_square_free(seq: List[int]) -> List[float]:
    return [1.0 if utils.is_square_free(x) else 0.0 for x in seq]

def is_prime(seq: List[int]) -> List[float]:
    return [1.0 if utils.is_prime(abs(x)) else 0.0 for x in seq]

def is_square(seq: List[int]) -> List[float]:
    return [1.0 if utils.is_square(x) else 0.0 for x in seq]

def is_cube(seq: List[int]) -> List[float]:
    res = []
    for x in seq:
        _, exact = integer_nthroot(abs(x), 3)
        res.append(1.0 if exact else 0.0)
    return res

# ==========================================
# 4. Digital Features
# ==========================================

def popcount(seq: List[int]) -> List[float]:
    return [math.log1p(utils.popcount(x)) for x in seq]

def digit_sum(seq: List[int]) -> List[float]:
    return [math.log1p(utils.digit_sum(x)) for x in seq]

def is_power_of_2(seq: List[int]) -> List[float]:
    res = []
    for x in seq:
        if x <= 0:
            res.append(0.0)
        else:
            res.append(1.0 if (x & (x - 1) == 0) else 0.0)
    return res

def extract_features(seq: List[int]) -> np.ndarray:
    """
    Extracts all features for a given sequence.
    Returns:
        np.ndarray: Shape (SeqLen, 35), dtype=float32
    """
    if not seq:
        return np.zeros((0, 35), dtype=np.float32)

    features_list = []

    # 1. Analytic (6)
    features_list.append(log_magnitude(seq))
    features_list.append(sign(seq))
    features_list.append(diff1(seq))
    features_list.append(diff2(seq))
    features_list.append(direction(seq))
    features_list.append(log_raw_diff(seq))

    # 2. Valuation (3)
    for p in [2, 3, 5]:
        features_list.append(valuation(seq, p))

    # 3. Boolean (5)
    features_list.append(is_zero(seq))
    features_list.append(is_square_free(seq))
    features_list.append(is_prime(seq))
    features_list.append(is_square(seq))
    features_list.append(is_cube(seq))

    # 4. Digital (3)
    features_list.append(popcount(seq))
    features_list.append(digit_sum(seq))
    features_list.append(is_power_of_2(seq))

    # 5. Algebraic (18)
    bases = [3, 4, 5, 6, 7, 8, 11, 13, 100]
    for m in bases:
        features_list.append(mod_sin(seq, m))
        features_list.append(mod_cos(seq, m))

    # Convert to numpy array [35, SeqLen] -> Transpose -> [SeqLen, 35]
    # Use float32 to match PyTorch default expectation
    return np.array(features_list, dtype=np.float32).T

