import math
from typing import List
from sympy import integer_nthroot

# Import the utility module as a namespace
from . import utils

# ==========================================
# 1. Analytic Features
# ==========================================

def log_magnitude(seq: List[int]) -> List[float]:
    """Computes log(1 + |x|)."""
    return [math.log1p(abs(x)) if x != 0 else 0.0 for x in seq]

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
    """Computes log(1 + |x_n - x_{n-1}|)."""
    diffs = [0.0] * len(seq)
    for i in range(1, len(seq)):
        raw_diff = abs(seq[i] - seq[i-1])
        diffs[i] = math.log1p(raw_diff)
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
    # FIX: Do not use abs(x) here. Negative numbers are not squares.
    # utils.is_square handles negative checks.
    return [1.0 if utils.is_square(x) else 0.0 for x in seq]

def is_cube(seq: List[int]) -> List[float]:
    res = []
    for x in seq:
        # For cubes, x^3 preserves sign. Checking abs(x) is sufficient
        # because if |x| is a cube k^3, then x is a cube of (sgn(x)*k).
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