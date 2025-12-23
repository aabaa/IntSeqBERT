import pytest
import math
from typing import List

# Import feature module from the package
from intseq_bert import features

# --- Helper function for float comparison ---
def assert_close(actual: List[float], expected: List[float], tol=1e-5):
    """
    Helper to verify that list elements are approximately equal.
    """
    assert len(actual) == len(expected), f"Length mismatch: {len(actual)} vs {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert math.isclose(a, e, abs_tol=tol), f"Mismatch at index {i}: {a} != {expected[i]}"

# ==========================================
# 1. Analytic Features Tests
# ==========================================

def test_analytic_basics():
    # 0, negative number, positive number
    seq = [0, -10, 100]
    
    # Log Magnitude: log(1 + |x|)
    # log(1+0)=0, log(1+10)=2.39..., log(1+100)=4.61...
    expected_log = [0.0, math.log(11), math.log(101)]
    assert_close(features.log_magnitude(seq), expected_log)
    
    # Sign: 1.0 (pos), -1.0 (neg), 0.0 (zero)
    expected_sign = [0.0, -1.0, 1.0]
    assert_close(features.sign(seq), expected_sign)

def test_direction():
    # Increase, Decrease, No change
    seq = [10, 20, 20, 15]
    # diff: [N/A, +10, 0, -5] -> [0.0, 1.0, 0.0, -1.0]
    expected = [0.0, 1.0, 0.0, -1.0]
    assert_close(features.direction(seq), expected)

def test_log_raw_diff():
    # Raw value differences (Sparsity)
    seq = [10, 10, 100]
    # diffs: [N/A, 0, 90]
    expected = [
        0.0, 
        math.log1p(0),   # log(1) = 0
        math.log1p(90)
    ]
    assert_close(features.log_raw_diff(seq), expected)

def test_diff_derivatives():
    # Exponential growth: 100, 1000, 10000
    # Larger numbers reduce the bias of log1p vs log, making the line straighter.
    seq = [100, 1000, 10000]
    logs = features.log_magnitude(seq)
    
    # Diff1 (Velocity)
    d1 = features.diff1(seq)
    assert d1[0] == 0.0
    assert math.isclose(d1[1], logs[1] - logs[0])
    
    # Diff2 (Acceleration)
    # For exponential growth, the log-plot is linear.
    # Therefore, Diff1 should be constant, and Diff2 should be close to 0.
    d2 = features.diff2(seq)
    assert d2[0] == 0.0
    
    # Check if acceleration is small enough (linear log-scale)
    assert abs(d2[2]) < 0.2 

# ==========================================
# 2. Algebraic Features Tests (Atomic)
# ==========================================

def test_mod_features():
    seq = [0, 1, 2, 3]
    
    # Mod 2 Sin: sin(2pi * (x%2)/2)
    # 0->0, 1->pi, 2->0, 3->pi
    # sin(0)=0, sin(pi)=0
    expected_sin_m2 = [0.0, 0.0, 0.0, 0.0] 
    assert_close(features.mod_sin(seq, m=2), expected_sin_m2)
    
    # Mod 2 Cos: cos(0)=1, cos(pi)=-1
    expected_cos_m2 = [1.0, -1.0, 1.0, -1.0]
    assert_close(features.mod_cos(seq, m=2), expected_cos_m2)

    # Mod 4 Cos check (Quarter cycle)
    # 1 -> 2pi * 1/4 = pi/2 -> cos(pi/2) approx 0
    cos_m4 = features.mod_cos(seq, m=4)
    assert math.isclose(cos_m4[1], 0.0, abs_tol=1e-5)

# ==========================================
# 3. Numeric Features Tests (Structure)
# ==========================================

def test_valuation():
    # 12 = 2^2 * 3^1 * 5^0
    seq = [12, 5, 0]
    
    # Valuation p=2
    # v_2(12)=2 -> log(1+2)
    # v_2(5)=0  -> log(1+0)
    # v_2(0)=0  -> 0.0 (Safety check)
    val2 = features.valuation(seq, p=2)
    assert math.isclose(val2[0], math.log1p(2))
    assert math.isclose(val2[1], 0.0)
    assert math.isclose(val2[2], 0.0)
    
    # Valuation p=5
    val5 = features.valuation(seq, p=5)
    assert math.isclose(val5[1], math.log1p(1))

def test_is_zero():
    seq = [0, 1, 0]
    assert_close(features.is_zero(seq), [1.0, 0.0, 1.0])

def test_is_prime():
    # 1: False (by utils definition)
    # 2: True
    # 4: False
    # -7: True (abs check)
    seq = [1, 2, 4, -7]
    expected = [0.0, 1.0, 0.0, 1.0]
    assert_close(features.is_prime(seq), expected)

def test_is_square():
    # 0, 1, 4 are squares. -4 is not.
    seq = [0, 1, 4, -4]
    expected = [1.0, 1.0, 1.0, 0.0]
    assert_close(features.is_square(seq), expected)

def test_is_cube():
    # 0, 1, 8 are cubes. 
    # -8 is a cube (-2)^3 = -8. 
    # 4 is not.
    seq = [0, 1, 8, 4, -8]
    expected = [1.0, 1.0, 1.0, 0.0, 1.0]
    assert_close(features.is_cube(seq), expected)

def test_is_square_free():
    # 1: True
    # 6: True (2*3)
    # 12: False (2^2 * 3)
    # 0: False (by utils definition)
    seq = [1, 6, 12, 0]
    expected = [1.0, 1.0, 0.0, 0.0]
    assert_close(features.is_square_free(seq), expected)

# ==========================================
# 4. Digital Features Tests
# ==========================================

def test_popcount():
    # 3 (binary 11) -> 2 bits
    # 12 (binary 1100) -> 2 bits
    seq = [3, 12, 0]
    expected = [math.log1p(2), math.log1p(2), 0.0]
    assert_close(features.popcount(seq), expected)

def test_digit_sum():
    # 123 -> 1+2+3=6
    seq = [123, -12] # -12 -> 1+2=3
    expected = [math.log1p(6), math.log1p(3)]
    assert_close(features.digit_sum(seq), expected)

def test_is_power_of_2():
    # 1, 2, 4 are powers of 2.
    # 0, -2, 3 are not.
    seq = [1, 2, 4, 3, 0, -2]
    expected = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert_close(features.is_power_of_2(seq), expected)