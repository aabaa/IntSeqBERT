import pytest
import math
from typing import List
from intseq_bert import features

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
    
    expected_log = [0.0, 1.0 + math.log(10), 1.0 + math.log(100)]
    assert_close(features.log_magnitude(seq), expected_log)
    
    # Sign remains the same
    expected_sign = [0.0, -1.0, 1.0]
    assert_close(features.sign(seq), expected_sign)

def test_direction():
    seq = [10, 20, 20, 15]
    expected = [0.0, 1.0, 0.0, -1.0]
    assert_close(features.direction(seq), expected)

def test_log_raw_diff():
    # Raw value differences
    seq = [10, 10, 100] 
    # Diffs: [N/A, 0, 90]
    expected = [
        0.0, 
        0.0,             # diff=0 -> 0.0
        1.0 + math.log(90)
    ]
    assert_close(features.log_raw_diff(seq), expected)

def test_diff_derivatives():
    # Exponential growth: 1, 10, 100
    seq = [1, 10, 100]
    logs = features.log_magnitude(seq)
    
    # Diff1 (Velocity)
    d1 = features.diff1(seq)
    assert d1[0] == 0.0
    assert math.isclose(d1[1], math.log(10))
    
    # Diff2 (Acceleration)
    d2 = features.diff2(seq)
    assert d2[0] == 0.0
    
    # Acceleration should be practically ZERO for exponential sequence
    assert math.isclose(d2[2], 0.0, abs_tol=1e-9)

# ==========================================
# 2. Algebraic Features Tests
# ==========================================

def test_mod_features():
    seq = [0, 1, 2, 3]
    expected_sin_m2 = [0.0, 0.0, 0.0, 0.0] 
    assert_close(features.mod_sin(seq, m=2), expected_sin_m2)
    expected_cos_m2 = [1.0, -1.0, 1.0, -1.0]
    assert_close(features.mod_cos(seq, m=2), expected_cos_m2)
    cos_m4 = features.mod_cos(seq, m=4)
    assert math.isclose(cos_m4[1], 0.0, abs_tol=1e-5)

# ==========================================
# 3. Numeric Features Tests
# ==========================================

def test_valuation():
    seq = [12, 5, 0]
    val2 = features.valuation(seq, p=2)
    assert math.isclose(val2[0], math.log1p(2))
    assert math.isclose(val2[1], 0.0)
    assert math.isclose(val2[2], 0.0)
    val5 = features.valuation(seq, p=5)
    assert math.isclose(val5[1], math.log1p(1))

def test_is_zero():
    seq = [0, 1, 0]
    assert_close(features.is_zero(seq), [1.0, 0.0, 1.0])

def test_is_prime():
    seq = [1, 2, 4, -7]
    expected = [0.0, 1.0, 0.0, 1.0]
    assert_close(features.is_prime(seq), expected)

def test_is_square():
    seq = [0, 1, 4, -4]
    expected = [1.0, 1.0, 1.0, 0.0]
    assert_close(features.is_square(seq), expected)

def test_is_cube():
    seq = [0, 1, 8, 4, -8]
    expected = [1.0, 1.0, 1.0, 0.0, 1.0]
    assert_close(features.is_cube(seq), expected)

def test_is_square_free():
    seq = [1, 6, 12, 0]
    expected = [1.0, 1.0, 0.0, 0.0]
    assert_close(features.is_square_free(seq), expected)

# ==========================================
# 4. Digital Features Tests
# ==========================================

def test_popcount():
    seq = [3, 12, 0]
    expected = [math.log1p(2), math.log1p(2), 0.0]
    assert_close(features.popcount(seq), expected)

def test_digit_sum():
    seq = [123, -12]
    expected = [math.log1p(6), math.log1p(3)]
    assert_close(features.digit_sum(seq), expected)

def test_is_power_of_2():
    seq = [1, 2, 4, 3, 0, -2]
    expected = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert_close(features.is_power_of_2(seq), expected)