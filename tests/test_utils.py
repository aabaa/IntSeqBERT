# tests/test_utils.py
import pytest
from intseq_bert import utils

def test_is_prime():
    assert not utils.is_prime(0)
    assert not utils.is_prime(1)
    assert utils.is_prime(2)
    assert utils.is_prime(3)
    assert not utils.is_prime(4)
    assert utils.is_prime(17)
    assert not utils.is_prime(100)
    assert utils.is_prime(104729) # 10000th prime

def test_is_square():
    assert utils.is_square(0)
    assert utils.is_square(1)
    assert utils.is_square(4)
    assert not utils.is_square(2)
    assert not utils.is_square(-1)
    assert utils.is_square(123456789 ** 2)

def test_is_square_free():
    # True cases
    assert utils.is_square_free(1)
    assert utils.is_square_free(-1)
    assert utils.is_square_free(2)
    assert utils.is_square_free(6)  # 2*3
    assert utils.is_square_free(10) # 2*5
    assert utils.is_square_free(30) # 2*3*5
    
    # False cases
    assert not utils.is_square_free(0)
    assert not utils.is_square_free(4)  # 2^2
    assert not utils.is_square_free(12) # 2^2 * 3
    assert not utils.is_square_free(18) # 2 * 3^2
    assert not utils.is_square_free(9)

def test_valuation():
    assert utils.valuation(8, 2) == 3 # 2^3
    assert utils.valuation(12, 2) == 2 # 2^2 * 3
    assert utils.valuation(12, 3) == 1
    assert utils.valuation(5, 2) == 0
    assert utils.valuation(0, 2) == 0 # definition check

def test_popcount():
    assert utils.popcount(0) == 0
    assert utils.popcount(1) == 1
    assert utils.popcount(2) == 1 # 10
    assert utils.popcount(3) == 2 # 11
    assert utils.popcount(7) == 3 # 111
    assert utils.popcount(-7) == 3 # abs check

def test_digit_sum():
    assert utils.digit_sum(0) == 0
    assert utils.digit_sum(123) == 6
    assert utils.digit_sum(-123) == 6