# tests/test_utils.py
import pytest
from intseq_bert.utils import is_prime_check, is_square_check, is_square_free_check, get_valuation, get_popcount, get_digit_sum

def test_is_prime_check():
    assert not is_prime_check(0)
    assert not is_prime_check(1)
    assert is_prime_check(2)
    assert is_prime_check(3)
    assert not is_prime_check(4)
    assert is_prime_check(17)
    assert not is_prime_check(100)
    assert is_prime_check(104729) # 10000th prime

def test_is_square_check():
    assert is_square_check(0)
    assert is_square_check(1)
    assert is_square_check(4)
    assert not is_square_check(2)
    assert not is_square_check(-1)
    assert is_square_check(123456789 ** 2)

def test_is_square_free_check():
    # True cases
    assert is_square_free_check(1)
    assert is_square_free_check(-1)
    assert is_square_free_check(2)
    assert is_square_free_check(6)  # 2*3
    assert is_square_free_check(10) # 2*5
    assert is_square_free_check(30) # 2*3*5
    
    # False cases
    assert not is_square_free_check(0)
    assert not is_square_free_check(4)  # 2^2
    assert not is_square_free_check(12) # 2^2 * 3
    assert not is_square_free_check(18) # 2 * 3^2
    assert not is_square_free_check(9)

def test_get_valuation():
    assert get_valuation(8, 2) == 3 # 2^3
    assert get_valuation(12, 2) == 2 # 2^2 * 3
    assert get_valuation(12, 3) == 1
    assert get_valuation(5, 2) == 0
    assert get_valuation(0, 2) == 0 # definition check

def test_get_popcount():
    assert get_popcount(0) == 0
    assert get_popcount(1) == 1
    assert get_popcount(2) == 1 # 10
    assert get_popcount(3) == 2 # 11
    assert get_popcount(7) == 3 # 111
    assert get_popcount(-7) == 3 # abs check

def test_get_digit_sum():
    assert get_digit_sum(0) == 0
    assert get_digit_sum(123) == 6
    assert get_digit_sum(-123) == 6