import math
from sympy import isprime, integer_nthroot
from sympy.ntheory import multiplicity
from sympy.ntheory.factor_ import core

LIMIT_SQUARE_FREE = 10**30
LIMIT_PRIME = 10**100

def is_prime(n: int) -> bool:
    """
    SymPy isprime wrapper with safety guard.
    Skip check for extremely large integers to prevent hanging.
    """
    if abs(n) > LIMIT_PRIME:
        return False  # Too large to check efficiently
    return isprime(n)

def is_square(n: int) -> bool:
    """Check if n is a perfect square. Handles negatives."""
    if n < 0: return False
    # integer_nthroot returns (root, is_exact)
    _, exact = integer_nthroot(n, 2)
    return exact

def is_square_free(n: int) -> bool:
    """
    Check if n is square-free. 
    HEAVY BOTTLENECK: factorization is required.
    Strictly restricted by LIMIT_SQUARE_FREE.
    """
    if n == 0: return False # Treat 0 as having square factors for feature consistency
    # core(n) returns the square-free part of n. If n is square-free, core(n) == n.
    n = abs(n)

    # Guard: Do not attempt to factor large numbers
    if n > LIMIT_SQUARE_FREE:
        return False
    
    return core(n) == n

def valuation(n: int, p: int) -> int:
    """
    p-adic valuation with ML safety guard.
    Mathematically v_p(0) = infinity, but we return 0 for feature stability.
    """
    if n == 0: return 0
    return multiplicity(p, n)

def popcount(n: int) -> int:
    """
    Population count for ML features.
    Uses absolute value to handle python's infinite bit-width for negatives.
    """
    # Python 3.10+ has int.bit_count(). Fallback for older versions if needed.
    return int(abs(n)).bit_count()

def digit_sum(n: int) -> int:
    """Sum of decimal digits of abs(n)."""
    # Simple implementation is sufficient for OEIS data scale
    return sum(int(d) for d in str(abs(n)))
