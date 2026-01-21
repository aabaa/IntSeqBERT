"""
solver.py:
Integer reconstruction from IntSeqBERT model predictions.

Implements IntegerSolver class with hybrid algorithm:
- Mode A (Dense): Full enumeration for small ranges
- Mode AB (Anchored Sieve): CRT-based sieving for medium ranges  
- Mode B (Sparse CRT): Pure CRT for huge ranges
"""

from __future__ import annotations

import math
import heapq
from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from . import config

if TYPE_CHECKING:
    from .intseq_models import IntSeqForPreTraining


# ============================================================
# Helper Functions (Public for unit testing)
# ============================================================


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean Algorithm (iterative version).
    
    Computes g, x, y such that ax + by = gcd(a, b).
    
    Args:
        a: First integer
        b: Second integer
    
    Returns:
        Tuple (g, x, y) where g = gcd(a, b) and ax + by = g
    
    Examples:
        >>> extended_gcd(15, 6)
        (3, 1, -2)  # 15*1 + 6*(-2) = 3
    """
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    
    return old_r, old_s, old_t


def solve_crt_pair(r1: int, m1: int, r2: int, m2: int) -> Tuple[int, int]:
    """
    Solve Chinese Remainder Theorem for two congruences.
    
    This is an efficient 2-variable version of solve_crt(), used internally
    by beam_search_crt() for incremental CRT computation.
    
    Find x such that:
        x ≡ r1 (mod m1)
        x ≡ r2 (mod m2)
    
    Args:
        r1: First remainder
        m1: First modulus
        r2: Second remainder
        m2: Second modulus
    
    Returns:
        Tuple (x, M) where x is the smallest non-negative solution
        and M = lcm(m1, m2) is the period.
    
    Raises:
        ValueError: If no solution exists (inconsistent remainders)
    
    Examples:
        >>> solve_crt_pair(2, 3, 3, 5)
        (8, 15)  # 8 ≡ 2 (mod 3), 8 ≡ 3 (mod 5)
    """
    g, p, _ = extended_gcd(m1, m2)
    diff = r2 - r1
    
    if diff % g != 0:
        raise ValueError(
            f"No CRT solution: {r1} mod {m1} and {r2} mod {m2} are inconsistent"
        )
    
    lcm_val = m1 * m2 // g
    x = (r1 + m1 * ((diff // g) * p)) % lcm_val
    return x, lcm_val


def solve_crt(equations: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Solve Chinese Remainder Theorem for multiple congruences.
    
    Args:
        equations: List of (remainder, modulus) pairs
    
    Returns:
        Tuple (x, M) where x is the smallest non-negative solution
        and M = lcm(all moduli) is the period.
    
    Raises:
        ValueError: If no solution exists
        ValueError: If equations list is empty
    
    Examples:
        >>> solve_crt([(2, 3), (3, 5), (2, 7)])
        (23, 105)
    """
    if not equations:
        raise ValueError("Empty equations list")
    
    x, M = equations[0]
    x = x % M  # Normalize first remainder
    
    for r, m in equations[1:]:
        x, M = solve_crt_pair(x, M, r % m, m)
    
    return x, M


def compute_lcm(a: int, b: int) -> int:
    """
    Compute Least Common Multiple.
    
    Args:
        a: First integer
        b: Second integer
    
    Returns:
        LCM of a and b
    """
    return abs(a * b) // math.gcd(a, b)


def compute_lcm_list(numbers: List[int]) -> int:
    """
    Compute LCM of a list of integers.
    
    Args:
        numbers: List of integers
    
    Returns:
        LCM of all numbers
    """
    if not numbers:
        return 1
    result = numbers[0]
    for n in numbers[1:]:
        result = compute_lcm(result, n)
    return result


# ============================================================
# Scoring Functions
# ============================================================


def compute_magnitude_score(
    n: int,
    mag_mu: float,
    sigma: float,
    epsilon: float = config.EPSILON
) -> float:
    """
    Compute Gaussian log-likelihood for magnitude prediction.
    
    Args:
        n: Candidate integer (must be positive)
        mag_mu: Predicted magnitude mean (1 + log10(|x|) scale)
        sigma: Magnitude standard deviation
        epsilon: Small value to prevent division by zero
    
    Returns:
        Log-likelihood score (higher is better)
    """
    if n <= 0:
        return float('-inf')
    
    log10_n = math.log10(n)
    mag_target = 1 + log10_n  # Convert to model's scale
    
    variance = sigma ** 2 + epsilon
    score = -((mag_target - mag_mu) ** 2) / (2 * variance)
    
    return score


def compute_modulo_score(
    n: int,
    mod_log_probs: List[torch.Tensor],
    mod_range: List[int]
) -> float:
    """
    Compute sum of log-probabilities for modulo predictions.
    
    Args:
        n: Candidate integer
        mod_log_probs: List of log-probability tensors for each modulus
        mod_range: List of moduli (e.g., [2, 3, ..., 101])
    
    Returns:
        Sum of log-probabilities (higher is better)
    """
    total = 0.0
    for i, m in enumerate(mod_range):
        remainder = n % m
        log_prob = mod_log_probs[i]
        
        if remainder < len(log_prob):
            total += log_prob[remainder].item()
        else:
            # Should not happen if mod_range matches log_probs dimensions
            total += -100.0  # Severe penalty
    
    return total


def compute_total_score(
    n: int,
    mag_mu: float,
    sigma: float,
    mod_log_probs: List[torch.Tensor],
    mod_range: List[int],
    mag_weight: float = config.SOLVER_MAG_WEIGHT,
    mod_weight: float = config.SOLVER_MOD_WEIGHT
) -> float:
    """
    Compute total score combining magnitude and modulo components.
    
    Total Score = mag_weight * LogLikelihood(Magnitude) + mod_weight * Sum(LogLikelihood(Mods))
    
    Args:
        n: Candidate integer (positive)
        mag_mu: Predicted magnitude mean
        sigma: Magnitude standard deviation
        mod_log_probs: List of log-probability tensors
        mod_range: List of moduli
        mag_weight: Weight for magnitude score (default from config)
        mod_weight: Weight for modulo score (default from config, discounted for redundancy)
    
    Returns:
        Total weighted log-likelihood score
    """
    mag_score = compute_magnitude_score(n, mag_mu, sigma)
    mod_score = compute_modulo_score(n, mod_log_probs, mod_range)
    return (mag_weight * mag_score) + (mod_weight * mod_score)


def compute_total_scores_batch(
    candidates: List[int],
    mag_mu: float,
    sigma: float,
    mod_log_probs: List[torch.Tensor],
    mod_range: List[int],
    mag_weight: float = config.SOLVER_MAG_WEIGHT,
    mod_weight: float = config.SOLVER_MOD_WEIGHT
) -> torch.Tensor:
    """
    Vectorized batch scoring for multiple candidates.
    
    Significantly faster than calling compute_total_score in a loop.
    
    Args:
        candidates: List of candidate integers (positive)
        mag_mu: Predicted magnitude mean
        sigma: Magnitude standard deviation
        mod_log_probs: List of log-probability tensors (on same device)
        mod_range: List of moduli
        mag_weight: Weight for magnitude score (default from config)
        mod_weight: Weight for modulo score (default from config, discounted for redundancy)
    
    Returns:
        Tensor of scores (same length as candidates)
    """
    if not candidates:
        return torch.tensor([])
    
    n_candidates = len(candidates)
    
    # Handle very large integers that don't fit in int64
    # Use Python's arbitrary precision arithmetic then convert
    try:
        candidates_arr = np.array(candidates, dtype=np.int64)
    except (OverflowError, ValueError):
        # Fall back to object dtype for arbitrarily large integers
        candidates_arr = np.array(candidates, dtype=object)
    
    # 1. Magnitude scores (vectorized)
    # Avoid log of zero/negative
    valid_mask = candidates_arr > 0
    log10_vals = np.zeros(n_candidates, dtype=np.float64)
    
    # Convert to float for log computation (handles both int64 and object dtype)
    if candidates_arr.dtype == object:
        # For arbitrarily large integers, convert to float64 (may lose precision but that's OK for scoring)
        valid_candidates = [float(c) for i, c in enumerate(candidates) if valid_mask[i]]
        if valid_candidates:
            log10_vals[valid_mask] = np.log10(valid_candidates)
    else:
        log10_vals[valid_mask] = np.log10(candidates_arr[valid_mask].astype(np.float64))
    
    mag_targets = 1 + log10_vals  # Model's scale
    
    variance = sigma ** 2 + config.EPSILON
    mag_scores = -((mag_targets - mag_mu) ** 2) / (2 * variance)
    mag_scores[~valid_mask] = -1e10  # Penalize invalid candidates
    
    # 2. Pre-cache all log_prob tensors as numpy arrays (avoid repeated GPU->CPU transfer)
    log_probs_np = [lp.detach().cpu().numpy() for lp in mod_log_probs]
    
    # 3. Modulo scores (vectorized per modulus)
    mod_scores = np.zeros(n_candidates)
    
    for i, m in enumerate(mod_range):
        # Compute remainders (works for both int64 and object dtype)
        if candidates_arr.dtype == object:
            remainders = np.array([int(c % m) for c in candidates], dtype=np.int64)
        else:
            remainders = candidates_arr % m
        
        log_prob_np = log_probs_np[i]
        valid_remainders = remainders < len(log_prob_np)
        
        # Safe indexing with numpy advanced indexing
        valid_idx = remainders[valid_remainders]
        mod_scores[valid_remainders] += log_prob_np[valid_idx]
        mod_scores[~valid_remainders] += -100.0  # Penalty for invalid
    
    # 4. Total scores (Apply weights)
    total_scores = (mag_weight * mag_scores) + (mod_weight * mod_scores)
    
    return torch.from_numpy(total_scores)


# ============================================================
# Top-K Remainder Extraction
# ============================================================


def get_top_remainders(
    log_probs: torch.Tensor,
    k: int
) -> List[Tuple[int, float]]:
    """
    Get top-k remainders by log-probability.
    
    Args:
        log_probs: Log-probability tensor of shape (m,)
        k: Number of top remainders to return
    
    Returns:
        List of (remainder, log_prob) tuples, sorted by prob descending
    """
    k = min(k, len(log_probs))
    values, indices = log_probs.topk(k)
    return [(idx.item(), val.item()) for idx, val in zip(indices, values)]


# ============================================================
# Mode A: Dense Search
# ============================================================


def solve_dense(
    n_min: int,
    n_max: int,
    mag_mu: float,
    sigma: float,
    mod_log_probs: List[torch.Tensor],
    mod_range: List[int],
    top_k: int
) -> List[Dict]:
    """
    Mode A: Exhaustive search over all integers in range.
    
    Suitable for small ranges (width <= DENSE_THRESHOLD).
    
    Args:
        n_min: Minimum of search range (inclusive)
        n_max: Maximum of search range (inclusive)
        mag_mu: Magnitude prediction mean
        sigma: Magnitude standard deviation
        mod_log_probs: Log-probabilities for each modulus
        mod_range: List of moduli
        top_k: Number of top candidates to return
    
    Returns:
        List of candidate dicts with keys: value, score, method
    """
    width = n_max - n_min + 1
    
    # For very small ranges (< 1000), use simple loop
    # For larger ranges, use vectorized batch scoring
    VECTORIZE_THRESHOLD = 1000
    
    if width <= VECTORIZE_THRESHOLD:
        # Small range: use heap-based approach
        candidates = []
        for n in range(n_min, n_max + 1):
            score = compute_total_score(n, mag_mu, sigma, mod_log_probs, mod_range)
            
            if len(candidates) < top_k:
                heapq.heappush(candidates, (score, n))
            else:
                if score > candidates[0][0]:
                    heapq.heapreplace(candidates, (score, n))
        
        results = []
        for score, n in sorted(candidates, key=lambda x: x[0], reverse=True):
            results.append({
                "value": n,
                "score": score,
                "method": "dense"
            })
        return results
    
    else:
        # Large range: use vectorized batch scoring
        candidate_list = list(range(n_min, n_max + 1))
        scores = compute_total_scores_batch(
            candidate_list, mag_mu, sigma, mod_log_probs, mod_range
        )
        
        # Get top-k using torch.topk
        if len(candidate_list) <= top_k:
            top_indices = list(range(len(candidate_list)))
        else:
            top_indices = scores.topk(top_k).indices.tolist()
        
        results = []
        for idx in top_indices:
            results.append({
                "value": candidate_list[idx],
                "score": scores[idx].item(),
                "method": "dense"
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ============================================================
# Mode AB: Anchored Sieve
# ============================================================


def select_anchors(
    mod_log_probs: List[torch.Tensor],
    mod_range: List[int],
    width: int,
    target_candidates: int,
    max_anchors: int
) -> List[int]:
    """
    Select anchor moduli for sieving based on confidence.
    
    Selects moduli in order of highest confidence (max probability)
    until expected candidate count is below target.
    
    Args:
        mod_log_probs: Log-probabilities for each modulus
        mod_range: List of moduli
        width: Search range width (n_max - n_min)
        target_candidates: Target number of candidates after sieving
        max_anchors: Maximum number of anchors to select
    
    Returns:
        List of anchor indices (into mod_range)
    """
    # Compute confidence (max log-prob) for each modulus
    confidences = []
    for i, log_probs in enumerate(mod_log_probs):
        max_log_prob = log_probs.max().item()
        confidences.append((i, max_log_prob))
    
    # Sort by confidence descending
    confidences.sort(key=lambda x: x[1], reverse=True)
    
    anchors = []
    lcm = 1
    
    for idx, _ in confidences:
        m = mod_range[idx]
        new_lcm = compute_lcm(lcm, m)
        anchors.append(idx)
        lcm = new_lcm
        
        # Check if candidate count is below target
        expected_candidates = width // lcm + 1
        if expected_candidates <= target_candidates:
            break
        
        if len(anchors) >= max_anchors:
            break
    
    return anchors


def beam_search_crt(
    anchor_indices: List[int],
    mod_log_probs: List[torch.Tensor],
    mod_range: List[int],
    beam_width: int
) -> List[Tuple[int, int, float]]:
    """
    Beam search over CRT combinations for anchor moduli.
    
    Follows the spec pattern: initialize with (0, 1, 0.0) and process
    all anchors uniformly through the loop.
    
    Args:
        anchor_indices: Indices of anchor moduli
        mod_log_probs: Log-probabilities for each modulus
        mod_range: List of moduli
        beam_width: Maximum beams to keep at each step
    
    Returns:
        List of (x, M, cumulative_log_prob) tuples
        where x ≡ r_i (mod m_i) for all anchors
    """
    if not anchor_indices:
        return [(0, 1, 0.0)]
    
    # Initialize with identity element for CRT
    # (x=0, M=1, cumulative_log_prob=0.0)
    beams = [(0, 1, 0.0)]
    
    # Process all anchors uniformly
    for anchor_idx in anchor_indices:
        m = mod_range[anchor_idx]
        top_rems = get_top_remainders(mod_log_probs[anchor_idx], beam_width)
        
        new_beams = []
        for x, M, cum_prob in beams:
            for r, log_p in top_rems:
                try:
                    new_x, new_M = solve_crt_pair(x, M, r, m)
                    new_beams.append((new_x, new_M, cum_prob + log_p))
                except ValueError:
                    # Inconsistent remainders, skip this combination
                    continue
        
        if not new_beams:
            # All combinations failed, keep previous beams
            continue
        
        # Keep top beam_width by cumulative probability (optimized with heapq)
        if len(new_beams) <= beam_width:
            beams = new_beams
        else:
            # Use heapq.nlargest for better performance than full sort
            beams = heapq.nlargest(beam_width, new_beams, key=lambda x: x[2])
    
    return beams


def enumerate_candidates_from_beams(
    beams: List[Tuple[int, int, float]],
    n_min: int,
    n_max: int,
    max_count: int = None,
    skip_threshold: int = None
) -> set:
    """
    Enumerate all integers in range that match beam CRT solutions.
    
    Args:
        beams: List of (x, M, log_prob) from beam search
        n_min: Minimum of search range
        n_max: Maximum of search range
        max_count: Safety limit on number of candidates to return
                   (default: config.SOLVER_MAX_ENUM_CANDIDATES)
        skip_threshold: Skip beams that would generate more than this many
                        candidates (default: config.SOLVER_BEAM_SKIP_THRESHOLD)
    
    Returns:
        Set of candidate integers
    """
    if max_count is None:
        max_count = config.SOLVER_MAX_ENUM_CANDIDATES
    if skip_threshold is None:
        skip_threshold = config.SOLVER_BEAM_SKIP_THRESHOLD
    
    candidates = set()
    
    for x, M, _ in beams:
        if M == 0:
            continue
        
        # Find all n = x + k*M in [n_min, n_max]
        if x < n_min:
            k_start = (n_min - x + M - 1) // M
        else:
            k_start = 0
        
        k_end = (n_max - x) // M
        
        # Skip this beam if it would generate too many candidates
        num_candidates_for_beam = k_end - k_start + 1
        if num_candidates_for_beam > skip_threshold:
            # This beam has too low LCM, skip it to avoid CPU thrashing
            continue
        
        for k in range(k_start, k_end + 1):
            n = x + k * M
            if n_min <= n <= n_max:
                candidates.add(n)
                if len(candidates) >= max_count:
                    return candidates
    
    return candidates


def solve_sieve(
    n_min: int,
    n_max: int,
    mag_mu: float,
    sigma: float,
    mod_log_probs: List[torch.Tensor],
    mod_range: List[int],
    top_k: int,
    sieve_target: int = config.SOLVER_SIEVE_TARGET,
    max_anchors: int = config.SOLVER_MAX_ANCHORS,
    beam_width: int = config.SOLVER_BEAM_WIDTH
) -> List[Dict]:
    """
    Mode AB: Anchored Sieve search.
    
    Uses high-confidence moduli to sieve candidates, then scores with all moduli.
    
    Args:
        n_min: Minimum of search range
        n_max: Maximum of search range
        mag_mu: Magnitude prediction mean
        sigma: Magnitude standard deviation
        mod_log_probs: Log-probabilities for each modulus
        mod_range: List of moduli
        top_k: Number of top candidates to return
        sieve_target: Target candidate count after sieving
        max_anchors: Maximum anchor moduli to use
        beam_width: Beam width for CRT search
    
    Returns:
        List of candidate dicts
    """
    width = n_max - n_min
    
    # 1. Select anchors
    anchors = select_anchors(
        mod_log_probs, mod_range, width, sieve_target, max_anchors
    )
    
    if not anchors:
        # Fallback to dense if no anchors selected
        return solve_dense(
            n_min, n_max, mag_mu, sigma, mod_log_probs, mod_range, top_k
        )
    
    # 2. Beam search CRT
    beams = beam_search_crt(anchors, mod_log_probs, mod_range, beam_width)
    
    # 3. Enumerate candidates (using config defaults)
    candidate_set = enumerate_candidates_from_beams(beams, n_min, n_max)
    
    if not candidate_set:
        # No valid candidates found, try single best per anchor
        return []
    
    # 4. Score all candidates with vectorized batch scoring
    candidate_list = list(candidate_set)
    scores = compute_total_scores_batch(
        candidate_list, mag_mu, sigma, mod_log_probs, mod_range
    )
    
    # 5. Get top-k using argsort (faster than heap for vectorized data)
    if len(candidate_list) <= top_k:
        # Return all if fewer than top_k
        top_indices = list(range(len(candidate_list)))
    else:
        # Get indices of top-k scores
        top_indices = scores.topk(top_k).indices.tolist()
    
    candidates = []
    for idx in top_indices:
        candidates.append({
            "value": candidate_list[idx],
            "score": scores[idx].item(),
            "method": "sieve"
        })
    
    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


# ============================================================
# Mode B: Sparse CRT
# ============================================================


def select_basis(
    mod_log_probs: List[torch.Tensor],
    mod_range: List[int],
    width: int
) -> List[int]:
    """
    Select basis moduli for CRT such that LCM exceeds width.
    
    Args:
        mod_log_probs: Log-probabilities for each modulus
        mod_range: List of moduli
        width: Search range width
    
    Returns:
        List of basis indices (into mod_range)
    """
    # Compute confidence for each modulus
    confidences = []
    for i, log_probs in enumerate(mod_log_probs):
        max_log_prob = log_probs.max().item()
        confidences.append((i, max_log_prob))
    
    confidences.sort(key=lambda x: x[1], reverse=True)
    
    basis = []
    lcm = 1
    
    for idx, _ in confidences:
        m = mod_range[idx]
        basis.append(idx)
        lcm = compute_lcm(lcm, m)
        
        # Stop when LCM exceeds width (solution becomes unique)
        if lcm > width:
            break
    
    return basis


def solve_sparse_crt(
    n_min: int,
    n_max: int,
    mag_mu: float,
    sigma: float,
    mod_log_probs: List[torch.Tensor],
    mod_range: List[int],
    top_k: int,
    beam_width: int = config.SOLVER_BEAM_WIDTH
) -> List[Dict]:
    """
    Mode B: Sparse CRT search for huge ranges.
    
    Constructs candidates directly from CRT without enumeration.
    
    Args:
        n_min: Minimum of search range
        n_max: Maximum of search range
        mag_mu: Magnitude prediction mean
        sigma: Magnitude standard deviation
        mod_log_probs: Log-probabilities for each modulus
        mod_range: List of moduli
        top_k: Number of top candidates to return
        beam_width: Beam width for CRT search
    
    Returns:
        List of candidate dicts
    """
    width = n_max - n_min
    
    # 1. Select basis
    basis = select_basis(mod_log_probs, mod_range, width)
    
    if not basis:
        return []
    
    # 2. Beam search CRT
    beams = beam_search_crt(basis, mod_log_probs, mod_range, beam_width)
    
    # 3. Adjust candidates to be within range and use batch scoring
    candidate_list = []
    seen = set()
    
    for x, M, _ in beams:
        if M == 0:
            continue
        
        # Adjust x to fall within [n_min, n_max]
        n = x
        if n < n_min:
            k = (n_min - n + M - 1) // M
            n = n + k * M
        elif n > n_max:
            k = (n - n_max + M - 1) // M
            n = n - k * M
        
        if n_min <= n <= n_max and n not in seen:
            seen.add(n)
            candidate_list.append(n)
    
    if not candidate_list:
        return []
    
    # Batch scoring for all candidates
    scores = compute_total_scores_batch(
        candidate_list, mag_mu, sigma, mod_log_probs, mod_range
    )
    
    # Build result with scores
    candidates = []
    for i, n in enumerate(candidate_list):
        candidates.append({
            "value": n,
            "score": scores[i].item(),
            "method": "crt"
        })
    
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


# ============================================================
# IntegerSolver Class
# ============================================================


class IntegerSolver:
    """
    Reconstruct integers from IntSeqBERT model predictions.
    
    Uses hybrid algorithm with three modes:
    - Mode A (Dense): Exhaustive search for small ranges
    - Mode AB (Sieve): CRT-based sieving for medium ranges
    - Mode B (CRT): Direct CRT for huge ranges
    
    Attributes:
        mod_range: List of moduli [2, 3, ..., 101]
        dense_threshold: Width threshold for Mode A
        sieve_threshold: Width threshold for Mode AB
        sieve_target: Target candidates after sieving
        max_anchors: Maximum anchors for sieving
        beam_width: Beam width for CRT search
    """
    
    def __init__(
        self,
        mod_range: List[int] = None,
        dense_threshold: int = config.SOLVER_DENSE_THRESHOLD,
        sieve_threshold: int = config.SOLVER_SIEVE_THRESHOLD,
        sieve_target: int = config.SOLVER_SIEVE_TARGET,
        max_anchors: int = config.SOLVER_MAX_ANCHORS,
        beam_width: int = config.SOLVER_BEAM_WIDTH
    ):
        """
        Initialize IntegerSolver.
        
        Args:
            mod_range: List of moduli (default: config.MOD_RANGE)
            dense_threshold: Mode A → AB threshold (default: 1M)
            sieve_threshold: Mode AB → B threshold (default: 10^14)
            sieve_target: Target candidates for sieving (default: 100K)
            max_anchors: Max anchors for sieving (default: 20)
            beam_width: Beam width for CRT (default: 10)
        """
        self.mod_range = mod_range if mod_range is not None else config.MOD_RANGE
        self.dense_threshold = dense_threshold
        self.sieve_threshold = sieve_threshold
        self.sieve_target = sieve_target
        self.max_anchors = max_anchors
        self.beam_width = beam_width
    
    def solve(
        self,
        mag_mu: float,
        mag_log_var: float,
        sign_idx: int,
        mod_log_probs: List[torch.Tensor],
        top_k: int = config.SOLVER_TOP_K_DEFAULT
    ) -> List[Dict]:
        """
        Reconstruct integer from model predictions.
        
        Args:
            mag_mu: Magnitude mean (1 + log10(|x|) scale)
            mag_log_var: Magnitude log-variance (uncertainty)
            sign_idx: Sign index (0=Positive, 1=Negative, 2=Zero)
            mod_log_probs: List of log-softmax tensors for each modulus
            top_k: Number of candidates to return
        
        Returns:
            List of dicts with keys: value, score, method
            Sorted by score descending.
        
        Raises:
            ValueError: If sign_idx not in {0, 1, 2}
        """
        # Validate sign
        if sign_idx not in (0, 1, 2):
            raise ValueError(f"Invalid sign_idx: {sign_idx}. Must be 0, 1, or 2.")
        
        # Handle zero case
        if sign_idx == config.SIGN_ZERO:
            return [{"value": 0, "score": 0.0, "method": "zero"}]
        
        # Determine if result should be negative
        is_negative = (sign_idx == config.SIGN_NEGATIVE)
        
        # Compute search range from magnitude prediction
        sigma = math.exp(0.5 * mag_log_var)
        sigma = max(sigma, config.EPSILON)  # Prevent zero sigma
        
        # mag_mu is 1 + log10(|x|), so log10(|x|) = mag_mu - 1
        log10_center = mag_mu - 1
        
        # Clamp extreme values to prevent overflow
        log10_min = max(0, log10_center - 3 * sigma)
        log10_max = min(100, log10_center + 3 * sigma)  # Cap at 10^100
        
        n_min = max(1, int(math.floor(10 ** log10_min)))
        n_max = int(math.ceil(10 ** log10_max))
        
        # Ensure valid range
        if n_max < n_min:
            n_max = n_min
        
        width = n_max - n_min
        
        # Select mode based on width
        if width <= self.dense_threshold:
            candidates = solve_dense(
                n_min, n_max, mag_mu, sigma,
                mod_log_probs, self.mod_range, top_k
            )
        elif width <= self.sieve_threshold:
            candidates = solve_sieve(
                n_min, n_max, mag_mu, sigma,
                mod_log_probs, self.mod_range, top_k,
                self.sieve_target, self.max_anchors, self.beam_width
            )
        else:
            candidates = solve_sparse_crt(
                n_min, n_max, mag_mu, sigma,
                mod_log_probs, self.mod_range, top_k,
                self.beam_width
            )
        
        # Apply sign
        if is_negative:
            for c in candidates:
                c["value"] = -c["value"]
        
        return candidates
    
    @staticmethod
    def from_model_output(
        predictions: Dict,
        position: int,
        model: "IntSeqForPreTraining",
        batch_idx: int = 0
    ) -> Tuple[float, float, int, List[torch.Tensor]]:
        """
        Convert model predictions to solve() input format.
        
        Args:
            predictions: model.forward()["predictions"] dict
            position: Sequence position index (0-based)
            model: Model instance with _split_mod_logits method
            batch_idx: Batch index (default: 0)
        
        Returns:
            Tuple of (mag_mu, mag_log_var, sign_idx, mod_log_probs)
            ready to be passed to solve().
        
        Example:
            >>> outputs = model(mag_features, mod_features, mask)
            >>> args = IntegerSolver.from_model_output(
            ...     outputs["predictions"], pos=5, model=model
            ... )
            >>> solver = IntegerSolver()
            >>> candidates = solver.solve(*args)
        """
        # Extract magnitude predictions
        mag_mu = predictions["mag_mu"][batch_idx, position].item()
        mag_log_var = predictions["mag_log_var"][batch_idx, position].item()
        
        # Extract sign prediction
        sign_logits = predictions["sign_logits"][batch_idx, position]
        sign_idx = sign_logits.argmax().item()
        
        # Extract and convert modulo predictions
        mod_logits = predictions["mod_logits"][batch_idx, position]  # (~5150,)
        
        # Split into per-modulus logits
        mod_logits_list = model._split_mod_logits(mod_logits.unsqueeze(0))
        
        # Apply log_softmax to each modulus
        mod_log_probs = []
        for logits in mod_logits_list:
            log_probs = F.log_softmax(logits.squeeze(0), dim=-1)
            # Clamp to prevent -inf
            log_probs = torch.clamp(log_probs, min=-100.0)
            mod_log_probs.append(log_probs)
        
        return mag_mu, mag_log_var, sign_idx, mod_log_probs
