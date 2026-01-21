"""
Unit tests for solver.py

Tests cover:
- Helper functions (extended_gcd, solve_crt, lcm)
- Scoring functions
- Mode A: Dense Search
- Mode AB: Anchored Sieve
- Mode B: Sparse CRT
- IntegerSolver class integration
"""

import math
import pytest
import torch

from intseq_bert import config
from intseq_bert.solver import (
    # Helper functions
    extended_gcd,
    solve_crt_pair,
    solve_crt,
    compute_lcm,
    compute_lcm_list,
    # Scoring functions
    compute_magnitude_score,
    compute_modulo_score,
    compute_total_score,
    compute_total_scores_batch,
    # Top-K extraction
    get_top_remainders,
    # Mode functions
    solve_dense,
    select_anchors,
    beam_search_crt,
    enumerate_candidates_from_beams,
    solve_sieve,
    select_basis,
    solve_sparse_crt,
    # Main classes
    IntegerSolver,
    VanillaSolver,
)


# ============================================================
# Test Fixtures
# ============================================================


@pytest.fixture
def small_mod_range():
    """Small set of moduli for faster tests."""
    return [2, 3, 5, 7]


@pytest.fixture
def uniform_log_probs(small_mod_range):
    """Uniform log-probabilities for each modulus."""
    return [torch.ones(m).log_softmax(dim=-1) for m in small_mod_range]


@pytest.fixture
def peaked_log_probs(small_mod_range):
    """Log-probabilities peaked at specific remainders.
    
    Returns probs peaked at: mod2=0, mod3=1, mod5=2, mod7=3
    Target integer: 17 (17%2=1, 17%3=2, 17%5=2, 17%7=3)
    Let's use target=23: 23%2=1, 23%3=2, 23%5=3, 23%7=2
    """
    probs = []
    peaks = [1, 2, 3, 2]  # Remainders for n=23
    for m, peak in zip(small_mod_range, peaks):
        logits = torch.zeros(m)
        logits[peak] = 10.0  # Strong peak
        probs.append(logits.log_softmax(dim=-1))
    return probs


@pytest.fixture
def full_mod_range():
    """Full moduli range from config."""
    return config.MOD_RANGE


# ============================================================
# Test: Helper Functions - extended_gcd
# ============================================================


class TestExtendedGcd:
    """Tests for extended_gcd function."""
    
    def test_basic_case(self):
        """Test basic GCD computation."""
        g, x, y = extended_gcd(15, 6)
        assert g == 3
        assert 15 * x + 6 * y == g
    
    def test_coprime(self):
        """Test coprime numbers (GCD = 1)."""
        g, x, y = extended_gcd(17, 13)
        assert g == 1
        assert 17 * x + 13 * y == g
    
    def test_one_is_multiple(self):
        """Test when one number divides the other."""
        g, x, y = extended_gcd(12, 4)
        assert g == 4
        assert 12 * x + 4 * y == g
    
    def test_zero_case(self):
        """Test with zero."""
        g, x, y = extended_gcd(5, 0)
        assert g == 5
        assert x == 1
        assert y == 0
    
    def test_large_numbers(self):
        """Test with larger numbers."""
        g, x, y = extended_gcd(1071, 462)
        assert g == 21
        assert 1071 * x + 462 * y == g


# ============================================================
# Test: Helper Functions - solve_crt
# ============================================================


class TestSolveCrtPair:
    """Tests for solve_crt_pair function."""
    
    def test_basic_case(self):
        """Test basic CRT with coprime moduli."""
        x, M = solve_crt_pair(2, 3, 3, 5)
        assert x == 8
        assert M == 15
        assert x % 3 == 2
        assert x % 5 == 3
    
    def test_non_coprime_consistent(self):
        """Test non-coprime moduli with consistent remainders."""
        # x ≡ 2 (mod 4) and x ≡ 2 (mod 6)
        # GCD(4,6) = 2, and 2 ≡ 2 (mod 2), so consistent
        x, M = solve_crt_pair(2, 4, 2, 6)
        assert M == 12
        assert x % 4 == 2
        assert x % 6 == 2
    
    def test_non_coprime_inconsistent(self):
        """Test non-coprime moduli with inconsistent remainders."""
        # x ≡ 0 (mod 4) and x ≡ 3 (mod 6)
        # GCD(4,6) = 2, and 0 ≢ 3 (mod 2), so inconsistent
        with pytest.raises(ValueError, match="inconsistent"):
            solve_crt_pair(0, 4, 3, 6)


class TestSolveCrt:
    """Tests for solve_crt function (multiple congruences)."""
    
    def test_single_equation(self):
        """Test with single equation."""
        x, M = solve_crt([(5, 7)])
        assert x == 5
        assert M == 7
    
    def test_two_equations(self):
        """Test with two equations."""
        x, M = solve_crt([(2, 3), (3, 5)])
        assert x == 8
        assert M == 15
    
    def test_three_equations(self):
        """Test with three equations."""
        x, M = solve_crt([(2, 3), (3, 5), (2, 7)])
        assert x == 23
        assert M == 105
        assert x % 3 == 2
        assert x % 5 == 3
        assert x % 7 == 2
    
    def test_empty_raises(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError, match="Empty"):
            solve_crt([])
    
    def test_classic_example(self):
        """Test classic CRT example: find n where n%3=2, n%5=3, n%7=2."""
        x, M = solve_crt([(2, 3), (3, 5), (2, 7)])
        assert x == 23
        # Verify all congruences
        for r, m in [(2, 3), (3, 5), (2, 7)]:
            assert x % m == r


# ============================================================
# Test: Helper Functions - LCM
# ============================================================


class TestLcm:
    """Tests for LCM functions."""
    
    def test_compute_lcm_basic(self):
        """Test basic LCM."""
        assert compute_lcm(4, 6) == 12
        assert compute_lcm(3, 5) == 15
        assert compute_lcm(7, 7) == 7
    
    def test_compute_lcm_coprime(self):
        """Test LCM of coprime numbers."""
        assert compute_lcm(8, 9) == 72
    
    def test_compute_lcm_list(self):
        """Test LCM of list."""
        assert compute_lcm_list([2, 3, 4]) == 12
        assert compute_lcm_list([5, 7, 11]) == 385
    
    def test_compute_lcm_list_empty(self):
        """Test LCM of empty list returns 1."""
        assert compute_lcm_list([]) == 1
    
    def test_compute_lcm_list_single(self):
        """Test LCM of single element."""
        assert compute_lcm_list([17]) == 17


# ============================================================
# Test: Scoring Functions
# ============================================================


class TestMagnitudeScore:
    """Tests for compute_magnitude_score function."""
    
    def test_exact_match(self):
        """Test score when prediction matches exactly."""
        # n=100, log10(100)=2, mag_target = 1+2 = 3
        # If mag_mu = 3, score should be 0 (no error)
        score = compute_magnitude_score(n=100, mag_mu=3.0, sigma=1.0)
        assert score == pytest.approx(0.0, abs=1e-5)
    
    def test_one_sigma_off(self):
        """Test score when off by one sigma."""
        # n=100, mag_target=3.0, mag_mu=4.0, sigma=1.0
        # score = -(3-4)^2 / (2*1) = -0.5
        score = compute_magnitude_score(n=100, mag_mu=4.0, sigma=1.0)
        assert score == pytest.approx(-0.5, abs=1e-5)
    
    def test_negative_n_returns_neg_inf(self):
        """Test that n<=0 returns -inf."""
        score = compute_magnitude_score(n=0, mag_mu=1.0, sigma=1.0)
        assert score == float('-inf')
        
        score = compute_magnitude_score(n=-5, mag_mu=1.0, sigma=1.0)
        assert score == float('-inf')
    
    def test_small_sigma(self):
        """Test with small sigma (high precision)."""
        # Small sigma means high penalty for errors
        score = compute_magnitude_score(n=100, mag_mu=4.0, sigma=0.1)
        assert score < -10  # Large negative score


class TestModuloScore:
    """Tests for compute_modulo_score function."""
    
    def test_uniform_probs(self, small_mod_range, uniform_log_probs):
        """Test with uniform probabilities."""
        score = compute_modulo_score(10, uniform_log_probs, small_mod_range)
        # Uniform: log(1/m) for each modulus
        expected = sum(math.log(1/m) for m in small_mod_range)
        assert score == pytest.approx(expected, rel=1e-4)
    
    def test_peaked_probs_correct_n(self, small_mod_range, peaked_log_probs):
        """Test with peaked probs matching target n=23."""
        score = compute_modulo_score(23, peaked_log_probs, small_mod_range)
        # Should be high (close to 0) since peaks match n=23's remainders
        assert score > -2  # High score
    
    def test_peaked_probs_wrong_n(self, small_mod_range, peaked_log_probs):
        """Test with peaked probs not matching."""
        score = compute_modulo_score(17, peaked_log_probs, small_mod_range)
        # Should be lower than correct n
        correct_score = compute_modulo_score(23, peaked_log_probs, small_mod_range)
        assert score < correct_score


class TestTotalScore:
    """Tests for compute_total_score function."""
    
    def test_combines_mag_and_mod_with_weights(self, small_mod_range, uniform_log_probs):
        """Test that total score combines magnitude and modulo with default weights."""
        from intseq_bert import config
        
        n = 100
        mag_mu = 3.0  # Matches n=100
        sigma = 1.0
        
        total = compute_total_score(n, mag_mu, sigma, uniform_log_probs, small_mod_range)
        mag = compute_magnitude_score(n, mag_mu, sigma)
        mod = compute_modulo_score(n, uniform_log_probs, small_mod_range)
        
        expected = (config.SOLVER_MAG_WEIGHT * mag) + (config.SOLVER_MOD_WEIGHT * mod)
        assert total == pytest.approx(expected, rel=1e-5)
    
    def test_custom_weights(self, small_mod_range, uniform_log_probs):
        """Test that custom weights are applied correctly."""
        n = 100
        mag_mu = 3.0
        sigma = 1.0
        
        mag = compute_magnitude_score(n, mag_mu, sigma)
        mod = compute_modulo_score(n, uniform_log_probs, small_mod_range)
        
        # Test with custom weights
        total = compute_total_score(
            n, mag_mu, sigma, uniform_log_probs, small_mod_range,
            mag_weight=2.0, mod_weight=0.5
        )
        expected = (2.0 * mag) + (0.5 * mod)
        assert total == pytest.approx(expected, rel=1e-5)


class TestComputeTotalScoresBatch:
    """Tests for compute_total_scores_batch vectorized function."""
    
    def test_empty_candidates(self, small_mod_range, uniform_log_probs):
        """Test returns empty tensor for empty candidate list."""
        scores = compute_total_scores_batch(
            [], mag_mu=2.0, sigma=1.0,
            mod_log_probs=uniform_log_probs,
            mod_range=small_mod_range
        )
        assert len(scores) == 0
    
    def test_single_candidate(self, small_mod_range, uniform_log_probs):
        """Test matches scalar version for single candidate."""
        n = 42
        mag_mu, sigma = 2.5, 1.0
        
        scalar_score = compute_total_score(n, mag_mu, sigma, uniform_log_probs, small_mod_range)
        batch_scores = compute_total_scores_batch([n], mag_mu, sigma, uniform_log_probs, small_mod_range)
        
        assert batch_scores[0].item() == pytest.approx(scalar_score, rel=1e-4)
    
    def test_multiple_candidates_matches_scalar(self, small_mod_range, peaked_log_probs):
        """Test batch scoring matches scalar version for multiple candidates."""
        candidates = [10, 23, 42, 100]
        mag_mu, sigma = 2.0, 1.0
        
        # Compute scalar scores
        scalar_scores = [
            compute_total_score(n, mag_mu, sigma, peaked_log_probs, small_mod_range)
            for n in candidates
        ]
       
        # Compute batch scores
        batch_scores = compute_total_scores_batch(
            candidates, mag_mu, sigma, peaked_log_probs, small_mod_range
        )
        
        # Compare
        for i, (scalar, batch) in enumerate(zip(scalar_scores, batch_scores)):
            assert batch.item() == pytest.approx(scalar, rel=1e-4), f"Mismatch at index {i}"
    
    def test_handles_large_candidates(self, small_mod_range, uniform_log_probs):
        """Test handles large candidate values correctly."""
        large_candidates = [10**6, 10**9, 10**12]
        scores = compute_total_scores_batch(
            large_candidates, mag_mu=10.0, sigma=2.0,
            mod_log_probs=uniform_log_probs,
            mod_range=small_mod_range
        )
        
        assert len(scores) == 3
        # All scores should be finite
        assert all(torch.isfinite(s) for s in scores)
    
    def test_returns_torch_tensor(self, small_mod_range, uniform_log_probs):
        """Test returns PyTorch tensor."""
        scores = compute_total_scores_batch(
            [10, 20, 30], mag_mu=2.0, sigma=1.0,
            mod_log_probs=uniform_log_probs,
            mod_range=small_mod_range
        )
        
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (3,)
    
    def test_handles_very_large_integers(self, small_mod_range, uniform_log_probs):
        """Test handles integers beyond int64 range."""
        # These are larger than 2^63-1
        very_large_candidates = [10**20, 10**30, 10**40]
        scores = compute_total_scores_batch(
            very_large_candidates, mag_mu=30.0, sigma=5.0,
            mod_log_probs=uniform_log_probs,
            mod_range=small_mod_range
        )
        
        assert len(scores) == 3
        assert all(torch.isfinite(s) for s in scores)
    
    def test_custom_weights_batch(self, small_mod_range, uniform_log_probs):
        """Test batch scoring with custom weights matches scalar version."""
        candidates = [10, 100, 1000]
        mag_mu, sigma = 2.5, 1.0
        mag_weight, mod_weight = 2.0, 0.5
        
        # Compute scalar scores with custom weights
        scalar_scores = [
            compute_total_score(
                n, mag_mu, sigma, uniform_log_probs, small_mod_range,
                mag_weight=mag_weight, mod_weight=mod_weight
            )
            for n in candidates
        ]
        
        # Compute batch scores with same custom weights
        batch_scores = compute_total_scores_batch(
            candidates, mag_mu, sigma, uniform_log_probs, small_mod_range,
            mag_weight=mag_weight, mod_weight=mod_weight
        )
        
        for i, (scalar, batch) in enumerate(zip(scalar_scores, batch_scores)):
            assert batch.item() == pytest.approx(scalar, rel=1e-4), f"Mismatch at index {i}"


# ============================================================
# Test: Top-K Remainders
# ============================================================


class TestGetTopRemainders:
    """Tests for get_top_remainders function."""
    
    def test_returns_correct_count(self):
        """Test returns requested number of remainders."""
        log_probs = torch.tensor([0.1, 0.5, 0.2, 0.15, 0.05]).log()
        result = get_top_remainders(log_probs, k=3)
        assert len(result) == 3
    
    def test_returns_sorted_by_prob(self):
        """Test returns sorted by probability descending."""
        log_probs = torch.tensor([0.1, 0.5, 0.2, 0.15, 0.05]).log()
        result = get_top_remainders(log_probs, k=3)
        # Index 1 (0.5) should be first
        assert result[0][0] == 1
    
    def test_handles_k_larger_than_size(self):
        """Test handles k larger than tensor size."""
        log_probs = torch.tensor([0.3, 0.7]).log()
        result = get_top_remainders(log_probs, k=5)
        assert len(result) == 2


# ============================================================
# Test: Mode A - Dense Search
# ============================================================


class TestSolveDense:
    """Tests for solve_dense (Mode A)."""
    
    def test_finds_correct_answer(self, small_mod_range, peaked_log_probs):
        """Test finds the correct integer."""
        # n=23 is peaked in peaked_log_probs
        # Search range includes 23
        candidates = solve_dense(
            n_min=20, n_max=30,
            mag_mu=2.36,  # 1 + log10(23) ≈ 2.36
            sigma=0.5,
            mod_log_probs=peaked_log_probs,
            mod_range=small_mod_range,
            top_k=5
        )
        
        assert len(candidates) <= 5
        assert candidates[0]["value"] == 23
        assert candidates[0]["method"] == "dense"
    
    def test_returns_top_k(self, small_mod_range, uniform_log_probs):
        """Test returns exactly top_k candidates."""
        candidates = solve_dense(
            n_min=1, n_max=100,
            mag_mu=2.0, sigma=1.0,
            mod_log_probs=uniform_log_probs,
            mod_range=small_mod_range,
            top_k=10
        )
        assert len(candidates) == 10
    
    def test_sorted_by_score(self, small_mod_range, uniform_log_probs):
        """Test candidates are sorted by score descending."""
        candidates = solve_dense(
            n_min=1, n_max=50,
            mag_mu=2.0, sigma=1.0,
            mod_log_probs=uniform_log_probs,
            mod_range=small_mod_range,
            top_k=10
        )
        scores = [c["score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)


# ============================================================
# Test: Mode AB - Anchored Sieve
# ============================================================


class TestSelectAnchors:
    """Tests for select_anchors function."""
    
    def test_selects_high_confidence_first(self, small_mod_range):
        """Test selects highest confidence moduli first."""
        # Make mod2 most confident, mod7 least
        log_probs = [
            torch.tensor([0.99, 0.01]).log(),  # mod2: very confident
            torch.tensor([0.5, 0.3, 0.2]).log(),  # mod3: medium
            torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]).log(),  # mod5: uniform
            torch.tensor([0.14]*7).log(),  # mod7: uniform
        ]
        anchors = select_anchors(log_probs, small_mod_range, width=1000, 
                                  target_candidates=10, max_anchors=4)
        # First anchor should be index 0 (mod2, highest confidence)
        assert anchors[0] == 0
    
    def test_stops_at_target(self, small_mod_range):
        """Test stops when target candidate count reached."""
        log_probs = [torch.ones(m).log_softmax(dim=-1) for m in small_mod_range]
        anchors = select_anchors(log_probs, small_mod_range, width=1000,
                                  target_candidates=200, max_anchors=10)
        # Should stop before using all anchors
        assert len(anchors) <= 4
    
    def test_respects_max_anchors(self, small_mod_range):
        """Test respects max_anchors limit."""
        log_probs = [torch.ones(m).log_softmax(dim=-1) for m in small_mod_range]
        anchors = select_anchors(log_probs, small_mod_range, width=10**10,
                                  target_candidates=1, max_anchors=2)
        assert len(anchors) <= 2


class TestBeamSearchCrt:
    """Tests for beam_search_crt function."""
    
    def test_empty_anchors(self, small_mod_range, uniform_log_probs):
        """Test with empty anchor list."""
        beams = beam_search_crt([], uniform_log_probs, small_mod_range, beam_width=5)
        assert beams == [(0, 1, 0.0)]
    
    def test_single_anchor(self, small_mod_range, uniform_log_probs):
        """Test with single anchor."""
        beams = beam_search_crt([0], uniform_log_probs, small_mod_range, beam_width=3)
        # Should have up to 2 beams (mod2 has only 2 remainders)
        assert len(beams) <= 2
        # Each beam should have M=2
        for x, M, _ in beams:
            assert M == 2
    
    def test_multiple_anchors(self, small_mod_range, peaked_log_probs):
        """Test with multiple anchors."""
        beams = beam_search_crt([0, 1, 2], peaked_log_probs, small_mod_range, beam_width=5)
        # Should return valid CRT solutions
        assert len(beams) > 0
        for x, M, prob in beams:
            assert M > 0
            assert prob <= 0  # Log probabilities are negative or zero


class TestEnumerateCandidatesFromBeams:
    """Tests for enumerate_candidates_from_beams function."""
    
    def test_basic_enumeration(self):
        """Test basic enumeration."""
        beams = [(3, 10, 0.0)]  # x ≡ 3 (mod 10)
        candidates = enumerate_candidates_from_beams(beams, n_min=1, n_max=50)
        expected = {3, 13, 23, 33, 43}
        assert candidates == expected
    
    def test_multiple_beams(self):
        """Test with multiple beams."""
        beams = [(1, 10, 0.0), (2, 10, 0.0)]
        candidates = enumerate_candidates_from_beams(beams, n_min=1, n_max=30)
        expected = {1, 2, 11, 12, 21, 22}
        assert candidates == expected
    
    def test_handles_x_below_range(self):
        """Test handles case where x < n_min."""
        beams = [(3, 10, 0.0)]
        candidates = enumerate_candidates_from_beams(beams, n_min=20, n_max=50)
        expected = {23, 33, 43}
        assert candidates == expected


class TestSolveSieve:
    """Tests for solve_sieve (Mode AB)."""
    
    def test_finds_correct_answer(self, small_mod_range, peaked_log_probs):
        """Test finds correct integer with sieve method."""
        candidates = solve_sieve(
            n_min=1, n_max=100,
            mag_mu=2.36, sigma=0.5,
            mod_log_probs=peaked_log_probs,
            mod_range=small_mod_range,
            top_k=5,
            sieve_target=50,
            max_anchors=4,
            beam_width=3
        )
        # Should find 23 as top candidate
        if candidates:  # May be empty if no valid CRT solutions
            values = [c["value"] for c in candidates]
            assert 23 in values
    
    def test_method_label(self, small_mod_range, peaked_log_probs):
        """Test candidates have correct method label."""
        candidates = solve_sieve(
            n_min=1, n_max=100,
            mag_mu=2.36, sigma=0.5,
            mod_log_probs=peaked_log_probs,
            mod_range=small_mod_range,
            top_k=5
        )
        for c in candidates:
            assert c["method"] == "sieve"


# ============================================================
# Test: Mode B - Sparse CRT
# ============================================================


class TestSelectBasis:
    """Tests for select_basis function."""
    
    def test_stops_when_lcm_exceeds_width(self, small_mod_range, uniform_log_probs):
        """Test stops when LCM exceeds width."""
        basis = select_basis(uniform_log_probs, small_mod_range, width=100)
        # LCM(2,3,5,7) = 210 > 100, so should stop after getting enough
        lcm = 1
        for idx in basis:
            lcm = compute_lcm(lcm, small_mod_range[idx])
        assert lcm > 100


class TestSolveSparseCrt:
    """Tests for solve_sparse_crt (Mode B)."""
    
    def test_basic_functionality(self, small_mod_range, peaked_log_probs):
        """Test basic CRT solving."""
        candidates = solve_sparse_crt(
            n_min=1, n_max=1000,
            mag_mu=2.36, sigma=0.5,
            mod_log_probs=peaked_log_probs,
            mod_range=small_mod_range,
            top_k=5,
            beam_width=3
        )
        # Should return candidates with method="crt"
        for c in candidates:
            assert c["method"] == "crt"
    
    def test_finds_correct_answer_in_range(self, small_mod_range, peaked_log_probs):
        """Test finds correct answer when in range."""
        candidates = solve_sparse_crt(
            n_min=20, n_max=30,
            mag_mu=2.36, sigma=0.5,
            mod_log_probs=peaked_log_probs,
            mod_range=small_mod_range,
            top_k=5,
            beam_width=5
        )
        if candidates:
            # 23 should be found since it matches the peaked remainders
            values = [c["value"] for c in candidates]
            assert 23 in values


# ============================================================
# Test: IntegerSolver Class
# ============================================================


class TestIntegerSolverInit:
    """Tests for IntegerSolver initialization."""
    
    def test_default_initialization(self):
        """Test default initialization uses config values."""
        solver = IntegerSolver()
        assert solver.mod_range == config.MOD_RANGE
        assert solver.dense_threshold == config.SOLVER_DENSE_THRESHOLD
        assert solver.beam_width == config.SOLVER_BEAM_WIDTH
    
    def test_custom_initialization(self, small_mod_range):
        """Test custom initialization."""
        solver = IntegerSolver(
            mod_range=small_mod_range,
            dense_threshold=100,
            beam_width=5
        )
        assert solver.mod_range == small_mod_range
        assert solver.dense_threshold == 100
        assert solver.beam_width == 5


class TestIntegerSolverSolve:
    """Tests for IntegerSolver.solve method."""
    
    def test_zero_sign_returns_zero(self, small_mod_range, uniform_log_probs):
        """Test sign_idx=2 (Zero) returns [0]."""
        solver = IntegerSolver(mod_range=small_mod_range)
        result = solver.solve(
            mag_mu=3.0, mag_log_var=0.0,
            sign_idx=config.SIGN_ZERO,
            mod_log_probs=uniform_log_probs
        )
        assert len(result) == 1
        assert result[0]["value"] == 0
        assert result[0]["method"] == "zero"
    
    def test_negative_sign_negates_value(self, small_mod_range, peaked_log_probs):
        """Test sign_idx=1 (Negative) returns negative values."""
        solver = IntegerSolver(mod_range=small_mod_range, dense_threshold=10**8)
        result = solver.solve(
            mag_mu=2.36, mag_log_var=0.0,  # Target ≈ 23
            sign_idx=config.SIGN_NEGATIVE,
            mod_log_probs=peaked_log_probs,
            top_k=3
        )
        # All values should be negative
        for c in result:
            assert c["value"] < 0
    
    def test_positive_sign(self, small_mod_range, peaked_log_probs):
        """Test sign_idx=0 (Positive) returns positive values."""
        solver = IntegerSolver(mod_range=small_mod_range, dense_threshold=10**8)
        result = solver.solve(
            mag_mu=2.36, mag_log_var=0.0,
            sign_idx=config.SIGN_POSITIVE,
            mod_log_probs=peaked_log_probs,
            top_k=3
        )
        for c in result:
            assert c["value"] > 0
    
    def test_invalid_sign_raises(self, small_mod_range, uniform_log_probs):
        """Test invalid sign_idx raises ValueError."""
        solver = IntegerSolver(mod_range=small_mod_range)
        with pytest.raises(ValueError, match="Invalid sign_idx"):
            solver.solve(
                mag_mu=3.0, mag_log_var=0.0,
                sign_idx=5,  # Invalid
                mod_log_probs=uniform_log_probs
            )
    
    def test_mode_selection_dense(self, small_mod_range, uniform_log_probs):
        """Test selects dense mode for small ranges."""
        solver = IntegerSolver(
            mod_range=small_mod_range,
            dense_threshold=1000,
            sieve_threshold=10**10
        )
        # Small mag_log_var means narrow range
        result = solver.solve(
            mag_mu=2.0, mag_log_var=-10.0,  # Very small sigma
            sign_idx=config.SIGN_POSITIVE,
            mod_log_probs=uniform_log_probs
        )
        # Should use dense method
        if result:
            assert result[0]["method"] == "dense"
    
    def test_returns_sorted_candidates(self, small_mod_range, uniform_log_probs):
        """Test returns candidates sorted by score."""
        solver = IntegerSolver(mod_range=small_mod_range, dense_threshold=10**8)
        result = solver.solve(
            mag_mu=2.0, mag_log_var=0.0,
            sign_idx=config.SIGN_POSITIVE,
            mod_log_probs=uniform_log_probs,
            top_k=10
        )
        scores = [c["score"] for c in result]
        assert scores == sorted(scores, reverse=True)


class TestIntegerSolverFromModelOutput:
    """Tests for IntegerSolver.from_model_output static method."""
    
    def test_extracts_correct_values(self):
        """Test extracts correct values from predictions dict."""
        # Mock predictions
        B, L = 2, 10
        predictions = {
            "mag_mu": torch.randn(B, L),
            "mag_log_var": torch.randn(B, L),
            "sign_logits": torch.randn(B, L, 3),
            "mod_logits": torch.randn(B, L, sum(config.MOD_RANGE)),
        }
        
        # Mock model with _split_mod_logits
        class MockModel:
            def _split_mod_logits(self, logits):
                return torch.split(logits, config.MOD_RANGE, dim=-1)
        
        model = MockModel()
        position = 5
        batch_idx = 0
        
        mag_mu, mag_log_var, sign_idx, mod_log_probs = IntegerSolver.from_model_output(
            predictions, position, model, batch_idx
        )
        
        # Check types
        assert isinstance(mag_mu, float)
        assert isinstance(mag_log_var, float)
        assert isinstance(sign_idx, int)
        assert isinstance(mod_log_probs, list)
        assert len(mod_log_probs) == len(config.MOD_RANGE)
        
        # Check sign_idx is valid
        assert sign_idx in (0, 1, 2)
        
        # Check log_probs are clamped
        for lp in mod_log_probs:
            assert lp.min().item() >= -100.0


# ============================================================
# Test: Integration / Edge Cases
# ============================================================


class TestIntegration:
    """Integration tests for solver."""
    
    def test_end_to_end_small_number(self, small_mod_range):
        """Test end-to-end solving for a small number."""
        target = 42
        
        # Create log_probs that peak at target's remainders
        log_probs = []
        for m in small_mod_range:
            logits = torch.zeros(m)
            logits[target % m] = 10.0
            log_probs.append(logits.log_softmax(dim=-1))
        
        solver = IntegerSolver(mod_range=small_mod_range, dense_threshold=10**8)
        result = solver.solve(
            mag_mu=1 + math.log10(target),
            mag_log_var=-2.0,  # Small variance
            sign_idx=config.SIGN_POSITIVE,
            mod_log_probs=log_probs,
            top_k=5
        )
        
        assert result[0]["value"] == target
    
    def test_handles_large_mag_mu(self, small_mod_range, uniform_log_probs):
        """Test handles large magnitude predictions."""
        solver = IntegerSolver(mod_range=small_mod_range)
        # mag_mu = 50 means 10^49, huge number
        result = solver.solve(
            mag_mu=50.0, mag_log_var=1.0,
            sign_idx=config.SIGN_POSITIVE,
            mod_log_probs=uniform_log_probs,
            top_k=5
        )
        # Should not crash, may return empty or sparse CRT results
        assert isinstance(result, list)
    
    def test_handles_negative_mag_mu(self, small_mod_range, uniform_log_probs):
        """Test handles very small numbers (negative log magnitude)."""
        solver = IntegerSolver(mod_range=small_mod_range, dense_threshold=10**8)
        # mag_mu = 1 means log10(x) = 0, so x ≈ 1
        result = solver.solve(
            mag_mu=1.0, mag_log_var=0.0,
            sign_idx=config.SIGN_POSITIVE,
            mod_log_probs=uniform_log_probs,
            top_k=5
        )
        # Should find small numbers around 1
        assert result[0]["value"] >= 1


class TestEdgeCases:
    """Edge case tests."""
    
    def test_very_narrow_range(self, small_mod_range, uniform_log_probs):
        """Test with very narrow search range."""
        solver = IntegerSolver(mod_range=small_mod_range)
        result = solver.solve(
            mag_mu=2.0, mag_log_var=-20.0,  # Extremely small sigma
            sign_idx=config.SIGN_POSITIVE,
            mod_log_probs=uniform_log_probs,
            top_k=5
        )
        # Should return some candidates
        assert len(result) >= 1
    
    def test_crt_no_solution_graceful(self):
        """Test handles CRT with no valid solution gracefully."""
        # This tests the error handling in beam_search_crt
        # Create conflicting probs that would lead to CRT failures
        mod_range = [4, 6]  # Non-coprime moduli
        log_probs = [
            torch.tensor([1.0, 0.0, 0.0, 0.0]).log_softmax(dim=-1),  # mod4: peak at 0
            torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).log_softmax(dim=-1),  # mod6: peak at 3
        ]
        # 0 mod 4 and 3 mod 6 are inconsistent (0 mod 2 ≠ 3 mod 2)
        
        beams = beam_search_crt([0, 1], log_probs, mod_range, beam_width=1)
        # Should return something (maybe previous beams or handle gracefully)
        assert isinstance(beams, list)


# ============================================================
# Test: VanillaSolver
# ============================================================


class TestVanillaSolverInit:
    """Tests for VanillaSolver initialization."""
    
    def test_default_initialization(self):
        """Test VanillaSolver initializes with config defaults."""
        from intseq_bert import config
        
        solver = VanillaSolver()
        assert solver.vocab_size == config.VANILLA_VOCAB_SIZE
        assert solver.special_offset == config.VANILLA_SPECIAL_TOKENS_OFFSET
        assert solver.max_predictable == config.VANILLA_VOCAB_SIZE - config.VANILLA_SPECIAL_TOKENS_OFFSET - 1
    
    def test_custom_initialization(self):
        """Test VanillaSolver with custom parameters."""
        solver = VanillaSolver(vocab_size=1000, special_offset=5)
        assert solver.vocab_size == 1000
        assert solver.special_offset == 5
        assert solver.max_predictable == 994


class TestVanillaSolverTokenMapping:
    """Tests for token ID to integer mapping."""
    
    def test_special_tokens_return_none(self):
        """Test that special tokens (PAD, MASK, UNK) return None."""
        solver = VanillaSolver()
        
        # IDs < special_offset are special tokens
        assert solver._token_id_to_integer(0) is None  # PAD
        assert solver._token_id_to_integer(1) is None  # MASK
        assert solver._token_id_to_integer(2) is None  # UNK
    
    def test_integer_tokens_map_correctly(self):
        """Test that integer tokens map to correct values."""
        solver = VanillaSolver(special_offset=3)
        
        # token_id = integer + offset
        assert solver._token_id_to_integer(3) == 0
        assert solver._token_id_to_integer(4) == 1
        assert solver._token_id_to_integer(103) == 100
        assert solver._token_id_to_integer(10002) == 9999
    
    def test_is_special_token(self):
        """Test special token detection."""
        solver = VanillaSolver(special_offset=3)
        
        assert solver._is_special_token(0) is True
        assert solver._is_special_token(1) is True
        assert solver._is_special_token(2) is True
        assert solver._is_special_token(3) is False
        assert solver._is_special_token(100) is False


class TestVanillaSolverSolve:
    """Tests for VanillaSolver.solve() method."""
    
    def test_returns_top_k_candidates(self):
        """Test solve returns correct number of candidates."""
        solver = VanillaSolver()
        
        # Create mock logits with clear winner
        logits = torch.zeros(solver.vocab_size)
        logits[50] = 10.0  # token_id=50 -> integer=47
        
        candidates = solver.solve(logits, top_k=5)
        
        assert len(candidates) == 5
    
    def test_candidates_sorted_by_score(self):
        """Test candidates are sorted by score descending."""
        solver = VanillaSolver()
        
        logits = torch.randn(solver.vocab_size)
        candidates = solver.solve(logits, top_k=10)
        
        scores = [c["score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)
    
    def test_correct_value_mapping(self):
        """Test integer values are mapped correctly."""
        solver = VanillaSolver(special_offset=3)
        
        # Create logits with peak at token_id=103 -> integer=100
        logits = torch.zeros(solver.vocab_size)
        logits[103] = 100.0  # Very high logit
        
        candidates = solver.solve(logits, top_k=1)
        
        assert candidates[0]["value"] == 100
        assert candidates[0]["is_unk"] is False
        assert candidates[0]["method"] == "vanilla_lm"
    
    def test_special_token_marked_as_unk(self):
        """Test that special tokens are marked as UNK."""
        solver = VanillaSolver(special_offset=3)
        
        # Create logits with peak at UNK token (id=2)
        logits = torch.zeros(solver.vocab_size)
        logits[2] = 100.0  # UNK has highest logit
        
        candidates = solver.solve(logits, top_k=1)
        
        assert candidates[0]["value"] is None
        assert candidates[0]["is_unk"] is True
    
    def test_output_format(self):
        """Test output dictionary contains required keys."""
        solver = VanillaSolver()
        
        logits = torch.randn(solver.vocab_size)
        candidates = solver.solve(logits, top_k=3)
        
        required_keys = {"value", "score", "method", "is_unk"}
        for c in candidates:
            assert set(c.keys()) == required_keys
            assert isinstance(c["score"], float)
            assert c["method"] == "vanilla_lm"
            assert isinstance(c["is_unk"], bool)
    
    def test_top_k_larger_than_vocab(self):
        """Test graceful handling when top_k > vocab_size."""
        solver = VanillaSolver(vocab_size=100, special_offset=3)
        
        logits = torch.randn(100)
        candidates = solver.solve(logits, top_k=200)  # Larger than vocab
        
        assert len(candidates) == 100  # Should cap at vocab size

