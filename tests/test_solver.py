"""
Tests for solver.py (Robust Bayesian Beam Search Solver).
"""

import pytest
import torch
import numpy as np
import math
from unittest.mock import MagicMock, patch

from intseq_bert import solver
from intseq_bert.bert_model import IntSeqBERT


# ==========================================
# 1. extended_gcd Tests
# ==========================================

class TestExtendedGCD:
    """Tests for extended_gcd function."""
    
    def test_basic_gcd(self):
        """Test basic GCD calculation."""
        d, x, y = solver.extended_gcd(15, 10)
        assert d == 5
    
    def test_coefficients_satisfy_equation(self):
        """Test that ax + by = gcd holds."""
        a, b = 48, 18
        d, x, y = solver.extended_gcd(a, b)
        assert d == 6
        assert a * x + b * y == d
    
    def test_coprime_numbers(self):
        """Test with coprime numbers."""
        d, x, y = solver.extended_gcd(17, 13)
        assert d == 1
        assert 17 * x + 13 * y == 1
    
    def test_zero_input(self):
        """Test with zero input."""
        d, x, y = solver.extended_gcd(0, 5)
        assert d == 5


# ==========================================
# 2. solve_congruence Tests
# ==========================================

class TestSolveCongruence:
    """Tests for solve_congruence (Generalized CRT)."""
    
    def test_coprime_moduli(self):
        """Test with coprime moduli."""
        x, lcm = solver.solve_congruence(2, 3, 3, 5)
        assert lcm == 15
        assert x % 3 == 2
        assert x % 5 == 3
    
    def test_non_coprime_consistent(self):
        """Test with non-coprime moduli that are consistent."""
        x, lcm = solver.solve_congruence(2, 6, 8, 9)
        assert lcm == 18
        assert x % 6 == 2
        assert x % 9 == 8
    
    def test_non_coprime_inconsistent(self):
        """Test with inconsistent congruences."""
        x, lcm = solver.solve_congruence(1, 4, 2, 6)
        assert x is None
    
    def test_same_modulus(self):
        """Test with same modulus."""
        x, lcm = solver.solve_congruence(3, 7, 3, 7)
        assert lcm == 7
        assert x == 3


# ==========================================
# 3. calculate_magnitude_log_prob Tests
# ==========================================

class TestCalculateMagnitudeLogProb:
    """Tests for calculate_magnitude_log_prob function."""
    
    def test_exact_match(self):
        """Test when value exactly matches target."""
        score = solver.calculate_magnitude_log_prob(100, 2.0)
        assert score == pytest.approx(0.0, abs=1e-5)
    
    def test_one_sigma_error(self):
        """Test with one sigma error."""
        score = solver.calculate_magnitude_log_prob(100, 2.2, sigma=0.2)
        expected = -0.5 * 1.0 ** 2
        assert score == pytest.approx(expected, rel=1e-3)
    
    def test_large_error(self):
        """Test with large error yields very negative score."""
        score = solver.calculate_magnitude_log_prob(10, 5.0, sigma=0.2)
        assert score < -100
    
    def test_zero_value(self):
        """Test with zero value."""
        score = solver.calculate_magnitude_log_prob(0, 2.0)
        assert score < 0
    
    def test_negative_value(self):
        """Test with negative value (uses absolute)."""
        score = solver.calculate_magnitude_log_prob(-100, 2.0)
        assert score == pytest.approx(0.0, abs=1e-5)


# ==========================================
# 4. beam_search_robust Tests
# ==========================================

class TestBeamSearchRobust:
    """Tests for beam_search_robust function."""
    
    def test_simple_case(self):
        """Test with simple mod probabilities."""
        mod_probs = {
            2: np.array([0.0, 1.0]),  # x = 1 (mod 2), high confidence
            3: np.array([0.0, 0.0, 1.0])  # x = 2 (mod 3), high confidence
        }
        result = solver.beam_search_robust(mod_probs, pred_log_mag=0.7, pred_sign=1.0)
        
        values = [r[0] for r in result]
        assert 5 in values or -1 in values
    
    def test_filters_low_confidence(self):
        """Test that low confidence moduli are filtered out."""
        mod_probs = {
            2: np.array([0.5, 0.5]),  # max=0.5, above threshold
            3: np.array([0.35, 0.35, 0.30]),  # max=0.35, below threshold (0.4)
        }
        result = solver.beam_search_robust(mod_probs, pred_log_mag=1.0, pred_sign=1.0)
        # Should still produce results (mod 2 is used)
        assert len(result) > 0
    
    def test_returns_scored_tuples(self):
        """Test that result contains (value, score) tuples."""
        mod_probs = {2: np.array([0.1, 0.9])}  # high confidence
        result = solver.beam_search_robust(mod_probs, pred_log_mag=1.0, pred_sign=1.0)
        
        assert len(result) > 0
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2
    
    def test_sorted_by_score(self):
        """Test that results are sorted by score descending."""
        mod_probs = {
            2: np.array([0.2, 0.8]),
            5: np.array([0.1, 0.1, 0.6, 0.1, 0.1])
        }
        result = solver.beam_search_robust(mod_probs, pred_log_mag=1.0, pred_sign=1.0)
        
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)
    
    def test_deduplication(self):
        """Test that duplicate values are removed."""
        mod_probs = {2: np.array([0.3, 0.7])}  # above threshold
        result = solver.beam_search_robust(mod_probs, pred_log_mag=1.0, pred_sign=1.0)
        
        values = [r[0] for r in result]
        assert len(values) == len(set(values))
    
    def test_sign_filtering_positive(self):
        """Test that positive sign filters out negative values."""
        mod_probs = {2: np.array([0.1, 0.9])}
        result = solver.beam_search_robust(mod_probs, pred_log_mag=1.0, pred_sign=0.5)  # > 0.2 → positive
        
        values = [r[0] for r in result]
        # All values should be positive
        assert all(v > 0 for v in values)
    
    def test_sign_filtering_negative(self):
        """Test that negative sign filters out positive values."""
        mod_probs = {2: np.array([0.1, 0.9])}
        result = solver.beam_search_robust(mod_probs, pred_log_mag=1.0, pred_sign=-0.5)  # < -0.2 → negative
        
        values = [r[0] for r in result]
        assert all(v < 0 for v in values)


# ==========================================
# 5. IntSeqSolver Tests
# ==========================================

class TestIntSeqSolverInit:
    """Tests for IntSeqSolver initialization."""
    
    def test_init_with_model(self):
        """Test initialization with model object."""
        model = IntSeqBERT()
        s = solver.IntSeqSolver(model=model, device="cpu")
        
        assert s.model is not None
        assert s.device == "cpu"
    
    def test_init_requires_model(self):
        """Test that initialization requires model or path."""
        with pytest.raises(ValueError):
            solver.IntSeqSolver()


class TestIntSeqSolverSolve:
    """Tests for IntSeqSolver.solve method."""
    
    @pytest.fixture
    def solver_instance(self):
        """Create solver with mock-like behavior."""
        model = IntSeqBERT()
        return solver.IntSeqSolver(model=model, device="cpu")
    
    def test_solve_returns_dict(self, solver_instance):
        """Test that solve returns expected dict structure."""
        result = solver_instance.solve([1, 2, 3, 4, 5])
        
        assert "candidates" in result
        assert "predicted_magnitude" in result
    
    def test_solve_candidates_format(self, solver_instance):
        """Test candidates are list of (value, score) tuples."""
        result = solver_instance.solve([1, 1, 2, 3, 5])
        
        candidates = result["candidates"]
        assert isinstance(candidates, list)
        if len(candidates) > 0:
            assert isinstance(candidates[0], tuple)
            assert len(candidates[0]) == 2
    
    def test_solve_respects_top_k(self, solver_instance):
        """Test that solve returns at most top_k candidates."""
        result = solver_instance.solve([1, 2, 3, 4, 5], top_k=3)
        
        assert len(result["candidates"]) <= 3


# ==========================================
# 6. Integration Tests
# ==========================================

class TestSolverIntegration:
    """Integration tests for the full solver pipeline."""
    
    def test_full_pipeline_smoke(self):
        """Smoke test for full solving pipeline."""
        model = IntSeqBERT()
        s = solver.IntSeqSolver(model=model, device="cpu")
        
        result = s.solve([1, 1, 2, 3, 5, 8, 13], top_k=5)
        
        assert "candidates" in result
        assert "predicted_magnitude" in result
        assert isinstance(result["predicted_magnitude"], float)
    
    def test_handles_varied_sequences(self):
        """Test solver handles different sequence types."""
        model = IntSeqBERT()
        s = solver.IntSeqSolver(model=model, device="cpu")
        
        result1 = s.solve([2, 4, 6, 8, 10])
        assert "candidates" in result1
        
        result2 = s.solve([1, 2, 4, 8, 16])
        assert "candidates" in result2
        
        result3 = s.solve([1, 4, 9, 16, 25])
        assert "candidates" in result3
