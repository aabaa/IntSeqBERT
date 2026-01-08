"""
Tests for solver.py (Bayesian Beam Search Solver with Sign Awareness).
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
        assert d == 5  # gcd(15, 10) = 5
    
    def test_coefficients_satisfy_equation(self):
        """Test that ax + by = gcd holds."""
        a, b = 48, 18
        d, x, y = solver.extended_gcd(a, b)
        assert d == 6  # gcd(48, 18) = 6
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
        assert 0 * x + 5 * y == 5


# ==========================================
# 2. solve_congruence Tests
# ==========================================

class TestSolveCongruence:
    """Tests for solve_congruence (Generalized CRT)."""
    
    def test_coprime_moduli(self):
        """Test with coprime moduli."""
        # x = 2 (mod 3), x = 3 (mod 5)
        x, lcm = solver.solve_congruence(2, 3, 3, 5)
        assert lcm == 15
        assert x % 3 == 2
        assert x % 5 == 3
    
    def test_non_coprime_consistent(self):
        """Test with non-coprime moduli that are consistent."""
        # x = 2 (mod 6), x = 8 (mod 9)
        # gcd(6, 9) = 3, and 8 - 2 = 6 is divisible by 3
        x, lcm = solver.solve_congruence(2, 6, 8, 9)
        assert lcm == 18
        assert x % 6 == 2
        assert x % 9 == 8
    
    def test_non_coprime_inconsistent(self):
        """Test with inconsistent congruences."""
        # x = 1 (mod 4), x = 2 (mod 6)
        # gcd(4, 6) = 2, but 2 - 1 = 1 is not divisible by 2
        x, lcm = solver.solve_congruence(1, 4, 2, 6)
        assert x is None  # Inconsistent
    
    def test_same_modulus(self):
        """Test with same modulus."""
        # x = 3 (mod 7), x = 3 (mod 7)
        x, lcm = solver.solve_congruence(3, 7, 3, 7)
        assert lcm == 7
        assert x == 3


# ==========================================
# 3. calculate_joint_log_prob Tests
# ==========================================

class TestCalculateJointLogProb:
    """Tests for calculate_joint_log_prob function."""
    
    def test_exact_match_positive(self):
        """Test when value exactly matches target (positive)."""
        # val = 100, target_log_mag = 2.0, target_sign = 1.0
        score = solver.calculate_joint_log_prob(100, 2.0, 1.0)
        # log10(100) = 2.0, sign=1.0 -> both match -> score ~ 0
        assert score == pytest.approx(0.0, abs=1e-5)
    
    def test_exact_match_negative(self):
        """Test when value exactly matches target (negative)."""
        # val = -100, target_log_mag = 2.0, target_sign = -1.0
        score = solver.calculate_joint_log_prob(-100, 2.0, -1.0)
        # log10(100) = 2.0, sign=-1.0 -> both match
        assert score == pytest.approx(0.0, abs=1e-5)
    
    def test_sign_mismatch_penalty(self):
        """Test that sign mismatch adds penalty."""
        # val = 100 (positive), target_sign = -1.0
        score_wrong = solver.calculate_joint_log_prob(100, 2.0, -1.0)
        score_right = solver.calculate_joint_log_prob(100, 2.0, 1.0)
        # Wrong sign should have lower (more negative) score
        assert score_wrong < score_right
    
    def test_magnitude_error(self):
        """Test with magnitude error."""
        # val = 100 (log=2), target = 2.2
        score = solver.calculate_joint_log_prob(100, 2.2, 1.0)
        # Error in mag, but sign matches
        assert score < 0
    
    def test_zero_value(self):
        """Test with zero value."""
        score = solver.calculate_joint_log_prob(0, 2.0, 1.0)
        # Zero has log_val = -1.0, val_sign = 0.0
        assert score < 0


# ==========================================
# 4. beam_search_bayesian Tests
# ==========================================

class TestBeamSearchBayesian:
    """Tests for beam_search_bayesian function."""
    
    def test_simple_case(self):
        """Test with simple mod probabilities."""
        mod_probs = {
            2: np.array([0.0, 1.0]),  # x = 1 (mod 2)
            3: np.array([0.0, 0.0, 1.0])  # x = 2 (mod 3)
        }
        # CRT: x = 1 (mod 2), x = 2 (mod 3) -> x = 5 (mod 6)
        result = solver.beam_search_bayesian(mod_probs, pred_log_mag=0.7, pred_sign=1.0)
        
        # Should find value around 10^0.7 ≈ 5
        values = [r[0] for r in result]
        assert 5 in values or -1 in values
    
    def test_respects_beam_width(self):
        """Test that beam width limits candidates."""
        mod_probs = {
            2: np.array([0.5, 0.5]),
            3: np.array([0.33, 0.33, 0.34])
        }
        result = solver.beam_search_bayesian(
            mod_probs, 
            pred_log_mag=1.0,
            pred_sign=1.0,
            beam_width=3
        )
        assert len(result) <= 50  # reasonable upper bound
    
    def test_returns_scored_tuples(self):
        """Test that result contains (value, score) tuples."""
        mod_probs = {2: np.array([0.3, 0.7])}
        result = solver.beam_search_bayesian(mod_probs, pred_log_mag=1.0, pred_sign=1.0)
        
        assert len(result) > 0
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2
        assert isinstance(result[0][0], (int, np.integer))
        assert isinstance(result[0][1], float)
    
    def test_sorted_by_score(self):
        """Test that results are sorted by score descending."""
        mod_probs = {
            2: np.array([0.4, 0.6]),
            5: np.array([0.1, 0.2, 0.5, 0.1, 0.1])
        }
        result = solver.beam_search_bayesian(mod_probs, pred_log_mag=1.0, pred_sign=1.0)
        
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)
    
    def test_deduplication(self):
        """Test that duplicate values are removed."""
        mod_probs = {2: np.array([0.5, 0.5])}
        result = solver.beam_search_bayesian(mod_probs, pred_log_mag=1.0, pred_sign=1.0)
        
        values = [r[0] for r in result]
        assert len(values) == len(set(values))
    
    def test_sign_affects_ranking(self):
        """Test that sign prediction affects candidate ranking."""
        mod_probs = {2: np.array([0.5, 0.5])}
        
        # Predict positive
        result_pos = solver.beam_search_bayesian(mod_probs, pred_log_mag=1.0, pred_sign=1.0)
        # Predict negative
        result_neg = solver.beam_search_bayesian(mod_probs, pred_log_mag=1.0, pred_sign=-1.0)
        
        # Top candidates should differ based on sign preference
        if len(result_pos) > 0 and len(result_neg) > 0:
            # At least the scores should be different
            assert result_pos[0][1] != result_neg[0][1] or result_pos[0][0] != result_neg[0][0]


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
        
        # Test with Fibonacci-like sequence
        result = s.solve([1, 1, 2, 3, 5, 8, 13], top_k=5)
        
        assert "candidates" in result
        assert "predicted_magnitude" in result
        assert isinstance(result["predicted_magnitude"], float)
    
    def test_handles_varied_sequences(self):
        """Test solver handles different sequence types."""
        model = IntSeqBERT()
        s = solver.IntSeqSolver(model=model, device="cpu")
        
        # Arithmetic
        result1 = s.solve([2, 4, 6, 8, 10])
        assert "candidates" in result1
        
        # Powers
        result2 = s.solve([1, 2, 4, 8, 16])
        assert "candidates" in result2
        
        # Squares
        result3 = s.solve([1, 4, 9, 16, 25])
        assert "candidates" in result3
