"""
Tests for solver.py (Encoder-Decoder Hybrid Solver).
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
# 3. IntSeqSolver Tests
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
# 4. _decoder_beam_search Tests
# ==========================================

class TestDecoderBeamSearch:
    """Tests for _decoder_beam_search internal method."""
    
    @pytest.fixture
    def solver_instance(self):
        model = IntSeqBERT()
        return solver.IntSeqSolver(model=model, device="cpu")
    
    def test_returns_list_of_tuples(self, solver_instance):
        """Test that _decoder_beam_search returns list of (value, score) tuples."""
        # Create minimal predictions dict
        predictions = {
            "mag_mu": 1.0,
            "mag_logvar": math.log(0.2**2),
            "sign_logits": np.array([-10.0, -10.0, 10.0])  # Positive
        }
        # Add mod probs (high confidence)
        for m in range(2, 102):
            probs = np.zeros(m)
            probs[1 % m] = 0.9
            probs[0] = 0.1 / (m - 1) if m > 1 else 0.1
            predictions[f"mod{m}"] = probs
        
        result = solver_instance._decoder_beam_search(predictions)
        
        assert isinstance(result, list)
        if len(result) > 0:
            assert isinstance(result[0], tuple)
            assert len(result[0]) == 2
    
    def test_handles_zero_sign(self, solver_instance):
        """Test that zero sign returns [(0, 0.0)]."""
        predictions = {
            "mag_mu": 1.0,
            "mag_logvar": math.log(0.2**2),
            "sign_logits": np.array([-10.0, 10.0, -10.0])  # Zero (index 1)
        }
        for m in range(2, 102):
            predictions[f"mod{m}"] = np.ones(m) / m
        
        result = solver_instance._decoder_beam_search(predictions)
        
        assert result == [(0, 0.0)]
    
    def test_positive_sign_returns_positive_values(self, solver_instance):
        """Test that positive sign produces positive candidate values."""
        predictions = {
            "mag_mu": 1.0,
            "mag_logvar": math.log(0.2**2),
            "sign_logits": np.array([-10.0, -10.0, 10.0])  # Positive
        }
        for m in range(2, 102):
            probs = np.zeros(m)
            probs[1 % m] = 0.9
            for i in range(m):
                if i != 1 % m:
                    probs[i] = 0.1 / (m - 1)
            predictions[f"mod{m}"] = probs
        
        result = solver_instance._decoder_beam_search(predictions)
        
        values = [r[0] for r in result]
        assert all(v > 0 for v in values if v != 0)
    
    def test_negative_sign_returns_negative_values(self, solver_instance):
        """Test that negative sign produces negative candidate values."""
        predictions = {
            "mag_mu": 1.0,
            "mag_logvar": math.log(0.2**2),
            "sign_logits": np.array([10.0, -10.0, -10.0])  # Negative
        }
        for m in range(2, 102):
            probs = np.zeros(m)
            probs[1 % m] = 0.9
            for i in range(m):
                if i != 1 % m:
                    probs[i] = 0.1 / (m - 1)
            predictions[f"mod{m}"] = probs
        
        result = solver_instance._decoder_beam_search(predictions)
        
        values = [r[0] for r in result]
        assert all(v < 0 for v in values if v != 0)


# ==========================================
# 5. Integration Tests
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
