"""
Tests for IntSeqSolver (Beam Search + CRT solver).
"""

import pytest
import torch
import numpy as np
import math
from pathlib import Path

from intseq_bert import solver, bert_model


# ==========================================
# 1. Feature Computation Tests
# ==========================================

class TestMagnitudeFeatures:
    """Tests for compute_magnitude_features function."""
    
    def test_basic_sequence(self):
        """Test feature computation for basic sequence."""
        seq = [1, 2, 3]
        features = solver.compute_magnitude_features(seq)
        
        assert len(features) == 3
        assert len(features[0]) == 5
    
    def test_log_magnitude(self):
        """Test log magnitude computation."""
        seq = [10, 100, 1000]
        features = solver.compute_magnitude_features(seq)
        
        # log10(10+1) ≈ 1.04, log10(100+1) ≈ 2.00, log10(1000+1) ≈ 3.00
        assert features[0][0] == pytest.approx(math.log10(11), rel=1e-5)
        assert features[1][0] == pytest.approx(math.log10(101), rel=1e-5)
        assert features[2][0] == pytest.approx(math.log10(1001), rel=1e-5)
    
    def test_sign_feature(self):
        """Test sign computation."""
        seq = [5, -3, 0]
        features = solver.compute_magnitude_features(seq)
        
        assert features[0][1] == 1   # positive
        assert features[1][1] == -1  # negative
        assert features[2][1] == 0   # zero
    
    def test_diff_features(self):
        """Test difference features."""
        seq = [10, 15, 12]
        features = solver.compute_magnitude_features(seq)
        
        # First element has no diff
        assert features[0][2] == 0  # diff_log
        assert features[0][3] == 0  # diff_sign
        
        # Second: diff = 15 - 10 = 5 (positive)
        assert features[1][2] == pytest.approx(math.log10(6), rel=1e-5)
        assert features[1][3] == 1
        
        # Third: diff = 12 - 15 = -3 (negative)
        assert features[2][2] == pytest.approx(math.log10(4), rel=1e-5)
        assert features[2][3] == -1
    
    def test_position_feature(self):
        """Test position normalization."""
        seq = [1, 2, 3, 4, 5]
        features = solver.compute_magnitude_features(seq)
        
        assert features[0][4] == 0.00
        assert features[1][4] == 0.01
        assert features[4][4] == 0.04


class TestModFeatures:
    """Tests for compute_mod_features function."""
    
    def test_basic_modulo(self):
        """Test basic modulo computation."""
        seq = [7]
        features = solver.compute_mod_features(seq)
        
        assert len(features) == 1
        assert len(features[0]) == 200  # 100 mods duplicated
        
        # 7 % 2 = 1, 7 % 3 = 1, 7 % 5 = 2, 7 % 7 = 0
        assert features[0][0] == 1.0   # mod 2
        assert features[0][1] == 1.0   # mod 3
        assert features[0][3] == 2.0   # mod 5
        assert features[0][5] == 0.0   # mod 7
    
    def test_negative_numbers(self):
        """Test that Python's % works correctly for negatives."""
        seq = [-7]
        features = solver.compute_mod_features(seq)
        
        # Python: -7 % 3 = 2 (positive result)
        assert features[0][1] == 2.0  # mod 3
    
    def test_large_number(self):
        """Test with large number."""
        seq = [12345678]
        features = solver.compute_mod_features(seq)
        
        assert features[0][0] == float(12345678 % 2)
        assert features[0][99] == float(12345678 % 101)


# ==========================================
# 2. Beam Search CRT Tests
# ==========================================

class TestBeamSearchCRT:
    """Tests for beam_search_crt function."""
    
    def test_single_prime_certain(self):
        """Test with single prime and certain prediction."""
        # Probability 1.0 for remainder 1 mod 2
        mod_probs = {2: np.array([0.0, 1.0])}
        
        candidates = solver.beam_search_crt(mod_probs, [2])
        
        assert len(candidates) >= 1
        # Should have remainder 1 mod 2
        assert candidates[0][0] == 1
        assert candidates[0][1] == 2
    
    def test_two_primes(self):
        """Test CRT with two primes."""
        # x ≡ 1 (mod 2), x ≡ 2 (mod 3) → x ≡ 5 (mod 6)
        mod_probs = {
            2: np.array([0.0, 1.0]),
            3: np.array([0.0, 0.0, 1.0])
        }
        
        candidates = solver.beam_search_crt(mod_probs, [2, 3])
        
        assert len(candidates) >= 1
        assert candidates[0][0] == 5
        assert candidates[0][1] == 6
    
    def test_beam_width_limits(self):
        """Test that beam width limits candidates."""
        # Uniform probabilities
        mod_probs = {
            2: np.array([0.5, 0.5]),
            3: np.array([0.33, 0.33, 0.34])
        }
        
        candidates = solver.beam_search_crt(mod_probs, [2, 3], beam_width=2)
        
        assert len(candidates) <= 2
    
    def test_probability_threshold(self):
        """Test probability threshold filtering."""
        mod_probs = {
            2: np.array([1e-10, 1.0])  # First is below threshold
        }
        
        candidates = solver.beam_search_crt(
            mod_probs, [2], prob_threshold=1e-5
        )
        
        # Only remainder 1 should survive
        remainders = [c[0] for c in candidates]
        assert 0 not in remainders
    
    def test_empty_primes(self):
        """Test with no primes."""
        candidates = solver.beam_search_crt({}, [])
        
        # Should return initial candidate (0, 1, 0.0)
        assert len(candidates) == 1
        assert candidates[0] == (0, 1, 0.0)


# ==========================================
# 3. Magnitude Matching Tests
# ==========================================

class TestMagnitudeMatching:
    """Tests for magnitude_matching function."""
    
    def test_exact_match(self):
        """Test when target matches exactly."""
        # Candidate: x ≡ 5 (mod 6), target = 5
        candidates = [(5, 6, 0.0)]
        
        results = solver.magnitude_matching(
            candidates,
            target_magnitude=5,
            pred_log_magnitude=math.log10(6),
            top_k=3
        )
        
        assert 5 in [r[0] for r in results]
    
    def test_magnitude_offset(self):
        """Test finding value with k offset."""
        # x ≡ 1 (mod 10), target = 31
        candidates = [(1, 10, 0.0)]
        
        results = solver.magnitude_matching(
            candidates,
            target_magnitude=31,
            pred_log_magnitude=math.log10(32),
            top_k=5
        )
        
        # Should find 31 (= 1 + 3*10)
        assert 31 in [r[0] for r in results]
    
    def test_deduplication(self):
        """Test that duplicate values are removed."""
        # Two candidates that resolve to same value
        candidates = [
            (5, 6, -1.0),
            (5, 6, -2.0),  # Same remainder/modulus, different score
        ]
        
        results = solver.magnitude_matching(
            candidates,
            target_magnitude=5,
            pred_log_magnitude=math.log10(6),
            top_k=5
        )
        
        # Value 5 should appear only once
        values = [r[0] for r in results]
        assert values.count(5) == 1
    
    def test_top_k_limit(self):
        """Test that top_k limits output."""
        candidates = [(i, 100, -float(i)) for i in range(10)]
        
        results = solver.magnitude_matching(
            candidates,
            target_magnitude=50,
            pred_log_magnitude=math.log10(51),
            top_k=3
        )
        
        assert len(results) <= 3


# ==========================================
# 4. IntSeqSolver Class Tests
# ==========================================

class TestIntSeqSolverInit:
    """Tests for IntSeqSolver initialization."""
    
    def test_init_with_model(self):
        """Test initialization with pre-loaded model."""
        model = bert_model.IntSeqBERT(
            d_model=32, num_layers=1, multitask=True
        )
        
        solver_obj = solver.IntSeqSolver(model=model, device='cpu')
        
        assert solver_obj.model is not None
        assert solver_obj.device == 'cpu'
        assert len(solver_obj.primes) == 26
    
    def test_init_requires_model(self):
        """Test that either model or model_path is required."""
        with pytest.raises(ValueError):
            solver.IntSeqSolver()
    
    def test_custom_primes(self):
        """Test custom prime list."""
        model = bert_model.IntSeqBERT(
            d_model=32, num_layers=1, multitask=True
        )
        custom_primes = [2, 3, 5, 7]
        
        solver_obj = solver.IntSeqSolver(
            model=model, device='cpu', primes=custom_primes
        )
        
        assert solver_obj.primes == custom_primes


class TestIntSeqSolverPreprocess:
    """Tests for IntSeqSolver preprocessing."""
    
    @pytest.fixture
    def solver_instance(self):
        model = bert_model.IntSeqBERT(
            d_model=32, num_layers=1, multitask=True
        )
        return solver.IntSeqSolver(model=model, device='cpu')
    
    def test_output_shapes(self, solver_instance):
        """Test output tensor shapes."""
        seq = [1, 2, 3, 4, 5]
        mag, mod, mask = solver_instance.preprocess_sequence(seq, max_len=10)
        
        assert mag.shape == (1, 10, 5)
        assert mod.shape == (1, 10, 200)
        assert mask.shape == (1, 10)
    
    def test_padding(self, solver_instance):
        """Test sequence padding."""
        seq = [1, 2, 3]
        mag, mod, mask = solver_instance.preprocess_sequence(seq, max_len=5)
        
        # First 3 positions are valid
        assert mask[0, 0].item() == 1.0
        assert mask[0, 2].item() == 1.0
        # Last 2 are padding
        assert mask[0, 3].item() == 0.0
        assert mask[0, 4].item() == 0.0
    
    def test_truncation(self, solver_instance):
        """Test sequence truncation (keep end)."""
        seq = list(range(20))
        mag, mod, mask = solver_instance.preprocess_sequence(seq, max_len=10)
        
        # All positions should be valid
        assert mask.sum().item() == 10.0


class TestIntSeqSolverSolve:
    """Tests for IntSeqSolver.solve method."""
    
    @pytest.fixture
    def solver_instance(self):
        model = bert_model.IntSeqBERT(
            d_model=32, num_layers=1, multitask=True
        )
        return solver.IntSeqSolver(model=model, device='cpu', primes=[2, 3, 5])
    
    def test_solve_returns_dict(self, solver_instance):
        """Test that solve returns expected structure."""
        seq = [1, 2, 3, 4, 5]
        result = solver_instance.solve(seq, top_k=3)
        
        assert 'candidates' in result
        assert 'predicted_magnitude' in result
    
    def test_solve_candidates_format(self, solver_instance):
        """Test candidates format."""
        seq = [1, 2, 3, 4, 5]
        result = solver_instance.solve(seq, top_k=3)
        
        candidates = result['candidates']
        assert isinstance(candidates, list)
        
        if len(candidates) > 0:
            # Each candidate is (value, magnitude_error)
            assert len(candidates[0]) == 2
            assert isinstance(candidates[0][0], (int, float))
            assert isinstance(candidates[0][1], float)
    
    def test_solve_respects_top_k(self, solver_instance):
        """Test that top_k limits candidates."""
        seq = [1, 2, 3, 4, 5]
        result = solver_instance.solve(seq, top_k=2)
        
        assert len(result['candidates']) <= 2


# ==========================================
# 5. Integration Tests
# ==========================================

class TestSolverIntegration:
    """Integration tests with full model."""
    
    def test_full_pipeline_smoke(self):
        """Smoke test for full solver pipeline."""
        model = bert_model.IntSeqBERT(
            d_model=32, num_layers=1, multitask=True
        )
        
        solver_obj = solver.IntSeqSolver(
            model=model, device='cpu', primes=[2, 3, 5, 7]
        )
        
        # Fibonacci-like sequence
        seq = [1, 1, 2, 3, 5, 8, 13]
        result = solver_obj.solve(seq, top_k=5, beam_width=10)
        
        assert 'candidates' in result
        assert 'predicted_magnitude' in result
        assert isinstance(result['predicted_magnitude'], float)
