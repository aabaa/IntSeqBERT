"""
Tests for the feature extraction module (Dual Model Architecture).
Tests both Magnitude and Mod Spectrum feature extraction.
"""

import pytest
import torch
import math
import numpy as np

from intseq_bert.features import (
    # Magnitude features
    compute_log_magnitude,
    compute_sign,
    compute_velocity,
    compute_acceleration,
    compute_normalized_index,
    # Mod Spectrum features
    compute_mod_residues,
    compute_mod_sin,
    compute_mod_cos,
    # Main extractor
    extract_features,
    MOD_RANGE
)


# ==========================================
# 1. Magnitude Feature Tests
# ==========================================

class TestLogMagnitude:
    """Tests for compute_log_magnitude function."""
    
    def test_positive_numbers(self):
        """Test log magnitude of positive numbers."""
        seq = [1, 10, 100, 1000]
        result = compute_log_magnitude(seq)
        
        assert result[0] == pytest.approx(0.0, abs=1e-6)  # log10(1) = 0
        assert result[1] == pytest.approx(1.0, abs=1e-6)  # log10(10) = 1
        assert result[2] == pytest.approx(2.0, abs=1e-6)  # log10(100) = 2
        assert result[3] == pytest.approx(3.0, abs=1e-6)  # log10(1000) = 3
    
    def test_negative_numbers(self):
        """Test log magnitude of negative numbers (uses abs)."""
        seq = [-1, -10, -100]
        result = compute_log_magnitude(seq)
        
        assert result[0] == pytest.approx(0.0, abs=1e-6)  # log10(|-1|) = 0
        assert result[1] == pytest.approx(1.0, abs=1e-6)  # log10(|-10|) = 1
        assert result[2] == pytest.approx(2.0, abs=1e-6)  # log10(|-100|) = 2
    
    def test_zero(self):
        """Test that zero returns 0.0."""
        seq = [0, 1, 0]
        result = compute_log_magnitude(seq)
        
        assert result[0] == 0.0
        assert result[2] == 0.0
    
    def test_empty_sequence(self):
        """Test empty sequence returns empty list."""
        assert compute_log_magnitude([]) == []


class TestSign:
    """Tests for compute_sign function."""
    
    def test_all_cases(self):
        """Test positive, negative, and zero."""
        seq = [5, -3, 0, 100, -1]
        result = compute_sign(seq)
        
        assert result == [1.0, -1.0, 0.0, 1.0, -1.0]
    
    def test_empty_sequence(self):
        """Test empty sequence."""
        assert compute_sign([]) == []


class TestVelocity:
    """Tests for compute_velocity function (1st order diff of log magnitude)."""
    
    def test_exponential_growth(self):
        """Exponential sequences should have constant velocity."""
        # 10^0, 10^1, 10^2, 10^3 -> velocity should be ~1.0
        seq = [1, 10, 100, 1000]
        result = compute_velocity(seq)
        
        assert result[0] == 0.0  # Padded first element
        assert result[1] == pytest.approx(1.0, abs=1e-6)
        assert result[2] == pytest.approx(1.0, abs=1e-6)
        assert result[3] == pytest.approx(1.0, abs=1e-6)
    
    def test_constant_sequence(self):
        """Constant sequence should have zero velocity."""
        seq = [5, 5, 5, 5]
        result = compute_velocity(seq)
        
        assert all(v == pytest.approx(0.0, abs=1e-6) for v in result)
    
    def test_empty_sequence(self):
        """Test empty sequence."""
        assert compute_velocity([]) == []

    def test_length_preservation(self):
        """Test that velocity keeps the same length as input (padding check)."""
        seq = [1, 10, 100]
        result = compute_velocity(seq)
        assert len(result) == len(seq)

class TestAcceleration:
    """Tests for compute_acceleration function (2nd order diff)."""
    
    def test_exponential_has_zero_acceleration(self):
        """Exponential sequences should have zero acceleration."""
        seq = [1, 10, 100, 1000, 10000]
        result = compute_acceleration(seq)
        
        # First two elements are padded
        assert result[0] == 0.0
        assert result[1] == 0.0
        # Rest should be ~0 for exponential
        assert result[2] == pytest.approx(0.0, abs=1e-6)
        assert result[3] == pytest.approx(0.0, abs=1e-6)
        assert result[4] == pytest.approx(0.0, abs=1e-6)
    
    def test_empty_sequence(self):
        """Test empty sequence."""
        assert compute_acceleration([]) == []


class TestNormalizedIndex:
    """Tests for compute_normalized_index function."""
    
    def test_standard_sequence(self):
        """Test normalized index for standard sequence."""
        seq = [10, 20, 30, 40, 50]  # Values don't matter
        result = compute_normalized_index(seq)
        
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[1] == pytest.approx(0.25, abs=1e-6)
        assert result[2] == pytest.approx(0.5, abs=1e-6)
        assert result[3] == pytest.approx(0.75, abs=1e-6)
        assert result[4] == pytest.approx(1.0, abs=1e-6)
    
    def test_single_element(self):
        """Single element should return [0.0]."""
        result = compute_normalized_index([42])
        assert result == [0.0]
    
    def test_empty_sequence(self):
        """Empty sequence should return empty list."""
        assert compute_normalized_index([]) == []


# ==========================================
# 2. Mod Spectrum Feature Tests
# ==========================================

class TestModResidues:
    """Tests for compute_mod_residues function."""
    
    def test_positive_numbers(self):
        """Test residues for positive numbers."""
        seq = [0, 1, 2, 3, 4, 5]
        result = compute_mod_residues(seq, 3)
        
        assert result == [0, 1, 2, 0, 1, 2]
    
    def test_negative_numbers(self):
        """Test that Python % handles negatives correctly (math convention)."""
        # -1 % 3 should be 2 in Python
        seq = [-1, -2, -3, -4, -5]
        result = compute_mod_residues(seq, 3)
        
        assert result == [2, 1, 0, 2, 1]
    
    def test_various_moduli(self):
        """Test various moduli."""
        seq = [42]
        
        assert compute_mod_residues(seq, 3) == [42 % 3]
        assert compute_mod_residues(seq, 7) == [42 % 7]
        assert compute_mod_residues(seq, 100) == [42 % 100]


class TestModSinCos:
    """Tests for compute_mod_sin and compute_mod_cos functions."""
    
    def test_mod_sin_values(self):
        """Test mod sin returns correct values."""
        # For m=4, residue 0 -> sin(0) = 0, residue 1 -> sin(pi/2) = 1
        seq = [0, 1, 2, 3]
        result = compute_mod_sin(seq, 4)
        
        assert result[0] == pytest.approx(0.0, abs=1e-6)  # sin(0)
        assert result[1] == pytest.approx(1.0, abs=1e-6)  # sin(pi/2)
        assert result[2] == pytest.approx(0.0, abs=1e-6)  # sin(pi)
        assert result[3] == pytest.approx(-1.0, abs=1e-6) # sin(3*pi/2)
    
    def test_mod_cos_values(self):
        """Test mod cos returns correct values."""
        # For m=4, residue 0 -> cos(0) = 1, residue 1 -> cos(pi/2) = 0
        seq = [0, 1, 2, 3]
        result = compute_mod_cos(seq, 4)
        
        assert result[0] == pytest.approx(1.0, abs=1e-6)  # cos(0)
        assert result[1] == pytest.approx(0.0, abs=1e-6)  # cos(pi/2)
        assert result[2] == pytest.approx(-1.0, abs=1e-6) # cos(pi)
        assert result[3] == pytest.approx(0.0, abs=1e-6)  # cos(3*pi/2)
    
    def test_sin_cos_bounds(self):
        """Test that sin/cos are always in [-1, 1]."""
        seq = list(range(-100, 101))
        
        for m in [3, 5, 7, 11, 100]:
            sin_vals = compute_mod_sin(seq, m)
            cos_vals = compute_mod_cos(seq, m)
            
            assert all(-1.0 <= v <= 1.0 for v in sin_vals)
            assert all(-1.0 <= v <= 1.0 for v in cos_vals)


# ==========================================
# 3. Main Extractor Tests
# ==========================================

class TestExtractFeatures:
    """Tests for the main extract_features function."""
    
    def test_output_structure(self):
        """Test that output has correct keys and types."""
        seq = [1, 2, 3, 4, 5]
        result = extract_features(seq)
        
        assert 'mag_features' in result
        assert 'mod_features' in result
        assert 'targets' in result
        
        assert isinstance(result['mag_features'], torch.Tensor)
        assert isinstance(result['mod_features'], torch.Tensor)
        assert isinstance(result['targets'], dict)
    
    def test_mag_features_shape(self):
        """Test magnitude features shape."""
        seq = [1, 2, 3, 4, 5]
        result = extract_features(seq)
        
        # Should be (SeqLen, 5)
        assert result['mag_features'].shape == (5, 5)
        assert result['mag_features'].dtype == torch.float32
    
    def test_mod_features_shape(self):
        """Test mod spectrum features shape."""
        seq = [1, 2, 3, 4, 5]
        result = extract_features(seq)
        
        # MOD_RANGE is 2..101 (100 moduli), each has sin+cos = 200 features
        expected_dim = len(list(MOD_RANGE)) * 2  # 100 * 2 = 200
        assert result['mod_features'].shape == (5, expected_dim)
        assert result['mod_features'].dtype == torch.float32
    
    def test_targets_structure(self):
        """Test targets dictionary structure."""
        seq = [1, 2, 3, 4, 5]
        result = extract_features(seq)
        targets = result['targets']
        
        # Should have 'mag' and all mod targets
        assert 'mag' in targets
        assert targets['mag'].dtype == torch.float32
        
        # Check some moduli targets
        for m in [2, 3, 5, 7, 100, 101]:
            key = f"mod{m}"
            assert key in targets
            assert targets[key].dtype == torch.long
            assert len(targets[key]) == 5
    
    def test_targets_correctness(self):
        """Test that targets contain correct values."""
        seq = [0, 7, 42, -5, 100]
        result = extract_features(seq)
        targets = result['targets']
        
        # Check mod3 targets
        expected_mod3 = [x % 3 for x in seq]
        assert targets['mod3'].tolist() == expected_mod3
        
        # Check mod7 targets
        expected_mod7 = [x % 7 for x in seq]
        assert targets['mod7'].tolist() == expected_mod7
        
        # Check mod100 targets
        expected_mod100 = [x % 100 for x in seq]
        assert targets['mod100'].tolist() == expected_mod100
    
    def test_empty_sequence_raises(self):
        """Test that empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="Sequence cannot be empty"):
            extract_features([])
    
    def test_single_element(self):
        """Test single element sequence."""
        seq = [42]
        result = extract_features(seq)
        
        assert result['mag_features'].shape == (1, 5)
        assert result['mod_features'].shape == (1, 200)  # 100 moduli * 2
    
    def test_large_numbers(self):
        """Test with large numbers (shouldn't overflow)."""
        seq = [10**50, 10**100, -10**75]
        result = extract_features(seq)
        
        # Should not raise, and should have valid finite values
        assert torch.isfinite(result['mag_features']).all()
        assert torch.isfinite(result['mod_features']).all()
    
    def test_negative_number_handling(self):
        """Test that negative numbers are handled correctly."""
        seq = [-5, -10, -100]
        result = extract_features(seq)
        
        # Log magnitude should use abs
        mag_log = result['mag_features'][:, 0]  # First column is log magnitude
        expected = [math.log10(5), math.log10(10), math.log10(100)]
        for i, exp in enumerate(expected):
            assert mag_log[i].item() == pytest.approx(exp, abs=1e-5)


class TestModRangeConfig:
    """Tests for MOD_RANGE configuration."""
    
    def test_mod_range_coverage(self):
        """Test that MOD_RANGE covers 2 to 101."""
        mod_list = list(MOD_RANGE)
        
        assert mod_list[0] == 2
        assert mod_list[-1] == 101
        assert len(mod_list) == 100


class TestFeatureConsistency:
    """Integration tests for feature consistency."""
    
    def test_deterministic(self):
        """Test that extraction is deterministic."""
        seq = [1, 2, 3, 42, 100, -5, 0]
        
        result1 = extract_features(seq)
        result2 = extract_features(seq)
        
        assert torch.equal(result1['mag_features'], result2['mag_features'])
        assert torch.equal(result1['mod_features'], result2['mod_features'])
        
        for key in result1['targets']:
            assert torch.equal(result1['targets'][key], result2['targets'][key])
    
    def test_mod_features_in_bounds(self):
        """Test that all mod features (sin/cos) are in [-1, 1]."""
        seq = list(range(-1000, 1001))
        result = extract_features(seq)
        
        mod_feats = result['mod_features']
        assert (mod_feats >= -1.0).all()
        assert (mod_feats <= 1.0).all()