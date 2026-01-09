"""
Tests for features.py module.

Covers:
1. Magnitude features (log10 scale + sign one-hot)
2. Modulo features (Sin/Cos embedding + integer labels)
3. Main process_sequence function
4. Edge cases (empty, large numbers, truncation)
"""

import pytest
import torch
import math

from intseq_bert import features, config


# ==========================================
# 1. Magnitude Features Tests
# ==========================================

class TestMagnitudeFeatures:
    """Tests for compute_magnitude_features function."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        seq = [1, 2, 3, 4, 5]
        result = features.compute_magnitude_features(seq)
        
        assert result.shape == (5, config.MAG_RAW_DIM)
        assert result.dtype == torch.float32
    
    def test_zero_value(self):
        """Test that x=0 produces log_val=0 and sign=[0,0,1]."""
        seq = [0]
        result = features.compute_magnitude_features(seq)
        
        # [log_val, sign+, sign-, sign0]
        expected = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        assert torch.allclose(result, expected)
    
    def test_positive_value(self):
        """Test that positive x has correct log and sign+."""
        seq = [10]  # log10(10) = 1, so 1 + 1 = 2.0
        result = features.compute_magnitude_features(seq)
        
        log_val = result[0, 0].item()
        signs = result[0, 1:].tolist()
        
        assert abs(log_val - 2.0) < 1e-6  # 1 + log10(10) = 2
        assert signs == [1.0, 0.0, 0.0]   # Positive sign
    
    def test_negative_value(self):
        """Test that negative x has correct log and sign-."""
        seq = [-100]  # log10(100) = 2, so 1 + 2 = 3.0
        result = features.compute_magnitude_features(seq)
        
        log_val = result[0, 0].item()
        signs = result[0, 1:].tolist()
        
        assert abs(log_val - 3.0) < 1e-6  # 1 + log10(100) = 3
        assert signs == [0.0, 1.0, 0.0]   # Negative sign
    
    def test_value_one(self):
        """Test that x=1 produces log_val=1.0 (since 1 + log10(1) = 1)."""
        seq = [1]
        result = features.compute_magnitude_features(seq)
        
        log_val = result[0, 0].item()
        assert abs(log_val - 1.0) < 1e-6
    
    def test_large_number(self):
        """Test overflow protection for extremely large numbers."""
        # 10^1000 is too large for float64
        large_num = 10 ** 1000
        seq = [large_num]
        
        result = features.compute_magnitude_features(seq)
        
        # Fallback: float(len(str(abs(x)))) = len("1" + "0"*1000) = 1001
        log_val = result[0, 0].item()
        assert abs(log_val - 1001.0) < 1e-6
    
    def test_empty_sequence(self):
        """Test that empty sequence returns (0, MAG_RAW_DIM) tensor."""
        seq = []
        result = features.compute_magnitude_features(seq)
        
        assert result.shape == (0, config.MAG_RAW_DIM)


# ==========================================
# 2. Modulo Features Tests
# ==========================================

class TestModuloFeatures:
    """Tests for compute_modulo_features function."""
    
    def test_output_shapes(self):
        """Test that outputs have correct shapes."""
        seq = [1, 2, 3]
        mod_features, mod_integers = features.compute_modulo_features(seq)
        
        assert mod_features.shape == (3, config.MOD_FEATURE_DIM)
        assert mod_integers.shape == (3, config.NUM_MODULI)
        assert mod_features.dtype == torch.float32
        assert mod_integers.dtype == torch.long
    
    def test_sin_cos_values(self):
        """Test Sin/Cos computation for known value."""
        # For x=0: r = 0 % m = 0, theta = 0, sin=0, cos=1
        seq = [0]
        mod_features, _ = features.compute_modulo_features(seq)
        
        # Check first modulus (m=2): sin(0)=0, cos(0)=1
        sin_val = mod_features[0, 0].item()
        cos_val = mod_features[0, 1].item()
        
        assert abs(sin_val - 0.0) < 1e-6
        assert abs(cos_val - 1.0) < 1e-6
    
    def test_integer_remainders(self):
        """Test that integer remainders are computed correctly."""
        seq = [5]
        _, mod_integers = features.compute_modulo_features(seq)
        
        # 5 % 2 = 1, 5 % 3 = 2, 5 % 4 = 1, 5 % 5 = 0, ...
        assert mod_integers[0, 0].item() == 1  # 5 % 2
        assert mod_integers[0, 1].item() == 2  # 5 % 3
        assert mod_integers[0, 2].item() == 1  # 5 % 4
        assert mod_integers[0, 3].item() == 0  # 5 % 5
    
    def test_negative_number_modulo(self):
        """Test that negative numbers produce positive remainders."""
        seq = [-5]
        _, mod_integers = features.compute_modulo_features(seq)
        
        # Python: -5 % 3 = 1 (not -2)
        assert mod_integers[0, 1].item() == 1  # -5 % 3
    
    def test_empty_sequence(self):
        """Test that empty sequence returns (0, D) tensors."""
        seq = []
        mod_features, mod_integers = features.compute_modulo_features(seq)
        
        assert mod_features.shape == (0, config.MOD_FEATURE_DIM)
        assert mod_integers.shape == (0, config.NUM_MODULI)


# ==========================================
# 3. Process Sequence Tests (Main Entry Point)
# ==========================================

class TestProcessSequence:
    """Tests for process_sequence function."""
    
    def test_output_keys(self):
        """Test that output contains all required keys."""
        seq = [1, 2, 3]
        result = features.process_sequence(seq)
        
        assert config.KEY_MAG_FEATURES in result
        assert config.KEY_MOD_FEATURES in result
        assert config.KEY_MOD_INTEGERS in result
    
    def test_output_shapes(self):
        """Test that all output tensors have correct shapes."""
        seq = [1, 2, 3, 4, 5]
        result = features.process_sequence(seq)
        
        L = 5
        assert result[config.KEY_MAG_FEATURES].shape == (L, config.MAG_RAW_DIM)
        assert result[config.KEY_MOD_FEATURES].shape == (L, config.MOD_FEATURE_DIM)
        assert result[config.KEY_MOD_INTEGERS].shape == (L, config.NUM_MODULI)
    
    def test_truncation(self):
        """Test that sequences longer than MAX_SEQUENCE_LENGTH are truncated."""
        # Create sequence longer than limit
        long_seq = list(range(config.MAX_SEQUENCE_LENGTH + 100))
        result = features.process_sequence(long_seq)
        
        # Should be truncated to MAX_SEQUENCE_LENGTH
        assert result[config.KEY_MAG_FEATURES].shape[0] == config.MAX_SEQUENCE_LENGTH
        assert result[config.KEY_MOD_FEATURES].shape[0] == config.MAX_SEQUENCE_LENGTH
        assert result[config.KEY_MOD_INTEGERS].shape[0] == config.MAX_SEQUENCE_LENGTH
    
    def test_no_padding(self):
        """Test that short sequences are NOT padded."""
        short_seq = [1, 2, 3]
        result = features.process_sequence(short_seq)
        
        # Length should be exactly 3, not padded to MAX_SEQUENCE_LENGTH
        assert result[config.KEY_MAG_FEATURES].shape[0] == 3
    
    def test_empty_sequence(self):
        """Test that empty sequence returns (0, D) tensors."""
        seq = []
        result = features.process_sequence(seq)
        
        assert result[config.KEY_MAG_FEATURES].shape == (0, config.MAG_RAW_DIM)
        assert result[config.KEY_MOD_FEATURES].shape == (0, config.MOD_FEATURE_DIM)
        assert result[config.KEY_MOD_INTEGERS].shape == (0, config.NUM_MODULI)


# ==========================================
# 4. Integration Tests
# ==========================================

class TestIntegration:
    """Integration tests ensuring features work with downstream components."""
    
    def test_output_compatible_with_collator(self):
        """Test that output matches OEISDataset -> OEISCollator contract."""
        seq = [0, 1, -1, 10, -100, 1000]
        result = features.process_sequence(seq)
        
        # Verify types and ranges expected by collator
        assert result[config.KEY_MAG_FEATURES].dtype == torch.float32
        assert result[config.KEY_MOD_FEATURES].dtype == torch.float32
        assert result[config.KEY_MOD_INTEGERS].dtype == torch.long
        
        # mod_integers should be in valid range for each modulus
        mod_ints = result[config.KEY_MOD_INTEGERS]
        for i, m in enumerate(config.MOD_RANGE):
            col = mod_ints[:, i]
            assert (col >= 0).all()
            assert (col < m).all()
    
    def test_sin_cos_on_unit_circle(self):
        """Test that Sin/Cos values are on unit circle."""
        seq = [1, 2, 3]
        result = features.process_sequence(seq)
        
        mod_feats = result[config.KEY_MOD_FEATURES]
        
        # For each modulus, sin^2 + cos^2 should equal 1
        for i in range(config.NUM_MODULI):
            sin_col = mod_feats[:, 2*i]
            cos_col = mod_feats[:, 2*i + 1]
            
            magnitudes = sin_col ** 2 + cos_col ** 2
            assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)