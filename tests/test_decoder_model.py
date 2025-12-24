"""
Tests for the number-theoretic decoder model.
"""

import pytest
import torch
import math

from intseq_bert.decoder_model import NumberTheoreticDecoder, inverse_magnitude
from intseq_bert.features import extract_features, log_magnitude


def test_inverse_magnitude():
    """Test inverse magnitude transformation."""
    # Below threshold
    assert inverse_magnitude(0.3) == 0.0
    assert inverse_magnitude(0.49) == 0.0
    
    # At threshold
    assert inverse_magnitude(0.5) == pytest.approx(math.exp(0.5 - 1.0), rel=1e-5)
    
    # Above threshold
    y = 2.5
    result = inverse_magnitude(y)
    expected = math.exp(y - 1.0)
    assert result == pytest.approx(expected, rel=1e-5)
    
    # Test inverse property for positive numbers
    for x in [1, 5, 10, 100]:
        # log_magnitude([x]) returns list
        y = log_magnitude([x])[0]
        x_reconstructed = inverse_magnitude(y)
        assert x_reconstructed == pytest.approx(x, rel=0.01)


def test_decoder_initialization():
    """Test decoder can be initialized with default parameters."""
    decoder = NumberTheoreticDecoder()
    
    # Check architecture components exist
    assert hasattr(decoder, 'shared_encoder')
    assert hasattr(decoder, 'sign_head')
    assert hasattr(decoder, 'mag_head')
    assert hasattr(decoder, 'mod3_head')
    assert hasattr(decoder, 'mod5_head')
    assert hasattr(decoder, 'mod8_head')
    assert hasattr(decoder, 'mod10_head')
    
    # Check dimensions
    assert decoder.input_dim == 27
    assert decoder.hidden_dim == 256
    
    # Count parameters
    num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    assert num_params > 0


def test_forward_output_structure():
    """Test forward pass returns correct output structure."""
    decoder = NumberTheoreticDecoder()
    decoder.eval()
    
    # Single vector input
    x = torch.randn(27)
    with torch.no_grad():
        output = decoder(x)
    
    # Check all keys present
    assert "sign" in output
    assert "mag" in output
    assert "mod3" in output
    assert "mod5" in output
    assert "mod8" in output
    assert "mod10" in output
    
    # Check shapes
    assert output["sign"].shape == (1, 3)
    assert output["mag"].shape == (1, 1)
    assert output["mod3"].shape == (1, 3)
    assert output["mod5"].shape == (1, 5)
    assert output["mod8"].shape == (1, 8)
    assert output["mod10"].shape == (1, 10)
    
    # Batch input
    x_batch = torch.randn(4, 27)
    with torch.no_grad():
        output_batch = decoder(x_batch)
    
    assert output_batch["sign"].shape == (4, 3)
    assert output_batch["mag"].shape == (4, 1)


def test_reconstruct_simple_numbers():
    """Test reconstruction of simple numbers."""
    decoder = NumberTheoreticDecoder()
    
    # Note: Untrained decoder won't give perfect results,
    # but we test the reconstruction mechanism works
    test_numbers = [0, 1, -1, 5, -5, 10]
    
    for num in test_numbers:
        # Extract features
        features = extract_features([num])  # (1, 27)
        features_tensor = torch.tensor(features[0], dtype=torch.float32)
        
        # Reconstruct
        reconstructed, confidence = decoder.reconstruct_value(features_tensor)
        
        # Check types
        assert isinstance(reconstructed, int)
        assert isinstance(confidence, float)
        
        # Confidence should be a reasonable number
        assert not math.isnan(confidence)
        assert not math.isinf(confidence)


def test_reconstruct_with_perfect_predictions():
    """Test reconstruction when predictions are perfect."""
    decoder = NumberTheoreticDecoder()
    
    # We'll manually set the decoder to output perfect predictions for a known number
    # This tests the CRT search logic, not the learned weights
    
    test_num = 42
    features = extract_features([test_num])
    features_tensor = torch.tensor(features[0], dtype=torch.float32)
    
    # Mock the decoder's forward to return perfect predictions
    original_forward = decoder.forward
    
    def mock_forward(x):
        # Perfect predictions for 42
        # sign: positive (class 2)
        # mag: log_magnitude of 42
        # mod3: 42 % 3 = 0
        # mod5: 42 % 5 = 2
        # mod8: 42 % 8 = 2
        # mod10: 42 % 10 = 2
        
        batch_size = x.shape[0] if x.dim() > 1 else 1
        
        # Create one-hot logits (high value for correct class)
        sign_logits = torch.full((batch_size, 3), -10.0)
        sign_logits[:, 2] = 10.0  # Positive
        
        mod3_logits = torch.full((batch_size, 3), -10.0)
        mod3_logits[:, 0] = 10.0
        
        mod5_logits = torch.full((batch_size, 5), -10.0)
        mod5_logits[:, 2] = 10.0
        
        mod8_logits = torch.full((batch_size, 8), -10.0)
        mod8_logits[:, 2] = 10.0
        
        mod10_logits = torch.full((batch_size, 10), -10.0)
        mod10_logits[:, 2] = 10.0
        
        mag_value = torch.tensor([[log_magnitude([test_num])[0]]], dtype=torch.float32)
        
        return {
            "sign": sign_logits,
            "mag": mag_value,
            "mod3": mod3_logits,
            "mod5": mod5_logits,
            "mod8": mod8_logits,
            "mod10": mod10_logits
        }
    
    decoder.forward = mock_forward
    
    # Reconstruct with perfect predictions
    reconstructed, confidence = decoder.reconstruct_value(features_tensor)
    
    # Should reconstruct perfectly or very close
    assert abs(reconstructed - test_num) <= 1  # Allow ±1 error due to rounding
    assert confidence > 0  # Should have positive confidence
    
    # Restore original forward
    decoder.forward = original_forward


def test_confidence_scoring():
    """Test that confidence metric is computed correctly."""
    decoder = NumberTheoreticDecoder()
    
    # Test with a number
    num = 7
    features = extract_features([num])
    features_tensor = torch.tensor(features[0], dtype=torch.float32)
    
    # Run reconstruction multiple times (should be deterministic in eval mode)
    reconstructed1, confidence1 = decoder.reconstruct_value(features_tensor)
    reconstructed2, confidence2 = decoder.reconstruct_value(features_tensor)
    
    # Should be deterministic
    assert reconstructed1 == reconstructed2
    assert confidence1 == pytest.approx(confidence2, abs=1e-5)


def test_edge_case_zero():
    """Test reconstruction of zero."""
    decoder = NumberTheoreticDecoder()
    
    features = extract_features([0])
    features_tensor = torch.tensor(features[0], dtype=torch.float32)
    
    reconstructed, confidence = decoder.reconstruct_value(features_tensor)
    
    # Should be integer
    assert isinstance(reconstructed, int)
    # For untrained decoder, just check it runs without error
    assert not math.isnan(confidence)


def test_search_window_parameter():
    """Test that search_window parameter affects search range."""
    decoder = NumberTheoreticDecoder()
    
    num = 50
    features = extract_features([num])
    features_tensor = torch.tensor(features[0], dtype=torch.float32)
    
    # Small window
    recon1, conf1 = decoder.reconstruct_value(features_tensor, search_window=10)
    
    # Large window
    recon2, conf2 = decoder.reconstruct_value(features_tensor, search_window=500)
    
    # Both should complete without error
    assert isinstance(recon1, int)
    assert isinstance(recon2, int)
    
    # Different windows might give different results (or same if within range)
    # Just verify both are valid integers


def test_decoder_eval_mode():
    """Test that decoder uses eval mode during reconstruction."""
    decoder = NumberTheoreticDecoder()
    
    # Start in train mode
    decoder.train()
    assert decoder.training
    
    features = torch.randn(27)
    
    # Reconstruction should work regardless of initial mode
    reconstructed, confidence = decoder.reconstruct_value(features)
    
    # Should still be in train mode after (original state preserved)
    assert decoder.training


def test_batch_forward():
    """Test forward pass with batch input."""
    decoder = NumberTheoreticDecoder()
    decoder.eval()
    
    batch_size = 8
    x = torch.randn(batch_size, 27)
    
    with torch.no_grad():
        output = decoder(x)
    
    # All outputs should have batch dimension
    for key in ["sign", "mag", "mod3", "mod5", "mod8", "mod10"]:
        assert output[key].shape[0] == batch_size
