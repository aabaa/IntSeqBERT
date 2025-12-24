"""
Tests for the number-theoretic decoder model.
"""

import pytest
import torch
import math

from intseq_bert.decoder_model import NumberTheoreticDecoder
from intseq_bert.features import extract_features, log_magnitude


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
    assert decoder.input_dim == 35
    assert decoder.hidden_dim == 256
    
    # Count parameters
    num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    assert num_params > 0


def test_forward_output_structure():
    """Test forward pass returns correct output structure."""
    decoder = NumberTheoreticDecoder()
    decoder.eval()
    
    # Single vector input
    x = torch.randn(35)
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
    assert output["mag"].shape == (1, 4096)  # Now 4096 bins instead of 1
    assert output["mod3"].shape == (1, 3)
    assert output["mod5"].shape == (1, 5)
    assert output["mod8"].shape == (1, 8)
    assert output["mod10"].shape == (1, 10)
    
    # Batch input
    x_batch = torch.randn(4, 35)
    with torch.no_grad():
        output_batch = decoder(x_batch)
    
    assert output_batch["sign"].shape == (4, 3)
    assert output_batch["mag"].shape == (4, 4096)  # Now 4096 bins


def test_batch_reconstruct_simple_numbers():
    """Test batch reconstruction of simple numbers."""
    decoder = NumberTheoreticDecoder()
    decoder.eval()
    
    # Note: Untrained decoder won't give perfect results,
    # but we test the reconstruction mechanism works
    test_numbers = [0, 1, -1, 5, -5, 10]
    
    # Extract all features at once
    all_features = [extract_features([num])[0] for num in test_numbers]
    features_batch = torch.tensor(all_features, dtype=torch.float32)  # (6, 35)
    
    # Batch reconstruct
    with torch.no_grad():
        reconstructed, confidences = decoder.batch_reconstruct(features_batch)
    
    # Check outputs
    assert len(reconstructed) == len(test_numbers)
    assert len(confidences) == len(test_numbers)
    
    for i, (recon, conf) in enumerate(zip(reconstructed, confidences)):
        # Check types - batch_reconstruct returns tensors
        assert isinstance(recon.item(), int)
        assert isinstance(conf, (float, torch.Tensor))
        
        # Confidence should be a reasonable number
        conf_val = conf.item() if isinstance(conf, torch.Tensor) else conf
        assert not math.isnan(conf_val)
        assert not math.isinf(conf_val)


def test_batch_reconstruct_deterministic():
    """Test that batch reconstruction is deterministic in eval mode."""
    decoder = NumberTheoreticDecoder()
    decoder.eval()
    
    # Test with a batch of numbers
    test_numbers = [7, 42, 100]
    all_features = [extract_features([num])[0] for num in test_numbers]
    features_batch = torch.tensor(all_features, dtype=torch.float32)
    
    # Run reconstruction multiple times
    with torch.no_grad():
        reconstructed1, confidences1 = decoder.batch_reconstruct(features_batch)
        reconstructed2, confidences2 = decoder.batch_reconstruct(features_batch)
    
    # Should be deterministic
    assert torch.equal(reconstructed1, reconstructed2)
    for c1, c2 in zip(confidences1, confidences2):
        c1_val = c1.item() if isinstance(c1, torch.Tensor) else c1
        c2_val = c2.item() if isinstance(c2, torch.Tensor) else c2
        assert c1_val == pytest.approx(c2_val, abs=1e-5)


def test_batch_reconstruct_edge_case_zero():
    """Test batch reconstruction handles zero correctly."""
    decoder = NumberTheoreticDecoder()
    decoder.eval()
    
    features = extract_features([0])
    features_batch = torch.tensor(features, dtype=torch.float32)  # (1, 35)
    
    with torch.no_grad():
        reconstructed, confidences = decoder.batch_reconstruct(features_batch)
    
    # Should return tensor with one integer
    assert len(reconstructed) == 1
    assert isinstance(reconstructed[0].item(), int)
    # For untrained decoder, just check it runs without error
    conf_val = confidences[0].item() if isinstance(confidences[0], torch.Tensor) else confidences[0]
    assert not math.isnan(conf_val)


def test_batch_reconstruct_with_top_k():
    """Test batch reconstruction with different top_k values."""
    decoder = NumberTheoreticDecoder()
    decoder.eval()
    
    num = 50
    features = extract_features([num])
    features_batch = torch.tensor(features, dtype=torch.float32)
    
    # Test with different top_k_bins values
    with torch.no_grad():
        recon1, conf1 = decoder.batch_reconstruct(features_batch, top_k_bins=5)
        recon2, conf2 = decoder.batch_reconstruct(features_batch, top_k_bins=20)
    
    # Both should complete without error
    assert len(recon1) == 1
    assert len(recon2) == 1
    assert isinstance(recon1[0].item(), int)
    assert isinstance(recon2[0].item(), int)


def test_batch_reconstruct_with_neighbors():
    """Test batch reconstruction with different neighbor values."""
    decoder = NumberTheoreticDecoder()
    decoder.eval()
    
    num = 50
    features = extract_features([num])
    features_batch = torch.tensor(features, dtype=torch.float32)
    
    # Test with different neighbor values
    with torch.no_grad():
        recon1, conf1 = decoder.batch_reconstruct(features_batch, neighbors=1)
        recon2, conf2 = decoder.batch_reconstruct(features_batch, neighbors=3)
    
    # Both should complete without error
    assert len(recon1) == 1
    assert len(recon2) == 1
    assert isinstance(recon1[0].item(), int)
    assert isinstance(recon2[0].item(), int)


def test_batch_reconstruct_eval_mode():
    """Test that batch reconstruction works regardless of initial model mode."""
    decoder = NumberTheoreticDecoder()
    
    # Start in train mode
    decoder.train()
    assert decoder.training
    
    features = torch.randn(2, 35)
    
    # Batch reconstruction should work
    with torch.no_grad():
        reconstructed, confidences = decoder.batch_reconstruct(features)
    
    assert len(reconstructed) == 2
    assert len(confidences) == 2
    
    # Model should still be in train mode after
    assert decoder.training


def test_batch_forward():
    """Test forward pass with batch input."""
    decoder = NumberTheoreticDecoder()
    decoder.eval()
    
    batch_size = 8
    x = torch.randn(batch_size, 35)
    
    with torch.no_grad():
        output = decoder(x)
    
    # All outputs should have batch dimension
    for key in ["sign", "mag", "mod3", "mod5", "mod8", "mod10"]:
        assert output[key].shape[0] == batch_size


def test_batch_reconstruct_large_batch():
    """Test batch reconstruction with larger batches for performance."""
    decoder = NumberTheoreticDecoder()
    decoder.eval()
    
    # Create a batch of random feature vectors
    batch_size = 32
    features_batch = torch.randn(batch_size, 35)
    
    # Should handle large batches efficiently
    with torch.no_grad():
        reconstructed, confidences = decoder.batch_reconstruct(features_batch)
    
    assert len(reconstructed) == batch_size
    assert len(confidences) == batch_size
    
    for recon, conf in zip(reconstructed, confidences):
        assert isinstance(recon.item(), int)
        conf_val = conf.item() if isinstance(conf, torch.Tensor) else conf
        assert isinstance(conf_val, float)
        assert not math.isnan(conf_val)
        assert not math.isinf(conf_val)
