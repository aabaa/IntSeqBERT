"""
Tests for IntSeqBERT model.
"""

import pytest
import torch
import torch.nn as nn

from intseq_bert.bert_model import IntSeqBERT, PositionalEncoding


@pytest.fixture
def sample_model():
    """Create a small IntSeqBERT model for testing."""
    return IntSeqBERT(
        input_dim=35,
        d_model=64,  # Smaller for faster tests
        nhead=4,
        num_layers=2,  # Fewer layers for faster tests
        dim_feedforward=128,
        max_len=100,
        dropout=0.1
    )


@pytest.fixture
def dummy_batch():
    """Create dummy batch tensors for testing."""
    batch_size = 2
    seq_len = 10
    input_dim = 35
    
    inputs = torch.randn(batch_size, seq_len, input_dim)
    attention_mask = torch.ones(batch_size, seq_len)  # All valid
    labels = torch.randn(batch_size, seq_len, input_dim)
    mask_matrix = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask_matrix[:, 2:5] = True  # Mask positions 2, 3, 4
    
    return {
        "inputs": inputs,
        "attention_mask": attention_mask,
        "labels": labels,
        "mask_matrix": mask_matrix
    }


def test_positional_encoding():
    """Test positional encoding module."""
    d_model = 64
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    
    # Test forward pass
    x = torch.randn(2, 10, d_model)
    output = pe(x)
    
    assert output.shape == x.shape
    assert output.dtype == x.dtype


def test_model_initialization():
    """Test that model can be initialized with default parameters."""
    model = IntSeqBERT()
    
    # Check model exists and has expected components
    assert isinstance(model, nn.Module)
    assert hasattr(model, 'input_proj')
    assert hasattr(model, 'pos_encoder')
    assert hasattr(model, 'encoder')
    assert hasattr(model, 'prediction_head')
    
    # Check dimensions
    assert model.input_dim == 35
    assert model.d_model == 128
    
    # Check model can be moved to device (basic sanity check)
    device = torch.device('cpu')
    model = model.to(device)
    assert next(model.parameters()).device.type == 'cpu'


def test_forward_inference(sample_model, dummy_batch):
    """Test forward pass in inference mode (no labels)."""
    model = sample_model
    inputs = dummy_batch["inputs"]
    attention_mask = dummy_batch["attention_mask"]
    
    # Forward without labels (inference mode)
    with torch.no_grad():
        output = model(inputs, attention_mask)
    
    # Check output structure
    assert "prediction" in output
    assert "loss" in output
    
    # Check prediction shape
    prediction = output["prediction"]
    assert prediction.shape == inputs.shape  # (batch, seq_len, input_dim)
    assert prediction.dtype == torch.float32
    
    # Check loss is None in inference mode
    assert output["loss"] is None


def test_forward_training(sample_model, dummy_batch):
    """Test forward pass in training mode (with labels and loss)."""
    model = sample_model
    inputs = dummy_batch["inputs"]
    attention_mask = dummy_batch["attention_mask"]
    labels = dummy_batch["labels"]
    mask_matrix = dummy_batch["mask_matrix"]
    
    # Forward with labels (training mode)
    output = model(inputs, attention_mask, labels=labels, mask_matrix=mask_matrix)
    
    # Check output structure
    assert "prediction" in output
    assert "loss" in output
    
    # Check prediction shape
    prediction = output["prediction"]
    assert prediction.shape == labels.shape
    
    # Check loss is a scalar
    loss = output["loss"]
    assert loss is not None
    assert loss.ndim == 0  # Scalar
    assert loss.dtype == torch.float32
    assert loss.item() >= 0  # MSE loss should be non-negative


def test_masked_loss_calculation(sample_model):
    """Test that loss is computed only on masked positions."""
    model = sample_model
    batch_size = 2
    seq_len = 10
    input_dim = 35
    
    # Create inputs where prediction equals labels
    inputs = torch.randn(batch_size, seq_len, input_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randn(batch_size, seq_len, input_dim)
    
    # Create specific mask pattern
    mask_matrix = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask_matrix[0, 2] = True  # Only one position masked
    
    # Forward pass
    output = model(inputs, attention_mask, labels=labels, mask_matrix=mask_matrix)
    loss = output["loss"]
    
    # Loss should be computed (non-zero since prediction != label)
    assert loss is not None
    assert loss.item() >= 0
    
    # Manual verification: compute expected loss
    prediction = output["prediction"]
    masked_pred = prediction[0, 2, :]  # The masked position
    masked_label = labels[0, 2, :]
    expected_loss = ((masked_pred - masked_label) ** 2).mean()
    
    # Should be close to manually computed loss
    assert torch.isclose(loss, expected_loss, rtol=1e-4)


def test_edge_case_no_mask(sample_model, dummy_batch):
    """Test edge case when no positions are masked."""
    model = sample_model
    inputs = dummy_batch["inputs"]
    attention_mask = dummy_batch["attention_mask"]
    labels = dummy_batch["labels"]
    
    # Create mask with all False (no positions masked)
    mask_matrix = torch.zeros_like(attention_mask, dtype=torch.bool)
    
    # Forward pass
    output = model(inputs, attention_mask, labels=labels, mask_matrix=mask_matrix)
    loss = output["loss"]
    
    # Loss should be 0.0 (not NaN or error)
    assert loss is not None
    assert loss.item() == 0.0
    assert not torch.isnan(loss)


def test_gradient_flow(sample_model, dummy_batch):
    """Test that gradients flow properly through the model."""
    model = sample_model
    inputs = dummy_batch["inputs"]
    attention_mask = dummy_batch["attention_mask"]
    labels = dummy_batch["labels"]
    mask_matrix = dummy_batch["mask_matrix"]
    
    # Ensure model is in training mode
    model.train()
    
    # Zero gradients
    model.zero_grad()
    
    # Forward pass
    output = model(inputs, attention_mask, labels=labels, mask_matrix=mask_matrix)
    loss = output["loss"]
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist for key parameters
    assert model.input_proj.weight.grad is not None
    assert model.input_proj.weight.grad.abs().sum() > 0  # Non-zero gradients
    
    # Check gradients exist for prediction head
    for param in model.prediction_head.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_attention_mask_handling(sample_model):
    """Test that attention mask properly handles padding."""
    model = sample_model
    batch_size = 2
    seq_len = 10
    input_dim = 35
    
    inputs = torch.randn(batch_size, seq_len, input_dim)
    
    # Create attention mask with padding
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, 7:] = 0  # Pad last 3 positions of first sequence
    attention_mask[1, 9:] = 0  # Pad last position of second sequence
    
    # Forward pass
    with torch.no_grad():
        output = model(inputs, attention_mask)
    
    # Should complete without errors
    assert output["prediction"].shape == inputs.shape
    
    # Predictions at padded positions should still be computed
    # (but they won't be used in loss computation)
    pred = output["prediction"]
    assert not torch.isnan(pred).any()


def test_batch_size_flexibility(sample_model):
    """Test that model handles different batch sizes."""
    model = sample_model
    input_dim = 35
    
    # Test with different batch sizes
    for batch_size in [1, 2, 8]:
        seq_len = 10
        inputs = torch.randn(batch_size, seq_len, input_dim)
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            output = model(inputs, attention_mask)
        
        assert output["prediction"].shape == (batch_size, seq_len, input_dim)


def test_sequence_length_flexibility(sample_model):
    """Test that model handles different sequence lengths."""
    model = sample_model
    batch_size = 2
    input_dim = 35
    
    # Test with different sequence lengths
    for seq_len in [5, 10, 50]:
        inputs = torch.randn(batch_size, seq_len, input_dim)
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            output = model(inputs, attention_mask)
        
        assert output["prediction"].shape == (batch_size, seq_len, input_dim)


def test_load_from_checkpoint(tmp_path):
    """Test loading a model from checkpoint file."""
    # Create a model and save a checkpoint
    model = IntSeqBERT(
        input_dim=35,
        d_model=64,
        nhead=4,
        num_layers=2
    )
    
    # Create a checkpoint
    checkpoint_path = tmp_path / "test_checkpoint.pt"
    checkpoint = {
        "epoch": 5,
        "model_state_dict": model.state_dict(),
        "config": {
            "input_dim": 35,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 512,
            "max_len": 5000,
            "dropout": 0.1
        },
        "train_loss": 0.5,
        "val_loss": 0.6
    }
    torch.save(checkpoint, checkpoint_path)
    
    # Load the model using classmethod (force CPU for consistent comparison)
    loaded_model, loaded_checkpoint = IntSeqBERT.load_from_checkpoint(
        str(checkpoint_path), device='cpu'
    )
    
    # Move original model to CPU for comparison
    model = model.cpu()
    
    # Verify model architecture matches
    assert loaded_model.input_dim == 35
    assert loaded_model.d_model == 64
    
    # Verify checkpoint data is returned
    assert loaded_checkpoint["epoch"] == 5
    assert loaded_checkpoint["train_loss"] == 0.5
    
    # Verify state dicts match
    original_params = model.state_dict()
    loaded_params = loaded_model.state_dict()
    
    for key in original_params.keys():
        assert torch.allclose(original_params[key], loaded_params[key])


def test_load_from_checkpoint_with_device(tmp_path):
    """Test loading with explicit device specification."""
    model = IntSeqBERT(d_model=32, nhead=2, num_layers=1)
    
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "d_model": 32,
            "nhead": 2,
            "num_layers": 1
        }
    }
    torch.save(checkpoint, checkpoint_path)
    
    # Load with explicit device (cpu always available)
    loaded_model, _ = IntSeqBERT.load_from_checkpoint(str(checkpoint_path), device='cpu')
    
    # Verify model is on CPU
    assert next(loaded_model.parameters()).device.type == 'cpu'

