"""
test_train.py:
Unit tests for IntSeqBERT training module.
Tests EarlyStopping, prepare_labels, evaluate, and training utilities.
"""

import pytest
import torch
import torch.nn as nn

from intseq_bert import config
from intseq_bert.train import (
    EarlyStopping,
    set_seed,
    prepare_labels,
    evaluate
)
from intseq_bert.models import IntSeqForPreTraining


# ==========================================
# Test Fixtures
# ==========================================

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def seq_len():
    return 16

@pytest.fixture
def mock_collator_output(batch_size, seq_len):
    """
    Creates a mock output from OEISCollator.
    This simulates what the collator produces.
    """
    # Lengths for each sample (variable length before padding)
    lengths = [seq_len - i for i in range(batch_size)]
    
    # mag_inputs: (B, L, 5) - includes is_masked flag
    mag_inputs = torch.randn(batch_size, seq_len, config.MAG_EXTENDED_DIM)
    
    # mod_inputs: (B, L, 200)
    mod_inputs = torch.randn(batch_size, seq_len, config.MOD_FEATURE_DIM)
    
    # mag_labels: (B, L, 4) - [log_val, sign+, sign-, sign0]
    mag_labels = torch.zeros(batch_size, seq_len, config.MAG_RAW_DIM)
    mag_labels[:, :, 0] = torch.randn(batch_size, seq_len)  # log_val
    # Set one-hot signs randomly
    for b in range(batch_size):
        for l in range(seq_len):
            sign_idx = torch.randint(0, 3, (1,)).item()
            mag_labels[b, l, 1 + sign_idx] = 1.0
    
    # mod_labels: (B, L, 100) - integer remainders
    mod_labels = torch.stack([
        torch.randint(0, m, (batch_size, seq_len))
        for m in config.MOD_RANGE
    ], dim=-1)
    
    # attention_mask: (B, L) - 1 for valid, 0 for padding
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    for b, length in enumerate(lengths):
        attention_mask[b, length:] = 0
    
    # mask_matrix: (B, L) - True where prediction is needed
    mask_matrix = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask_matrix[:, :8] = True  # First 8 positions masked
    # Don't mask padding
    for b, length in enumerate(lengths):
        mask_matrix[b, length:] = False
    
    return {
        "mag_inputs": mag_inputs,
        "mod_inputs": mod_inputs,
        "mag_labels": mag_labels,
        "mod_labels": mod_labels,
        "attention_mask": attention_mask,
        "mask_matrix": mask_matrix,
        "oeis_ids": [f"A{100000 + i}" for i in range(batch_size)]
    }

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def small_model():
    """Small model for fast testing."""
    return IntSeqForPreTraining(d_model=32, nhead=2, num_layers=1)


# ==========================================
# EarlyStopping Tests
# ==========================================

class TestEarlyStopping:
    """Tests for EarlyStopping class."""
    
    def test_initial_state(self):
        """Test initial state is correct."""
        es = EarlyStopping(patience=5, delta=0.0)
        assert es.patience == 5
        assert es.counter == 0
        assert es.best_loss == float("inf")
        assert es.early_stop == False
    
    def test_improvement_resets_counter(self):
        """Test that improvement resets the counter."""
        es = EarlyStopping(patience=3)
        
        # Initial call with loss
        result = es(1.0)
        assert result == False
        assert es.counter == 0
        assert es.best_loss == 1.0
        
        # Improvement
        result = es(0.5)
        assert result == False
        assert es.counter == 0
        assert es.best_loss == 0.5
    
    def test_no_improvement_increments_counter(self):
        """Test that no improvement increments counter."""
        es = EarlyStopping(patience=3)
        
        es(1.0)  # Set baseline
        
        # No improvement
        result = es(1.0)
        assert result == True
        assert es.counter == 1
        
        result = es(1.1)  # Worse
        assert result == True
        assert es.counter == 2
    
    def test_early_stop_triggered(self):
        """Test early stopping is triggered after patience epochs."""
        es = EarlyStopping(patience=3)
        
        es(1.0)  # Set baseline
        es(1.0)  # No improvement, counter = 1
        es(1.0)  # No improvement, counter = 2
        result = es(1.0)  # No improvement, counter = 3
        
        assert result == True
        assert es.counter == 3
        assert es.early_stop == True
    
    def test_delta_threshold(self):
        """Test delta threshold for improvement detection."""
        es = EarlyStopping(patience=3, delta=0.1)
        
        es(1.0)  # Set baseline
        
        # Improvement less than delta doesn't count
        result = es(0.95)  # 1.0 - 0.95 = 0.05 < delta
        assert result == True
        assert es.counter == 1
        
        # Improvement greater than delta counts
        es = EarlyStopping(patience=3, delta=0.1)
        es(1.0)
        result = es(0.85)  # 1.0 - 0.85 = 0.15 > delta
        assert result == False
        assert es.counter == 0


# ==========================================
# set_seed Tests
# ==========================================

class TestSetSeed:
    """Tests for set_seed function."""
    
    def test_reproducibility(self):
        """Test that set_seed ensures reproducibility."""
        set_seed(42)
        r1 = torch.randn(5)
        
        set_seed(42)
        r2 = torch.randn(5)
        
        assert torch.allclose(r1, r2)
    
    def test_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        r1 = torch.randn(5)
        
        set_seed(123)
        r2 = torch.randn(5)
        
        assert not torch.allclose(r1, r2)


# ==========================================
# prepare_labels Tests
# ==========================================

class TestPrepareLabels:
    """Tests for prepare_labels function."""
    
    def test_output_keys(self, mock_collator_output, device):
        """Test that prepare_labels returns the expected keys."""
        result = prepare_labels(mock_collator_output, device)
        
        assert "mag_features" in result
        assert "mod_features" in result
        assert "src_key_padding_mask" in result
        assert "labels" in result
        
        labels = result["labels"]
        assert "mag_targets" in labels
        assert "sign_targets" in labels
        assert "mod_targets" in labels
        assert "mask_map" in labels
    
    def test_mag_targets_extraction(self, mock_collator_output, device, batch_size, seq_len):
        """Test that mag_targets is correctly extracted from mag_labels."""
        result = prepare_labels(mock_collator_output, device)
        
        mag_targets = result["labels"]["mag_targets"]
        expected = mock_collator_output["mag_labels"][:, :, 0]
        
        assert mag_targets.shape == (batch_size, seq_len)
        assert torch.allclose(mag_targets, expected.to(device))
    
    def test_sign_targets_conversion(self, mock_collator_output, device, batch_size, seq_len):
        """Test that sign one-hot is correctly converted to class indices."""
        result = prepare_labels(mock_collator_output, device)
        
        sign_targets = result["labels"]["sign_targets"]
        
        assert sign_targets.shape == (batch_size, seq_len)
        assert sign_targets.dtype == torch.int64
        
        # Check values are 0, 1, or 2
        assert (sign_targets >= 0).all()
        assert (sign_targets <= 2).all()
    
    def test_padding_mask_conversion(self, mock_collator_output, device, batch_size, seq_len):
        """Test attention_mask is correctly inverted to src_key_padding_mask."""
        result = prepare_labels(mock_collator_output, device)
        
        padding_mask = result["src_key_padding_mask"]
        attention_mask = mock_collator_output["attention_mask"]
        
        # Should be True where attention_mask is 0
        expected = (attention_mask == 0).to(device)
        assert torch.equal(padding_mask, expected)
    
    def test_device_transfer(self, mock_collator_output):
        """Test that tensors are moved to the specified device."""
        device = torch.device("cpu")
        result = prepare_labels(mock_collator_output, device)
        
        assert result["mag_features"].device == device
        assert result["mod_features"].device == device
        assert result["labels"]["mag_targets"].device == device


# ==========================================
# evaluate Tests
# ==========================================

class TestEvaluate:
    """Tests for evaluate function."""
    
    def test_output_keys(self, small_model, mock_collator_output, device):
        """Test that evaluate returns expected metric keys."""
        from torch.utils.data import DataLoader
        
        # Create a simple dataset from mock batch
        class MockDataset:
            def __init__(self, batch):
                self.batch = batch
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return {k: v[0] if isinstance(v, list) else v for k, v in self.batch.items()}
        
        def mock_collate(samples):
            return mock_collator_output
        
        loader = DataLoader(MockDataset(mock_collator_output), batch_size=4, collate_fn=mock_collate)
        
        metrics = evaluate(small_model, loader, device)
        
        assert "val_loss" in metrics
        assert "mag_mse" in metrics
        assert "mag_acc" in metrics
        assert "sign_acc" in metrics
        assert "mod_acc" in metrics
        assert "mod_loss" in metrics
    
    def test_accuracy_ranges(self, small_model, mock_collator_output, device):
        """Test that accuracy metrics are in valid percentage range."""
        from torch.utils.data import DataLoader
        
        def mock_collate(samples):
            return mock_collator_output
        
        class MockDataset:
            def __len__(self): return 1
            def __getitem__(self, idx): return {}
        
        loader = DataLoader(MockDataset(), batch_size=4, collate_fn=mock_collate)
        
        metrics = evaluate(small_model, loader, device)
        
        # Accuracies should be 0-100%
        assert 0 <= metrics["mag_acc"] <= 100
        assert 0 <= metrics["sign_acc"] <= 100
        assert 0 <= metrics["mod_acc"] <= 100
        
        # MSE should be non-negative
        assert metrics["mag_mse"] >= 0
    
    def test_eval_mode(self, small_model, mock_collator_output, device):
        """Test that model is in eval mode during evaluation."""
        from torch.utils.data import DataLoader
        
        def mock_collate(samples):
            return mock_collator_output
        
        class MockDataset:
            def __len__(self): return 1
            def __getitem__(self, idx): return {}
        
        loader = DataLoader(MockDataset(), batch_size=4, collate_fn=mock_collate)
        
        small_model.train()  # Start in train mode
        
        # Evaluate should switch to eval
        evaluate(small_model, loader, device)
        
        # Model should still be in eval mode after
        assert not small_model.training


# ==========================================
# Integration Tests
# ==========================================

class TestTrainingIntegration:
    """Integration tests for training components."""
    
    def test_full_forward_backward(self, small_model, mock_collator_output, device):
        """Test complete forward-backward pass with prepare_labels."""
        inputs = prepare_labels(mock_collator_output, device)
        
        # Forward pass
        outputs = small_model(
            mag_features=inputs["mag_features"],
            mod_features=inputs["mod_features"],
            src_key_padding_mask=inputs["src_key_padding_mask"],
            labels=inputs["labels"]
        )
        
        assert "loss" in outputs
        assert outputs["loss"].requires_grad
        
        # Backward pass
        outputs["loss"].backward()
        
        # Check gradients exist
        for name, param in small_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_optimizer_step(self, small_model, mock_collator_output, device):
        """Test optimizer can update parameters."""
        optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-3)
        
        # Get initial weights
        initial_weight = small_model.bert.embeddings.mag_proj.weight.clone()
        
        inputs = prepare_labels(mock_collator_output, device)
        outputs = small_model(
            mag_features=inputs["mag_features"],
            mod_features=inputs["mod_features"],
            src_key_padding_mask=inputs["src_key_padding_mask"],
            labels=inputs["labels"]
        )
        
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        
        # Weights should have changed
        updated_weight = small_model.bert.embeddings.mag_proj.weight
        assert not torch.allclose(initial_weight, updated_weight)
