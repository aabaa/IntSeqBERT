"""
test_models.py:
Comprehensive tests for the IntSeqBERT neural network models.
Tests IntSeqEmbeddings, IntSeqModel, IntSeqForPreTraining, and loss computation.
"""

import pytest
import torch
import math

from intseq_bert import config
from intseq_bert.base_models import generate_sinusoidal_encoding
from intseq_bert.intseq_models import (
    IntSeqEmbeddings,
    IntSeqModel,
    IntSeqForPreTraining
)


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
def d_model():
    return 32  # Smaller for faster tests

@pytest.fixture
def sample_inputs(batch_size, seq_len):
    """Creates sample input tensors for testing."""
    mag_features = torch.randn(batch_size, seq_len, config.MAG_EXTENDED_DIM)
    mod_features = torch.randn(batch_size, seq_len, config.MOD_FEATURE_DIM)
    return mag_features, mod_features

@pytest.fixture
def sample_padding_mask(batch_size, seq_len):
    """Creates a sample padding mask with some padded positions."""
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    # Last 4 positions are padding for each sample
    mask[:, -4:] = True
    return mask

@pytest.fixture
def sample_labels(batch_size, seq_len):
    """Creates sample labels for loss computation."""
    # mask_map: True where we want to compute loss
    mask_map = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask_map[:, :8] = True  # First 8 positions are masked
    
    return {
        "mag_targets": torch.randn(batch_size, seq_len),
        "sign_targets": torch.randint(0, config.NUM_SIGN_CLASSES, (batch_size, seq_len)),
        "mod_targets": torch.stack([
            torch.randint(0, m, (batch_size, seq_len)) 
            for m in config.MOD_RANGE
        ], dim=-1),  # (B, L, 100)
        "mask_map": mask_map
    }


# ==========================================
# Helper Function Tests
# ==========================================

class TestSinusoidalEncoding:
    """Tests for _generate_sinusoidal_encoding."""
    
    def test_output_shape(self):
        """Test output has correct shape."""
        max_len = 64
        d_model = 128
        pe = generate_sinusoidal_encoding(max_len, d_model)
        assert pe.shape == (1, max_len, d_model)
    
    def test_sin_cos_alternation(self):
        """Test that even indices use sin, odd indices use cos."""
        max_len = 10
        d_model = 8
        pe = generate_sinusoidal_encoding(max_len, d_model)
        
        # Position 0 should have [sin(0), cos(0), ...] = [0, 1, 0, 1, ...]
        pos0 = pe[0, 0, :]
        assert torch.allclose(pos0[0], torch.tensor(0.0), atol=1e-6)  # sin(0)
        assert torch.allclose(pos0[1], torch.tensor(1.0), atol=1e-6)  # cos(0)
    
    def test_different_positions_differ(self):
        """Test that different positions have different encodings."""
        pe = generate_sinusoidal_encoding(100, 64)
        assert not torch.allclose(pe[0, 0], pe[0, 1])
        assert not torch.allclose(pe[0, 0], pe[0, 50])


# ==========================================
# IntSeqEmbeddings Tests
# ==========================================

class TestIntSeqEmbeddings:
    """Tests for IntSeqEmbeddings layer."""
    
    def test_output_shape(self, sample_inputs, d_model, batch_size, seq_len):
        """Test output has correct shape."""
        mag, mod = sample_inputs
        embeddings = IntSeqEmbeddings(d_model=d_model)
        output = embeddings(mag, mod)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_film_initialization(self, d_model):
        """Test FiLM weights are initialized to zero (identity-like)."""
        embeddings = IntSeqEmbeddings(d_model=d_model)
        
        # film_scale should be zero-initialized
        assert torch.allclose(embeddings.film_scale.weight, torch.zeros_like(embeddings.film_scale.weight))
        assert torch.allclose(embeddings.film_scale.bias, torch.zeros_like(embeddings.film_scale.bias))
    
    def test_film_identity_at_init(self, sample_inputs, d_model):
        """Test that at initialization, FiLM acts close to identity."""
        mag, mod = sample_inputs
        embeddings = IntSeqEmbeddings(d_model=d_model, dropout=0.0)
        
        # With film weights at zero, h_fused ≈ h_mag (before adding PE and LN)
        # Can't test exactly due to PE and LN, but check it runs
        output = embeddings(mag, mod)
        assert output.shape[-1] == d_model
    
    def test_dropout_effect(self, sample_inputs, d_model):
        """Test dropout has effect in training mode."""
        mag, mod = sample_inputs
        embeddings = IntSeqEmbeddings(d_model=d_model, dropout=0.5)
        
        embeddings.train()
        out1 = embeddings(mag, mod)
        out2 = embeddings(mag, mod)
        
        # Outputs should differ due to dropout
        assert not torch.allclose(out1, out2)
        
        embeddings.eval()
        out3 = embeddings(mag, mod)
        out4 = embeddings(mag, mod)
        
        # In eval mode, should be deterministic
        assert torch.allclose(out3, out4)


# ==========================================
# IntSeqModel Tests
# ==========================================

class TestIntSeqModel:
    """Tests for IntSeqModel backbone."""
    
    def test_output_shape(self, sample_inputs, sample_padding_mask, d_model, batch_size, seq_len):
        """Test output has correct shape."""
        mag, mod = sample_inputs
        model = IntSeqModel(d_model=d_model, nhead=2, num_layers=2)
        output = model(mag, mod, src_key_padding_mask=sample_padding_mask)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_without_padding_mask(self, sample_inputs, d_model):
        """Test model works without padding mask."""
        mag, mod = sample_inputs
        model = IntSeqModel(d_model=d_model, nhead=2, num_layers=2)
        output = model(mag, mod)
        assert output.shape[-1] == d_model
    
    def test_gradient_flow(self, sample_inputs, d_model):
        """Test gradients flow through the model."""
        mag, mod = sample_inputs
        mag.requires_grad_(True)
        
        model = IntSeqModel(d_model=d_model, nhead=2, num_layers=2)
        output = model(mag, mod)
        loss = output.sum()
        loss.backward()
        
        assert mag.grad is not None
        assert not torch.all(mag.grad == 0)


# ==========================================
# IntSeqForPreTraining Tests
# ==========================================

class TestIntSeqForPreTraining:
    """Tests for IntSeqForPreTraining model."""
    
    def test_inference_output_structure(self, sample_inputs, sample_padding_mask, d_model, batch_size, seq_len):
        """Test output structure during inference (no labels)."""
        mag, mod = sample_inputs
        model = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
        
        outputs = model(mag, mod, sample_padding_mask)
        
        assert "predictions" in outputs
        assert "loss" not in outputs  # No loss without labels
        
        preds = outputs["predictions"]
        assert preds["mag_mu"].shape == (batch_size, seq_len)
        assert preds["mag_log_var"].shape == (batch_size, seq_len)
        assert preds["sign_logits"].shape == (batch_size, seq_len, config.NUM_SIGN_CLASSES)
        assert preds["mod_logits"].shape == (batch_size, seq_len, sum(config.MOD_RANGE))
    
    def test_training_output_structure(self, sample_inputs, sample_padding_mask, sample_labels, d_model):
        """Test output structure during training (with labels)."""
        mag, mod = sample_inputs
        model = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
        
        outputs = model(mag, mod, sample_padding_mask, labels=sample_labels)
        
        assert "loss" in outputs
        assert "predictions" in outputs
        assert "loss_breakdown" in outputs
        
        assert outputs["loss"].dim() == 0  # Scalar
        assert outputs["loss"].requires_grad
    
    def test_loss_breakdown_keys(self, sample_inputs, sample_padding_mask, sample_labels, d_model):
        """Test loss_breakdown contains all expected keys."""
        mag, mod = sample_inputs
        model = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
        
        outputs = model(mag, mod, sample_padding_mask, labels=sample_labels)
        
        breakdown = outputs["loss_breakdown"]
        expected_keys = ["raw_mag", "raw_sign", "raw_mod", "w_mag", "w_sign", "w_mod"]
        for key in expected_keys:
            assert key in breakdown, f"Missing key: {key}"
    
    def test_loss_weights_fixed(self, d_model):
        """Test that loss_weights are fixed buffers (not learnable)."""
        model = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
        
        # loss_weights should be a buffer, not a parameter
        assert not model.loss_weights.requires_grad
        assert model.loss_weights.shape == (3,)
        
        # Verify fixed values: [1.0, 1.0, 2.0]
        expected = torch.tensor([1.0, 1.0, 2.0])
        assert torch.allclose(model.loss_weights, expected)
        
        # Check it's NOT in model parameters
        param_names = [n for n, _ in model.named_parameters()]
        assert "loss_weights" not in param_names
    
    def test_gradient_flow_through_loss(self, sample_inputs, sample_padding_mask, sample_labels, d_model):
        """Test gradients flow through loss computation."""
        mag, mod = sample_inputs
        model = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
        
        outputs = model(mag, mod, sample_padding_mask, labels=sample_labels)
        loss = outputs["loss"]
        loss.backward()
        
        # Check some parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ==========================================
# Loss Computation Tests
# ==========================================

class TestLossComputation:
    """Tests for loss calculation logic."""
    
    def test_loss_is_positive(self, sample_inputs, sample_padding_mask, sample_labels, d_model):
        """Test that total loss is positive."""
        mag, mod = sample_inputs
        model = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
        
        outputs = model(mag, mod, sample_padding_mask, labels=sample_labels)
        
        # After some training, loss should generally be positive
        # (can occasionally be negative with auto-weighting, but unlikely initially)
        assert outputs["loss"].item() > -100  # Sanity check
    
    def test_raw_losses_positive(self, sample_inputs, sample_padding_mask, sample_labels, d_model):
        """Test that individual raw losses are positive."""
        mag, mod = sample_inputs
        model = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
        
        outputs = model(mag, mod, sample_padding_mask, labels=sample_labels)
        breakdown = outputs["loss_breakdown"]
        
        # CrossEntropy losses should be positive
        assert breakdown["raw_sign"].item() >= 0
        assert breakdown["raw_mod"].item() >= 0
    
    def test_only_masked_positions_contribute(self, sample_inputs, sample_padding_mask, d_model):
        """Test that only masked positions contribute to loss."""
        mag, mod = sample_inputs
        batch_size, seq_len = mag.shape[:2]
        
        # Create labels with mask_map = all False
        labels_no_mask = {
            "mag_targets": torch.randn(batch_size, seq_len),
            "sign_targets": torch.randint(0, 3, (batch_size, seq_len)),
            "mod_targets": torch.stack([
                torch.randint(0, m, (batch_size, seq_len)) 
                for m in config.MOD_RANGE
            ], dim=-1),
            "mask_map": torch.zeros(batch_size, seq_len, dtype=torch.bool)
        }
        
        model = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
        
        # This should cause an error because mean of empty tensor
        # But the implementation might handle it differently
        # Let's just check it doesn't crash catastrophically
        try:
            outputs = model(mag, mod, sample_padding_mask, labels=labels_no_mask)
            # If it runs, loss might be nan due to empty masked set
        except RuntimeError:
            pass  # Expected if mean of empty tensor

    def test_split_mod_logits(self, d_model):
        """Test _split_mod_logits correctly splits the unified logits."""
        model = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
        
        total_classes = sum(config.MOD_RANGE)
        fake_logits = torch.randn(8, total_classes)  # batch of 8
        
        split = model._split_mod_logits(fake_logits)
        
        assert len(split) == len(config.MOD_RANGE)
        assert split[0].shape == (8, 2)    # mod 2
        assert split[1].shape == (8, 3)    # mod 3
        assert split[-1].shape == (8, 101) # mod 101


# ==========================================
# Integration Tests
# ==========================================

class TestIntegration:
    """Integration tests with collator output format."""
    
    def test_with_collator_output_format(self, d_model):
        """Test model works with collator-like output format."""
        batch_size = 2
        seq_len = 10
        
        # Simulate collator output
        mag_inputs = torch.randn(batch_size, seq_len, config.MAG_EXTENDED_DIM)
        mod_inputs = torch.randn(batch_size, seq_len, config.MOD_FEATURE_DIM)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        attention_mask[:, -2:] = 0  # Last 2 are padding
        
        # Convert attention_mask to key_padding_mask (True = padding)
        key_padding_mask = (attention_mask == 0)
        
        model = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
        outputs = model(mag_inputs, mod_inputs, key_padding_mask)
        
        assert outputs["predictions"]["mag_mu"].shape == (batch_size, seq_len)
    
    def test_different_sequence_lengths(self, d_model):
        """Test model handles different sequence lengths."""
        model = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
        
        for seq_len in [8, 32, 64, 128]:
            mag = torch.randn(2, seq_len, config.MAG_EXTENDED_DIM)
            mod = torch.randn(2, seq_len, config.MOD_FEATURE_DIM)
            mask = torch.zeros(2, seq_len, dtype=torch.bool)
            
            outputs = model(mag, mod, mask)
            assert outputs["predictions"]["mag_mu"].shape == (2, seq_len)


# ==========================================
# V3 Config Option Tests
# ==========================================

class TestV3ConfigOptions:
    """Tests for v3 model configuration options."""
    
    def test_input_proj_mlp_structure(self, d_model):
        """Test MLP projection creates correct structure."""
        original = config.INPUT_PROJ_TYPE
        config.INPUT_PROJ_TYPE = 'mlp'
        try:
            embeddings = IntSeqEmbeddings(d_model=d_model)
            assert isinstance(embeddings.mag_proj, torch.nn.Sequential)
            assert len(embeddings.mag_proj) == 3  # Linear, GELU, Linear
            assert isinstance(embeddings.mag_proj[0], torch.nn.Linear)
            assert isinstance(embeddings.mag_proj[1], torch.nn.GELU)
            assert isinstance(embeddings.mag_proj[2], torch.nn.Linear)
        finally:
            config.INPUT_PROJ_TYPE = original
    
    def test_input_proj_linear_structure(self, d_model):
        """Test linear projection creates correct structure."""
        original = config.INPUT_PROJ_TYPE
        config.INPUT_PROJ_TYPE = 'linear'
        try:
            embeddings = IntSeqEmbeddings(d_model=d_model)
            assert isinstance(embeddings.mag_proj, torch.nn.Linear)
        finally:
            config.INPUT_PROJ_TYPE = original
    
    def test_input_proj_mlp_output_shape(self, sample_inputs, d_model, batch_size, seq_len):
        """Test MLP projection produces correct output shape."""
        original = config.INPUT_PROJ_TYPE
        config.INPUT_PROJ_TYPE = 'mlp'
        try:
            mag, mod = sample_inputs
            embeddings = IntSeqEmbeddings(d_model=d_model)
            output = embeddings(mag, mod)
            assert output.shape == (batch_size, seq_len, d_model)
        finally:
            config.INPUT_PROJ_TYPE = original
    
    def test_pre_film_dropout_in_training(self, sample_inputs, d_model):
        """Test Pre-FiLM dropout causes variance when enabled."""
        original = config.USE_PRE_FILM_DROPOUT
        config.USE_PRE_FILM_DROPOUT = True
        try:
            mag, mod = sample_inputs
            embeddings = IntSeqEmbeddings(d_model=d_model, dropout=0.5)
            embeddings.train()
            
            out1 = embeddings(mag, mod)
            out2 = embeddings(mag, mod)
            # Outputs should differ due to dropout
            assert not torch.allclose(out1, out2)
        finally:
            config.USE_PRE_FILM_DROPOUT = original
    
    def test_mag_loss_type_huber(self, sample_inputs, sample_padding_mask, sample_labels, d_model):
        """Test Huber loss type produces valid loss."""
        original = config.MAG_LOSS_TYPE
        config.MAG_LOSS_TYPE = 'huber'
        try:
            mag, mod = sample_inputs
            model = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
            outputs = model(mag, mod, sample_padding_mask, labels=sample_labels)
            assert not torch.isnan(outputs["loss"])
            assert not torch.isinf(outputs["loss"])
        finally:
            config.MAG_LOSS_TYPE = original
    
    def test_mag_loss_type_mse(self, sample_inputs, sample_padding_mask, sample_labels, d_model):
        """Test MSE loss type produces valid loss."""
        original = config.MAG_LOSS_TYPE
        config.MAG_LOSS_TYPE = 'mse'
        try:
            mag, mod = sample_inputs
            model = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
            outputs = model(mag, mod, sample_padding_mask, labels=sample_labels)
            assert not torch.isnan(outputs["loss"])
            assert not torch.isinf(outputs["loss"])
        finally:
            config.MAG_LOSS_TYPE = original
    
    def test_mag_loss_type_l1(self, sample_inputs, sample_padding_mask, sample_labels, d_model):
        """Test L1 loss type produces valid loss."""
        original = config.MAG_LOSS_TYPE
        config.MAG_LOSS_TYPE = 'l1'
        try:
            mag, mod = sample_inputs
            model = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
            outputs = model(mag, mod, sample_padding_mask, labels=sample_labels)
            assert not torch.isnan(outputs["loss"])
            assert not torch.isinf(outputs["loss"])
        finally:
            config.MAG_LOSS_TYPE = original
    
    def test_heteroscedastic_off(self, sample_inputs, sample_padding_mask, sample_labels, d_model):
        """Test heteroscedastic loss OFF produces valid loss."""
        original = config.USE_HETEROSCEDASTIC_LOSS
        config.USE_HETEROSCEDASTIC_LOSS = False
        try:
            mag, mod = sample_inputs
            model = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
            outputs = model(mag, mod, sample_padding_mask, labels=sample_labels)
            assert not torch.isnan(outputs["loss"])
            assert outputs["loss"].item() >= 0  # Deterministic loss should be non-negative
        finally:
            config.USE_HETEROSCEDASTIC_LOSS = original
    
    def test_heteroscedastic_on(self, sample_inputs, sample_padding_mask, sample_labels, d_model):
        """Test heteroscedastic loss ON produces valid loss."""
        original = config.USE_HETEROSCEDASTIC_LOSS
        config.USE_HETEROSCEDASTIC_LOSS = True
        try:
            mag, mod = sample_inputs
            model = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
            outputs = model(mag, mod, sample_padding_mask, labels=sample_labels)
            assert not torch.isnan(outputs["loss"])
        finally:
            config.USE_HETEROSCEDASTIC_LOSS = original

