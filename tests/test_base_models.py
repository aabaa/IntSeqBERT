"""
test_base_models.py:
Unit tests for shared base components in base_models.py.
Tests ModLogitsMixin, PositionalEncoding, and base class functionality.
"""

import pytest
import torch
import torch.nn as nn
import math
import tempfile
import os

from intseq_bert import config
from intseq_bert.base_models import (
    ModLogitsMixin,
    generate_sinusoidal_encoding,
    PositionalEncoding,
    BasePreTrainedModel,
    BaseEmbeddings,
    BaseTransformerModel,
    BaseForPreTraining,
)


# ============================================================
# Test Fixtures
# ============================================================


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_len():
    return 16


@pytest.fixture
def d_model():
    return 32  # Smaller for faster tests


# ============================================================
# ModLogitsMixin Tests
# ============================================================


class TestModLogitsMixin:
    """Tests for ModLogitsMixin helper."""
    
    def test_split_mod_logits_correct_count(self):
        """Test correct number of splits."""
        class MixinUser(ModLogitsMixin):
            pass
        
        user = MixinUser()
        total_classes = sum(config.MOD_RANGE)
        logits = torch.randn(8, total_classes)
        
        split = user._split_mod_logits(logits)
        
        assert len(split) == len(config.MOD_RANGE)
    
    def test_split_mod_logits_correct_sizes(self):
        """Test each split has correct size."""
        class MixinUser(ModLogitsMixin):
            pass
        
        user = MixinUser()
        total_classes = sum(config.MOD_RANGE)
        logits = torch.randn(8, total_classes)
        
        split = user._split_mod_logits(logits)
        
        for i, s in enumerate(split):
            expected_size = config.MOD_RANGE[i]
            assert s.shape == (8, expected_size), f"Split {i} has wrong shape"
    
    def test_split_mod_logits_3d_input(self):
        """Test with 3D input (B, L, C)."""
        class MixinUser(ModLogitsMixin):
            pass
        
        user = MixinUser()
        total_classes = sum(config.MOD_RANGE)
        logits = torch.randn(4, 10, total_classes)
        
        split = user._split_mod_logits(logits)
        
        assert split[0].shape == (4, 10, 2)   # mod 2
        assert split[-1].shape == (4, 10, 101)  # mod 101


# ============================================================
# generate_sinusoidal_encoding Tests
# ============================================================


class TestGenerateSinusoidalEncoding:
    """Tests for generate_sinusoidal_encoding function."""
    
    def test_output_shape(self):
        """Test output has correct shape (1, max_len, d_model)."""
        max_len = 64
        d_model = 128
        pe = generate_sinusoidal_encoding(max_len, d_model)
        
        assert pe.shape == (1, max_len, d_model)
    
    def test_sin_cos_alternation(self):
        """Test even indices use sin, odd indices use cos."""
        pe = generate_sinusoidal_encoding(10, 8)
        
        # Position 0 should have [sin(0), cos(0), ...] = [0, 1, ...]
        pos0 = pe[0, 0, :]
        assert torch.allclose(pos0[0], torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(pos0[1], torch.tensor(1.0), atol=1e-6)
    
    def test_different_positions_differ(self):
        """Test different positions have different encodings."""
        pe = generate_sinusoidal_encoding(100, 64)
        
        assert not torch.allclose(pe[0, 0], pe[0, 1])
        assert not torch.allclose(pe[0, 0], pe[0, 50])
    
    def test_deterministic(self):
        """Test encoding is deterministic."""
        pe1 = generate_sinusoidal_encoding(50, 32)
        pe2 = generate_sinusoidal_encoding(50, 32)
        
        assert torch.allclose(pe1, pe2)


# ============================================================
# PositionalEncoding Module Tests
# ============================================================


class TestPositionalEncoding:
    """Tests for PositionalEncoding module."""
    
    def test_output_shape(self, batch_size, seq_len, d_model):
        """Test output has correct shape."""
        pe = PositionalEncoding(d_model, dropout=0.0, max_len=100)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pe(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_adds_positional_encoding(self, batch_size, seq_len, d_model):
        """Test positional encoding is added to input."""
        pe = PositionalEncoding(d_model, dropout=0.0, max_len=100)
        x = torch.zeros(batch_size, seq_len, d_model)
        
        output = pe(x)
        
        # Output should not be all zeros
        assert not torch.allclose(output, torch.zeros_like(output))
    
    def test_dropout_effect(self, batch_size, seq_len, d_model):
        """Test dropout has effect in training mode."""
        pe = PositionalEncoding(d_model, dropout=0.5, max_len=100)
        x = torch.randn(batch_size, seq_len, d_model)
        
        pe.train()
        out1 = pe(x)
        out2 = pe(x)
        
        # Outputs should differ due to dropout
        assert not torch.allclose(out1, out2)
        
        pe.eval()
        out3 = pe(x)
        out4 = pe(x)
        
        # Eval mode should be deterministic
        assert torch.allclose(out3, out4)
    
    def test_pe_is_buffer(self, d_model):
        """Test positional encoding is registered as buffer."""
        pe = PositionalEncoding(d_model, dropout=0.1, max_len=100)
        
        assert 'pe' in dict(pe.named_buffers())
        assert pe.pe.shape == (1, 100, d_model)


# ============================================================
# BasePreTrainedModel Tests
# ============================================================


class TestBasePreTrainedModel:
    """Tests for BasePreTrainedModel base class."""
    
    def test_init_weights_linear(self, d_model):
        """Test _init_weights correctly initializes Linear layers."""
        model = BasePreTrainedModel()
        linear = nn.Linear(d_model, d_model)
        
        model._init_weights(linear)
        
        # Check mean is close to 0, std is close to 0.02
        assert abs(linear.weight.mean().item()) < 0.1
        assert abs(linear.weight.std().item() - 0.02) < 0.01
        # Bias should be zero
        assert torch.allclose(linear.bias, torch.zeros_like(linear.bias))
    
    def test_init_weights_embedding(self, d_model):
        """Test _init_weights correctly initializes Embedding layers."""
        model = BasePreTrainedModel()
        embedding = nn.Embedding(100, d_model, padding_idx=0)
        
        model._init_weights(embedding)
        
        # Padding index should be zero
        assert torch.allclose(embedding.weight[0], torch.zeros(d_model))
        # Other weights should be initialized
        assert not torch.allclose(embedding.weight[1], torch.zeros(d_model))
    
    def test_init_weights_layer_norm(self, d_model):
        """Test _init_weights correctly initializes LayerNorm."""
        model = BasePreTrainedModel()
        ln = nn.LayerNorm(d_model)
        
        model._init_weights(ln)
        
        assert torch.allclose(ln.weight, torch.ones(d_model))
        assert torch.allclose(ln.bias, torch.zeros(d_model))
    
    def test_from_checkpoint(self, d_model):
        """Test from_checkpoint loads saved model."""
        # Create and save a simple model
        class SimpleModel(BasePreTrainedModel):
            def __init__(self, d_model=32):
                super().__init__()
                self.linear = nn.Linear(d_model, d_model)
                self.d_model = d_model
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel(d_model)
        # Set specific weights
        model.linear.weight.data.fill_(1.5)
        model.linear.bias.data.fill_(0.5)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"d_model": d_model}
            }, path)
            
            # Load model
            loaded = SimpleModel.from_checkpoint(path)
            
            assert torch.allclose(loaded.linear.weight, torch.full_like(loaded.linear.weight, 1.5))
            assert torch.allclose(loaded.linear.bias, torch.full_like(loaded.linear.bias, 0.5))
            assert not loaded.training  # Should be in eval mode


# ============================================================
# BaseEmbeddings Tests
# ============================================================


class TestBaseEmbeddings:
    """Tests for BaseEmbeddings base class."""
    
    def test_has_layer_norm(self, d_model):
        """Test BaseEmbeddings has layer_norm component."""
        embeddings = BaseEmbeddings(d_model, dropout=0.1, max_len=100)
        
        assert hasattr(embeddings, 'layer_norm')
        assert isinstance(embeddings.layer_norm, nn.LayerNorm)
        assert embeddings.layer_norm.normalized_shape == (d_model,)
    
    def test_has_dropout(self, d_model):
        """Test BaseEmbeddings has dropout component."""
        embeddings = BaseEmbeddings(d_model, dropout=0.2, max_len=100)
        
        assert hasattr(embeddings, 'dropout')
        assert isinstance(embeddings.dropout, nn.Dropout)
        assert embeddings.dropout.p == 0.2
    
    def test_stores_d_model(self, d_model):
        """Test d_model is stored."""
        embeddings = BaseEmbeddings(d_model, dropout=0.1, max_len=100)
        
        assert embeddings.d_model == d_model


# ============================================================
# BaseTransformerModel Tests
# ============================================================


class TestBaseTransformerModel:
    """Tests for BaseTransformerModel base class."""
    
    def test_has_encoder(self, d_model):
        """Test encoder is created."""
        model = BaseTransformerModel(d_model=d_model, nhead=2, num_layers=2, dropout=0.1)
        
        assert hasattr(model, 'encoder')
        assert isinstance(model.encoder, nn.TransformerEncoder)
    
    def test_encoder_num_layers(self, d_model):
        """Test correct number of encoder layers."""
        model = BaseTransformerModel(d_model=d_model, nhead=2, num_layers=3, dropout=0.1)
        
        assert model.encoder.num_layers == 3
    
    def test_stores_config(self, d_model):
        """Test config values are stored."""
        model = BaseTransformerModel(d_model=d_model, nhead=4, num_layers=2, dropout=0.15)
        
        assert model.d_model == d_model
        assert model.nhead == 4
        assert model.num_layers == 2
    
    def test_encoder_batch_first(self, d_model):
        """Test encoder uses batch_first=True."""
        model = BaseTransformerModel(d_model=d_model, nhead=2, num_layers=2,dropout=0.1)
        
        # TransformerEncoderLayer should have batch_first=True
        layer = model.encoder.layers[0]
        assert layer.self_attn.batch_first == True


# ============================================================
# BaseForPreTraining Tests
# ============================================================


class TestBaseForPreTraining:
    """Tests for BaseForPreTraining base class."""
    
    def test_has_mag_head(self, d_model):
        """Test mag_head is created with correct structure."""
        model = BaseForPreTraining(d_model=d_model)
        
        assert hasattr(model, 'mag_head')
        assert isinstance(model.mag_head, nn.Sequential)
        assert len(model.mag_head) == 3  # Linear, ReLU, Linear
        assert model.mag_head[-1].out_features == 2  # mu, log_var
    
    def test_has_sign_head(self, d_model):
        """Test sign_head is created."""
        model = BaseForPreTraining(d_model=d_model)
        
        assert hasattr(model, 'sign_head')
        assert isinstance(model.sign_head, nn.Linear)
        assert model.sign_head.out_features == config.NUM_SIGN_CLASSES
    
    def test_has_mod_head(self, d_model):
        """Test mod_head is created."""
        model = BaseForPreTraining(d_model=d_model)
        
        assert hasattr(model, 'mod_head')
        assert isinstance(model.mod_head, nn.Linear)
        assert model.mod_head.out_features == sum(config.MOD_RANGE)
    
    def test_loss_weights_is_buffer(self, d_model):
        """Test loss_weights is registered as buffer."""
        model = BaseForPreTraining(d_model=d_model)
        
        assert 'loss_weights' in dict(model.named_buffers())
        assert not model.loss_weights.requires_grad
    
    def test_loss_weights_values(self, d_model):
        """Test loss_weights has correct values."""
        model = BaseForPreTraining(d_model=d_model)
        
        expected = torch.tensor([
            config.LOSS_WEIGHT_MAG,
            config.LOSS_WEIGHT_SIGN,
            config.LOSS_WEIGHT_MOD
        ])
        assert torch.allclose(model.loss_weights, expected)
    
    def test_compute_mag_loss_huber(self, d_model):
        """Test _compute_mag_loss with huber loss."""
        original = config.MAG_LOSS_TYPE
        config.MAG_LOSS_TYPE = 'huber'
        try:
            model = BaseForPreTraining(d_model=d_model)
            
            pred_mu = torch.randn(10)
            pred_log_var = torch.randn(10)
            target = torch.randn(10)
            
            loss = model._compute_mag_loss(pred_mu, pred_log_var, target)
            
            assert loss.dim() == 0  # Scalar
            assert not torch.isnan(loss)
        finally:
            config.MAG_LOSS_TYPE = original
    
    def test_compute_mag_loss_mse(self, d_model):
        """Test _compute_mag_loss with mse loss."""
        original = config.MAG_LOSS_TYPE
        config.MAG_LOSS_TYPE = 'mse'
        try:
            model = BaseForPreTraining(d_model=d_model)
            
            pred_mu = torch.randn(10)
            pred_log_var = torch.randn(10)
            target = torch.randn(10)
            
            loss = model._compute_mag_loss(pred_mu, pred_log_var, target)
            
            assert loss.dim() == 0
            assert not torch.isnan(loss)
        finally:
            config.MAG_LOSS_TYPE = original
    
    def test_compute_mag_loss_l1(self, d_model):
        """Test _compute_mag_loss with l1 loss."""
        original = config.MAG_LOSS_TYPE
        config.MAG_LOSS_TYPE = 'l1'
        try:
            model = BaseForPreTraining(d_model=d_model)
            
            pred_mu = torch.randn(10)
            pred_log_var = torch.randn(10)
            target = torch.randn(10)
            
            loss = model._compute_mag_loss(pred_mu, pred_log_var, target)
            
            assert loss.dim() == 0
            assert not torch.isnan(loss)
        finally:
            config.MAG_LOSS_TYPE = original
    
    def test_compute_mod_loss(self, d_model):
        """Test _compute_mod_loss returns valid loss."""
        model = BaseForPreTraining(d_model=d_model)
        
        total_classes = sum(config.MOD_RANGE)
        pred_logits = torch.randn(10, total_classes)
        target_mods = torch.stack([
            torch.randint(0, m, (10,))
            for m in config.MOD_RANGE
        ], dim=-1)
        
        loss = model._compute_mod_loss(pred_logits, target_mods)
        
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert loss.item() >= 0  # CrossEntropy >= 0
    
    def test_compute_mod_loss_normalized(self, d_model):
        """Test _compute_mod_loss normalizes by log(m)."""
        model = BaseForPreTraining(d_model=d_model)
        
        # With random predictions, normalized loss should be around 1.0
        # Random prediction gives CE = log(m), normalized = 1.0
        total_classes = sum(config.MOD_RANGE)
        pred_logits = torch.zeros(100, total_classes)  # Uniform logits
        target_mods = torch.stack([
            torch.randint(0, m, (100,))
            for m in config.MOD_RANGE
        ], dim=-1)
        
        loss = model._compute_mod_loss(pred_logits, target_mods)
        
        # Should be close to 1.0 for uniform random prediction
        assert 0.5 < loss.item() < 2.0
