"""
test_vanilla_models.py:
Comprehensive tests for the Vanilla Transformer models (baseline for comparison).
Tests VanillaEmbeddings, VanillaModel, VanillaTransformerForPreTraining.
"""

import pytest
import torch
import math

from intseq_bert import config
from intseq_bert.vanilla_models import (
    VanillaEmbeddings,
    VanillaModel,
    VanillaTransformerForPreTraining
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


@pytest.fixture
def vocab_size():
    return 1000  # Smaller vocab for tests


@pytest.fixture
def sample_input_ids(batch_size, seq_len, vocab_size):
    """Creates sample input token IDs."""
    return torch.randint(1, vocab_size, (batch_size, seq_len))  # Avoid 0 (PAD)


@pytest.fixture
def sample_padding_mask(batch_size, seq_len):
    """Creates a sample padding mask with some padded positions."""
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    # Last 4 positions are padding for each sample
    mask[:, -4:] = True
    return mask


@pytest.fixture
def sample_labels(batch_size, seq_len, vocab_size):
    """Creates sample labels for loss computation."""
    # mask_map: True where we want to compute loss
    mask_map = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask_map[:, :8] = True  # First 8 positions are masked
    
    return {
        "token_targets": torch.randint(1, vocab_size, (batch_size, seq_len)),
        "mag_targets": torch.randn(batch_size, seq_len),
        "sign_targets": torch.randint(0, config.NUM_SIGN_CLASSES, (batch_size, seq_len)),
        "mod_targets": torch.stack([
            torch.randint(0, m, (batch_size, seq_len)) 
            for m in config.MOD_RANGE
        ], dim=-1),  # (B, L, 100)
        "mask_map": mask_map
    }


# ============================================================
# VanillaEmbeddings Tests
# ============================================================


class TestVanillaEmbeddings:
    """Tests for VanillaEmbeddings layer."""
    
    def test_output_shape(self, sample_input_ids, d_model, batch_size, seq_len, vocab_size):
        """Test output has correct shape."""
        embeddings = VanillaEmbeddings(d_model=d_model, vocab_size=vocab_size)
        output = embeddings(sample_input_ids)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_padding_idx(self, d_model, vocab_size):
        """Test padding index is correctly set."""
        embeddings = VanillaEmbeddings(d_model=d_model, vocab_size=vocab_size, pad_token_id=0)
        
        # Padding embedding should be all zeros
        pad_embedding = embeddings.token_embedding.weight[0]
        assert torch.allclose(pad_embedding, torch.zeros_like(pad_embedding))
    
    def test_embedding_initialization(self, d_model, vocab_size):
        """Test embeddings are properly initialized."""
        embeddings = VanillaEmbeddings(d_model=d_model, vocab_size=vocab_size)
        
        # Non-padding embeddings should not be all zeros
        non_pad_embedding = embeddings.token_embedding.weight[1]
        assert not torch.allclose(non_pad_embedding, torch.zeros_like(non_pad_embedding))
    
    def test_dropout_effect(self, sample_input_ids, d_model, vocab_size):
        """Test dropout has effect in training mode."""
        embeddings = VanillaEmbeddings(d_model=d_model, vocab_size=vocab_size, dropout=0.5)
        
        embeddings.train()
        out1 = embeddings(sample_input_ids)
        out2 = embeddings(sample_input_ids)
        
        # Outputs should differ due to dropout
        assert not torch.allclose(out1, out2)
        
        embeddings.eval()
        out3 = embeddings(sample_input_ids)
        out4 = embeddings(sample_input_ids)
        
        # In eval mode, should be deterministic
        assert torch.allclose(out3, out4)
    
    def test_scaling_factor(self, d_model, vocab_size):
        """Test embedding scaling by sqrt(d_model)."""
        embeddings = VanillaEmbeddings(d_model=d_model, vocab_size=vocab_size)
        assert embeddings.scale == pytest.approx(math.sqrt(d_model))


# ============================================================
# VanillaModel Tests
# ============================================================


class TestVanillaModel:
    """Tests for VanillaModel backbone."""
    
    def test_output_shape(self, sample_input_ids, sample_padding_mask, d_model, batch_size, seq_len, vocab_size):
        """Test output has correct shape."""
        model = VanillaModel(d_model=d_model, nhead=2, num_layers=2, vocab_size=vocab_size)
        output = model(sample_input_ids, src_key_padding_mask=sample_padding_mask)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_without_padding_mask(self, sample_input_ids, d_model, vocab_size):
        """Test model works without padding mask."""
        model = VanillaModel(d_model=d_model, nhead=2, num_layers=2, vocab_size=vocab_size)
        output = model(sample_input_ids)
        assert output.shape[-1] == d_model
    
    def test_gradient_flow(self, sample_input_ids, d_model, vocab_size):
        """Test gradients flow through the model."""
        model = VanillaModel(d_model=d_model, nhead=2, num_layers=2, vocab_size=vocab_size)
        output = model(sample_input_ids)
        loss = output.sum()
        loss.backward()
        
        # Check embedding gradients exist
        assert model.embeddings.token_embedding.weight.grad is not None
    
    def test_different_sequence_lengths(self, d_model, vocab_size):
        """Test model handles different sequence lengths."""
        model = VanillaModel(d_model=d_model, nhead=2, num_layers=2, vocab_size=vocab_size)
        
        for seq_len in [8, 32, 64, 128]:
            input_ids = torch.randint(1, vocab_size, (2, seq_len))
            output = model(input_ids)
            assert output.shape == (2, seq_len, d_model)


# ============================================================
# VanillaTransformerForPreTraining Tests
# ============================================================


class TestVanillaTransformerForPreTraining:
    """Tests for VanillaTransformerForPreTraining model."""
    
    def test_inference_output_structure(self, sample_input_ids, sample_padding_mask, d_model, batch_size, seq_len, vocab_size):
        """Test output structure during inference (no labels)."""
        model = VanillaTransformerForPreTraining(d_model=d_model, nhead=2, num_layers=2, vocab_size=vocab_size)
        
        outputs = model(sample_input_ids, sample_padding_mask)
        
        assert "predictions" in outputs
        assert "loss" not in outputs  # No loss without labels
        
        preds = outputs["predictions"]
        assert preds["logits"].shape == (batch_size, seq_len, vocab_size)
        assert preds["mag_mu"].shape == (batch_size, seq_len)
        assert preds["mag_log_var"].shape == (batch_size, seq_len)
        assert preds["sign_logits"].shape == (batch_size, seq_len, config.NUM_SIGN_CLASSES)
        assert preds["mod_logits"].shape == (batch_size, seq_len, sum(config.MOD_RANGE))
    
    def test_training_output_structure(self, sample_input_ids, sample_padding_mask, sample_labels, d_model, vocab_size):
        """Test output structure during training (with labels)."""
        model = VanillaTransformerForPreTraining(d_model=d_model, nhead=2, num_layers=2, vocab_size=vocab_size)
        
        outputs = model(sample_input_ids, sample_padding_mask, labels=sample_labels)
        
        assert "loss" in outputs
        assert "predictions" in outputs
        assert "loss_breakdown" in outputs
        
        assert outputs["loss"].dim() == 0  # Scalar
        assert outputs["loss"].requires_grad
    
    def test_loss_breakdown_keys(self, sample_input_ids, sample_padding_mask, sample_labels, d_model, vocab_size):
        """Test loss_breakdown contains all expected keys."""
        model = VanillaTransformerForPreTraining(d_model=d_model, nhead=2, num_layers=2, vocab_size=vocab_size)
        
        outputs = model(sample_input_ids, sample_padding_mask, labels=sample_labels)
        
        breakdown = outputs["loss_breakdown"]
        expected_keys = ["raw_lm", "raw_mag", "raw_sign", "raw_mod"]
        for key in expected_keys:
            assert key in breakdown, f"Missing key: {key}"
    
    def test_lm_head_output_shape(self, sample_input_ids, sample_padding_mask, d_model, batch_size, seq_len, vocab_size):
        """Test LM head produces correct output shape."""
        model = VanillaTransformerForPreTraining(d_model=d_model, nhead=2, num_layers=2, vocab_size=vocab_size)
        
        outputs = model(sample_input_ids, sample_padding_mask)
        
        assert outputs["predictions"]["logits"].shape == (batch_size, seq_len, vocab_size)
    
    def test_gradient_flow_through_loss(self, sample_input_ids, sample_padding_mask, sample_labels, d_model, vocab_size):
        """Test gradients flow through loss computation."""
        model = VanillaTransformerForPreTraining(d_model=d_model, nhead=2, num_layers=2, vocab_size=vocab_size)
        
        outputs = model(sample_input_ids, sample_padding_mask, labels=sample_labels)
        loss = outputs["loss"]
        loss.backward()
        
        # Check some parameters have gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients found"


# ============================================================
# Loss Computation Tests
# ============================================================


class TestVanillaLossComputation:
    """Tests for Vanilla Transformer loss calculation."""
    
    def test_loss_is_positive(self, sample_input_ids, sample_padding_mask, sample_labels, d_model, vocab_size):
        """Test that total loss is positive."""
        model = VanillaTransformerForPreTraining(d_model=d_model, nhead=2, num_layers=2, vocab_size=vocab_size)
        
        outputs = model(sample_input_ids, sample_padding_mask, labels=sample_labels)
        
        assert outputs["loss"].item() > -100  # Sanity check
    
    def test_raw_losses_valid(self, sample_input_ids, sample_padding_mask, sample_labels, d_model, vocab_size):
        """Test that individual raw losses are valid."""
        model = VanillaTransformerForPreTraining(d_model=d_model, nhead=2, num_layers=2, vocab_size=vocab_size)
        
        outputs = model(sample_input_ids, sample_padding_mask, labels=sample_labels)
        breakdown = outputs["loss_breakdown"]
        
        # CrossEntropy losses should be positive
        assert not torch.isnan(breakdown["raw_lm"])
        assert not torch.isnan(breakdown["raw_sign"])
        assert not torch.isnan(breakdown["raw_mod"])
    
    def test_split_mod_logits(self, d_model, vocab_size):
        """Test _split_mod_logits correctly splits the unified logits."""
        model = VanillaTransformerForPreTraining(d_model=d_model, nhead=2, num_layers=2, vocab_size=vocab_size)
        
        total_classes = sum(config.MOD_RANGE)
        fake_logits = torch.randn(8, total_classes)  # batch of 8
        
        split = model._split_mod_logits(fake_logits)
        
        assert len(split) == len(config.MOD_RANGE)
        assert split[0].shape == (8, 2)    # mod 2
        assert split[1].shape == (8, 3)    # mod 3
        assert split[-1].shape == (8, 101) # mod 101


# ============================================================
# Compatibility Tests
# ============================================================


class TestVanillaIntSeqCompatibility:
    """Tests for API compatibility between Vanilla and IntSeq models."""
    
    def test_predictions_have_same_keys(self, sample_input_ids, sample_padding_mask, d_model, vocab_size):
        """Test Vanilla model has compatible prediction keys with IntSeq."""
        from intseq_bert.intseq_models import IntSeqForPreTraining
        
        vanilla = VanillaTransformerForPreTraining(d_model=d_model, nhead=2, num_layers=2, vocab_size=vocab_size)
        intseq = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
        
        # Get prediction keys
        vanilla_out = vanilla(sample_input_ids, sample_padding_mask)
        
        # IntSeq needs different inputs
        batch_size, seq_len = sample_input_ids.shape
        mag = torch.randn(batch_size, seq_len, config.MAG_EXTENDED_DIM)
        mod = torch.randn(batch_size, seq_len, config.MOD_FEATURE_DIM)
        intseq_out = intseq(mag, mod, sample_padding_mask)
        
        vanilla_keys = set(vanilla_out["predictions"].keys())
        intseq_keys = set(intseq_out["predictions"].keys())
        
        # Vanilla has "logits" (token prediction), IntSeq doesn't
        # Both should have diagnostic keys
        common_keys = {"mag_mu", "mag_log_var", "sign_logits", "mod_logits"}
        assert common_keys.issubset(vanilla_keys)
        assert common_keys.issubset(intseq_keys)
    
    def test_mod_logits_same_shape(self, sample_input_ids, sample_padding_mask, d_model, batch_size, seq_len, vocab_size):
        """Test mod_logits have same shape between models."""
        from intseq_bert.intseq_models import IntSeqForPreTraining
        
        vanilla = VanillaTransformerForPreTraining(d_model=d_model, nhead=2, num_layers=2, vocab_size=vocab_size)
        intseq = IntSeqForPreTraining(d_model=d_model, nhead=2, num_layers=2)
        
        vanilla_out = vanilla(sample_input_ids, sample_padding_mask)
        
        mag = torch.randn(batch_size, seq_len, config.MAG_EXTENDED_DIM)
        mod = torch.randn(batch_size, seq_len, config.MOD_FEATURE_DIM)
        intseq_out = intseq(mag, mod, sample_padding_mask)
        
        assert vanilla_out["predictions"]["mod_logits"].shape == intseq_out["predictions"]["mod_logits"].shape


# ============================================================
# Config Tests
# ============================================================


class TestVanillaConfig:
    """Tests for Vanilla Transformer configuration."""
    
    def test_default_vocab_size(self, d_model):
        """Test default vocab size is used when not specified."""
        model = VanillaTransformerForPreTraining(d_model=d_model, nhead=2, num_layers=2)
        assert model.vocab_size == getattr(config, "VANILLA_VOCAB_SIZE", 30000)
    
    def test_custom_vocab_size(self, d_model):
        """Test custom vocab size is used when specified."""
        custom_vocab = 5000
        model = VanillaTransformerForPreTraining(d_model=d_model, nhead=2, num_layers=2, vocab_size=custom_vocab)
        assert model.vocab_size == custom_vocab
        assert model.lm_head.out_features == custom_vocab
    
    def test_custom_pad_token_id(self, d_model, vocab_size):
        """Test custom padding token ID is used."""
        custom_pad = 99
        model = VanillaTransformerForPreTraining(d_model=d_model, nhead=2, num_layers=2, vocab_size=vocab_size, pad_token_id=custom_pad)
        assert model.pad_token_id == custom_pad
