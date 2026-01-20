"""
tests/test_ablation_models.py

Unit tests for AblationForPreTraining and related classes.
Tests follow spec/ablation_models.md Section 4.1.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile

# Skip all tests if torch is not available or fails to import models
try:
    from intseq_bert import config
    from intseq_bert.ablation_models import (
        AblationEmbeddings,
        AblationModel,
        AblationForPreTraining,
    )
    ABLATION_AVAILABLE = True
except ImportError:
    ABLATION_AVAILABLE = False

requires_ablation = pytest.mark.skipif(
    not ABLATION_AVAILABLE,
    reason="Ablation models not available"
)


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def seq_length():
    return 16

@pytest.fixture
def d_model():
    return 64  # Small for testing

@pytest.fixture
def sample_mag_features(batch_size, seq_length):
    """Sample magnitude features (B, L, MAG_EXTENDED_DIM)."""
    return torch.randn(batch_size, seq_length, config.MAG_EXTENDED_DIM)

@pytest.fixture
def sample_mod_features(batch_size, seq_length):
    """Sample modulo features (B, L, MOD_FEATURE_DIM)."""
    return torch.randn(batch_size, seq_length, config.MOD_FEATURE_DIM)

@pytest.fixture
def sample_padding_mask(batch_size, seq_length):
    """Sample padding mask (B, L), True where padding."""
    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    # Set last 2 positions as padding for first batch
    mask[0, -2:] = True
    return mask

@pytest.fixture
def sample_labels(batch_size, seq_length):
    """Sample labels for training."""
    mask_map = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    # Mask some random positions
    mask_map[:, 5:8] = True
    
    return {
        "mag_targets": torch.randn(batch_size, seq_length) * 10,
        "sign_targets": torch.randint(0, 3, (batch_size, seq_length)),
        "mod_targets": torch.stack([
            torch.randint(0, m, (batch_size, seq_length))
            for m in config.MOD_RANGE
        ], dim=-1),
        "mask_map": mask_map
    }


# ==========================================
# AblationEmbeddings Tests
# ==========================================

@requires_ablation
class TestAblationEmbeddings:
    """Tests for AblationEmbeddings class."""
    
    def test_forward_output_shape(self, sample_mag_features, d_model):
        """Test that AblationEmbeddings produces correct output shape."""
        embeddings = AblationEmbeddings(d_model=d_model)
        
        output = embeddings(sample_mag_features)
        
        B, L = sample_mag_features.shape[:2]
        assert output.shape == (B, L, d_model)
    
    def test_forward_dtype(self, sample_mag_features, d_model):
        """Test that output is float32 (FP32 enforced)."""
        embeddings = AblationEmbeddings(d_model=d_model)
        
        output = embeddings(sample_mag_features.half())  # Input as FP16
        
        # Output should still be FP32 due to autocast disabled
        assert output.dtype == torch.float32
    
    def test_no_mod_features_input(self, sample_mag_features, d_model):
        """Test that forward only accepts mag_features (no mod_features)."""
        embeddings = AblationEmbeddings(d_model=d_model)
        
        # Should work with only mag_features
        output = embeddings(sample_mag_features)
        assert output is not None
    
    def test_layer_norm_exists(self, d_model):
        """Test that LayerNorm is applied."""
        embeddings = AblationEmbeddings(d_model=d_model)
        
        assert hasattr(embeddings, 'layer_norm')
        assert isinstance(embeddings.layer_norm, nn.LayerNorm)
    
    def test_positional_encoding_exists(self, d_model):
        """Test that PositionalEncoding is applied."""
        embeddings = AblationEmbeddings(d_model=d_model)
        
        assert hasattr(embeddings, 'pos_encoding')


# ==========================================
# AblationModel Tests
# ==========================================

@requires_ablation
class TestAblationModel:
    """Tests for AblationModel (backbone) class."""
    
    def test_forward_output_shape(self, sample_mag_features, sample_padding_mask, d_model):
        """Test AblationModel produces (B, L, d_model) output."""
        model = AblationModel(d_model=d_model, nhead=4, num_layers=2)
        
        output = model(sample_mag_features, sample_padding_mask)
        
        B, L = sample_mag_features.shape[:2]
        assert output.shape == (B, L, d_model)
    
    def test_forward_without_padding_mask(self, sample_mag_features, d_model):
        """Test model works without padding mask."""
        model = AblationModel(d_model=d_model, nhead=4, num_layers=2)
        
        output = model(sample_mag_features)
        
        B, L = sample_mag_features.shape[:2]
        assert output.shape == (B, L, d_model)
    
    def test_encoder_layers(self, d_model):
        """Test that encoder has correct number of layers."""
        num_layers = 3
        model = AblationModel(d_model=d_model, nhead=4, num_layers=num_layers)
        
        assert len(model.encoder.layers) == num_layers
    
    def test_inherits_base_pretrained_model(self):
        """Test that AblationModel inherits from BasePreTrainedModel."""
        from intseq_bert.base_models import BasePreTrainedModel
        
        model = AblationModel(d_model=64, nhead=4, num_layers=2)
        assert isinstance(model, BasePreTrainedModel)


# ==========================================
# AblationForPreTraining Tests
# ==========================================

@requires_ablation
class TestAblationForPreTraining:
    """Tests for AblationForPreTraining class."""
    
    def test_forward_predictions_keys(
        self, sample_mag_features, sample_mod_features, sample_padding_mask, d_model
    ):
        """Test predictions contain expected keys: mag_mu, sign_logits, mod_logits."""
        model = AblationForPreTraining(d_model=d_model, nhead=4, num_layers=2)
        
        outputs = model(
            mag_features=sample_mag_features,
            mod_features=sample_mod_features,
            src_key_padding_mask=sample_padding_mask
        )
        
        assert "predictions" in outputs
        preds = outputs["predictions"]
        assert "mag_mu" in preds
        assert "mag_log_var" in preds
        assert "sign_logits" in preds
        assert "mod_logits" in preds
    
    def test_forward_prediction_shapes(
        self, sample_mag_features, sample_mod_features, sample_padding_mask, d_model, 
        batch_size, seq_length
    ):
        """Test prediction tensors have correct shapes."""
        model = AblationForPreTraining(d_model=d_model, nhead=4, num_layers=2)
        
        outputs = model(
            mag_features=sample_mag_features,
            mod_features=sample_mod_features,
            src_key_padding_mask=sample_padding_mask
        )
        
        preds = outputs["predictions"]
        
        assert preds["mag_mu"].shape == (batch_size, seq_length)
        assert preds["mag_log_var"].shape == (batch_size, seq_length)
        assert preds["sign_logits"].shape == (batch_size, seq_length, config.NUM_SIGN_CLASSES)
        
        total_mod_classes = sum(config.MOD_RANGE)
        assert preds["mod_logits"].shape == (batch_size, seq_length, total_mod_classes)
    
    def test_loss_computation(
        self, sample_mag_features, sample_mod_features, sample_padding_mask, 
        sample_labels, d_model
    ):
        """Test that labels produces loss output."""
        model = AblationForPreTraining(d_model=d_model, nhead=4, num_layers=2)
        
        outputs = model(
            mag_features=sample_mag_features,
            mod_features=sample_mod_features,
            src_key_padding_mask=sample_padding_mask,
            labels=sample_labels
        )
        
        assert "loss" in outputs
        assert isinstance(outputs["loss"], torch.Tensor)
        assert outputs["loss"].shape == ()  # Scalar
        assert outputs["loss"].requires_grad
    
    def test_loss_breakdown(
        self, sample_mag_features, sample_mod_features, sample_padding_mask, 
        sample_labels, d_model
    ):
        """Test that loss_breakdown contains component losses."""
        model = AblationForPreTraining(d_model=d_model, nhead=4, num_layers=2)
        
        outputs = model(
            mag_features=sample_mag_features,
            mod_features=sample_mod_features,
            src_key_padding_mask=sample_padding_mask,
            labels=sample_labels
        )
        
        assert "loss_breakdown" in outputs
        breakdown = outputs["loss_breakdown"]
        assert "raw_mag" in breakdown
        assert "raw_sign" in breakdown
        assert "raw_mod" in breakdown
    
    def test_mod_features_ignored(
        self, sample_mag_features, sample_padding_mask, d_model
    ):
        """Test that mod_features is ignored (changing it doesn't change output)."""
        model = AblationForPreTraining(d_model=d_model, nhead=4, num_layers=2)
        model.eval()  # Disable dropout for deterministic comparison
        
        B, L = sample_mag_features.shape[:2]
        
        # Different mod_features
        mod_features_1 = torch.zeros(B, L, config.MOD_FEATURE_DIM)
        mod_features_2 = torch.ones(B, L, config.MOD_FEATURE_DIM) * 100
        
        with torch.no_grad():
            output_1 = model(
                mag_features=sample_mag_features,
                mod_features=mod_features_1,
                src_key_padding_mask=sample_padding_mask
            )
            output_2 = model(
                mag_features=sample_mag_features,
                mod_features=mod_features_2,
                src_key_padding_mask=sample_padding_mask
            )
        
        # Predictions should be identical since mod_features is ignored
        torch.testing.assert_close(
            output_1["predictions"]["mag_mu"],
            output_2["predictions"]["mag_mu"]
        )
        torch.testing.assert_close(
            output_1["predictions"]["sign_logits"],
            output_2["predictions"]["sign_logits"]
        )
    
    def test_inherits_base_for_pretraining(self, d_model):
        """Test that AblationForPreTraining inherits from BaseForPreTraining."""
        from intseq_bert.base_models import BaseForPreTraining
        
        model = AblationForPreTraining(d_model=d_model, nhead=4, num_layers=2)
        assert isinstance(model, BaseForPreTraining)
    
    def test_has_split_mod_logits(self, d_model):
        """Test that _split_mod_logits method is available."""
        model = AblationForPreTraining(d_model=d_model, nhead=4, num_layers=2)
        
        assert hasattr(model, '_split_mod_logits')
        assert callable(model._split_mod_logits)


# ==========================================
# Checkpoint Tests
# ==========================================

@requires_ablation
class TestCheckpointLoading:
    """Tests for checkpoint save/load functionality."""
    
    def test_save_and_load_checkpoint(self, d_model):
        """Test that model can be saved and loaded from checkpoint."""
        model = AblationForPreTraining(d_model=d_model, nhead=4, num_layers=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            
            # Save checkpoint
            state = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "d_model": d_model,
                    "nhead": 4,
                    "num_layers": 2
                }
            }
            torch.save(state, checkpoint_path)
            
            # Load checkpoint
            loaded_model = AblationForPreTraining.from_checkpoint(
                str(checkpoint_path), 
                device="cpu",
                d_model=d_model,
                nhead=4,
                num_layers=2
            )
            
            assert loaded_model is not None
            
            # Compare state dicts
            for key in model.state_dict():
                torch.testing.assert_close(
                    model.state_dict()[key],
                    loaded_model.state_dict()[key]
                )
    
    def test_from_checkpoint_class_method(self, d_model):
        """Test that from_checkpoint is a classmethod."""
        assert hasattr(AblationForPreTraining, 'from_checkpoint')
        
        # It should be callable as a class method
        assert callable(getattr(AblationForPreTraining, 'from_checkpoint'))


# ==========================================
# Integration Tests
# ==========================================

@requires_ablation
class TestAblationIntegration:
    """Integration tests for ablation model with other components."""
    
    def test_model_in_models_module(self):
        """Test that ablation models are exported from models.py."""
        from intseq_bert import models
        
        assert hasattr(models, 'AblationEmbeddings')
        assert hasattr(models, 'AblationModel')
        assert hasattr(models, 'AblationForPreTraining')
    
    def test_backward_pass(
        self, sample_mag_features, sample_mod_features, sample_padding_mask, 
        sample_labels, d_model
    ):
        """Test that backward pass works correctly."""
        model = AblationForPreTraining(d_model=d_model, nhead=4, num_layers=2)
        
        outputs = model(
            mag_features=sample_mag_features,
            mod_features=sample_mod_features,
            src_key_padding_mask=sample_padding_mask,
            labels=sample_labels
        )
        
        loss = outputs["loss"]
        loss.backward()
        
        # Check that gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
