"""
Tests for IntSeqBERT model (Dual Stream Architecture + Multitask Classification).
"""

import pytest
import torch
import tempfile
from pathlib import Path

from intseq_bert import bert_model


# ==========================================
# Helper Functions
# ==========================================

def get_minimal_model_config():
    """Get minimal model configuration for testing."""
    return {
        'mag_dim': 5,
        'mod_dim': 200,
        'd_model': 32,
        'nhead': 2,
        'num_layers': 1,
        'dim_feedforward': 64,
        'dropout': 0.1,
        'max_len': 100,
        'multitask': True
    }


def create_mock_inputs(batch_size=2, seq_len=10, mag_dim=5, mod_dim=200):
    """Create mock inputs for testing."""
    return {
        'mag_inputs': torch.randn(batch_size, seq_len, mag_dim),
        'mod_inputs': torch.randn(batch_size, seq_len, mod_dim),
        'attention_mask': torch.ones(batch_size, seq_len),
    }


# ==========================================
# 1. MOD_RANGE Tests
# ==========================================

class TestModRange:
    """Tests for MOD_RANGE constant."""
    
    def test_mod_range_defined(self):
        """Test MOD_RANGE is defined in bert_model."""
        assert hasattr(bert_model, 'MOD_RANGE')
    
    def test_mod_range_values(self):
        """Test MOD_RANGE covers 2 to 101."""
        assert bert_model.MOD_RANGE == list(range(2, 102))
        assert len(bert_model.MOD_RANGE) == 100


# ==========================================
# 2. PositionalEncoding Tests
# ==========================================

class TestPositionalEncoding:
    """Tests for PositionalEncoding module."""
    
    def test_output_shape(self):
        """Test output shape matches input."""
        pe = bert_model.PositionalEncoding(d_model=32, max_len=100)
        x = torch.randn(2, 10, 32)
        out = pe(x)
        assert out.shape == x.shape
    
    def test_long_sequence_handling(self):
        """Test that long sequences are truncated gracefully."""
        pe = bert_model.PositionalEncoding(d_model=32, max_len=50)
        x = torch.randn(2, 100, 32)  # Longer than max_len
        out = pe(x)
        assert out.shape == (2, 50, 32)


# ==========================================
# 3. Model Initialization Tests
# ==========================================

class TestModelInitialization:
    """Tests for IntSeqBERT initialization."""
    
    def test_default_initialization(self):
        """Test model initializes with defaults."""
        model = bert_model.IntSeqBERT()
        assert model.d_model == 128
        assert model.mag_dim == 5
        assert model.mod_dim == 200
        assert model.multitask == True
    
    def test_custom_initialization(self):
        """Test model with custom config."""
        config = get_minimal_model_config()
        model = bert_model.IntSeqBERT(**config)
        assert model.d_model == 32
        assert model.multitask == True
    
    def test_multitask_disabled(self):
        """Test model with multitask disabled."""
        model = bert_model.IntSeqBERT(d_model=32, num_layers=1, multitask=False)
        assert model.multitask == False
        assert not hasattr(model, 'mod_cls_heads')
    
    def test_required_components_exist(self):
        """Test all required components are present."""
        model = bert_model.IntSeqBERT(**get_minimal_model_config())
        
        # Projections
        assert hasattr(model, 'mag_proj')
        assert hasattr(model, 'mod_proj')
        assert hasattr(model, 'fusion_norm')
        
        # Encoder
        assert hasattr(model, 'pos_encoder')
        assert hasattr(model, 'encoder')
        
        # Reconstruction heads
        assert hasattr(model, 'mag_head')
        assert hasattr(model, 'mod_head')
        
        # Multitask heads
        assert hasattr(model, 'mod_cls_heads')
    
    def test_mod_cls_heads_structure(self):
        """Test multitask classification heads structure."""
        model = bert_model.IntSeqBERT(**get_minimal_model_config())
        
        # 100 heads for mod 2-101
        assert len(model.mod_cls_heads) == 100
        
        # Each head outputs correct number of classes
        for m in range(2, 102):
            assert f"mod{m}" in model.mod_cls_heads
            head = model.mod_cls_heads[f"mod{m}"]
            assert head.out_features == m


# ==========================================
# 4. Forward Pass Tests
# ==========================================

class TestForwardPass:
    """Tests for forward pass."""
    
    def test_output_keys_multitask(self):
        """Test output contains all expected keys with multitask."""
        model = bert_model.IntSeqBERT(**get_minimal_model_config())
        inputs = create_mock_inputs()
        
        output = model(**inputs)
        
        # Core outputs
        assert 'encoded_state' in output
        assert 'pred_mag' in output
        assert 'pred_mod' in output
        assert 'loss' in output
        
        # Multitask outputs
        for m in range(2, 102):
            assert f"mod{m}" in output
    
    def test_output_keys_no_multitask(self):
        """Test output without multitask."""
        config = get_minimal_model_config()
        config['multitask'] = False
        model = bert_model.IntSeqBERT(**config)
        inputs = create_mock_inputs()
        
        output = model(**inputs)
        
        # Core outputs only
        assert 'encoded_state' in output
        assert 'pred_mag' in output
        assert 'pred_mod' in output
        assert 'mod2' not in output
    
    def test_output_shapes(self):
        """Test output tensor shapes."""
        config = get_minimal_model_config()
        model = bert_model.IntSeqBERT(**config)
        
        batch_size, seq_len = 2, 10
        inputs = create_mock_inputs(batch_size, seq_len)
        
        output = model(**inputs)
        
        assert output['encoded_state'].shape == (batch_size, seq_len, config['d_model'])
        assert output['pred_mag'].shape == (batch_size, seq_len, config['mag_dim'])
        assert output['pred_mod'].shape == (batch_size, seq_len, config['mod_dim'])
    
    def test_mod_cls_output_shapes(self):
        """Test classification head output shapes."""
        config = get_minimal_model_config()
        model = bert_model.IntSeqBERT(**config)
        
        batch_size, seq_len = 2, 10
        inputs = create_mock_inputs(batch_size, seq_len)
        
        output = model(**inputs)
        
        # Each mod head outputs (B, L, m)
        for m in [2, 10, 50, 101]:
            assert output[f"mod{m}"].shape == (batch_size, seq_len, m)
    
    def test_loss_is_none_without_labels(self):
        """Test loss is None when labels not provided."""
        model = bert_model.IntSeqBERT(**get_minimal_model_config())
        inputs = create_mock_inputs()
        
        output = model(**inputs)
        
        assert output['loss'] is None


# ==========================================
# 5. Loss Computation Tests
# ==========================================

class TestLossComputation:
    """Tests for loss computation."""
    
    def test_loss_computed_with_labels(self):
        """Test loss is computed when labels provided."""
        config = get_minimal_model_config()
        model = bert_model.IntSeqBERT(**config)
        
        batch_size, seq_len = 2, 10
        inputs = create_mock_inputs(batch_size, seq_len)
        
        # Add labels and mask
        inputs['mag_labels'] = torch.randn(batch_size, seq_len, config['mag_dim'])
        inputs['mod_labels'] = torch.randn(batch_size, seq_len, config['mod_dim'])
        inputs['mask_matrix'] = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        inputs['mask_matrix'][:, :3] = True  # Mask first 3 positions
        
        output = model(**inputs)
        
        assert output['loss'] is not None
        assert output['loss'].dim() == 0  # Scalar
    
    def test_loss_zero_with_no_mask(self):
        """Test loss is zero when no positions are masked."""
        config = get_minimal_model_config()
        model = bert_model.IntSeqBERT(**config)
        
        batch_size, seq_len = 2, 10
        inputs = create_mock_inputs(batch_size, seq_len)
        
        inputs['mag_labels'] = torch.randn(batch_size, seq_len, config['mag_dim'])
        inputs['mod_labels'] = torch.randn(batch_size, seq_len, config['mod_dim'])
        inputs['mask_matrix'] = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        output = model(**inputs)
        
        assert output['loss'].item() == 0.0


# ==========================================
# 6. Gradient Flow Tests
# ==========================================

class TestGradientFlow:
    """Tests for gradient flow."""
    
    def test_gradients_flow_to_all_components(self):
        """Test gradients flow through all components."""
        config = get_minimal_model_config()
        model = bert_model.IntSeqBERT(**config)
        
        batch_size, seq_len = 2, 10
        inputs = create_mock_inputs(batch_size, seq_len)
        inputs['mag_labels'] = torch.randn(batch_size, seq_len, config['mag_dim'])
        inputs['mod_labels'] = torch.randn(batch_size, seq_len, config['mod_dim'])
        inputs['mask_matrix'] = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        output = model(**inputs)
        output['loss'].backward()
        
        # Check gradients on projection layers
        assert model.mag_proj.weight.grad is not None
        assert model.mod_proj.weight.grad is not None
    
    def test_multitask_heads_gradients(self):
        """Test gradients flow to multitask heads."""
        config = get_minimal_model_config()
        model = bert_model.IntSeqBERT(**config)
        
        inputs = create_mock_inputs(2, 10)
        output = model(**inputs)
        
        # Use mod100 output for loss
        loss = output['mod100'].mean()
        loss.backward()
        
        # Multitask head should have gradients
        assert model.mod_cls_heads['mod100'].weight.grad is not None


# ==========================================
# 7. Checkpoint Tests
# ==========================================

class TestCheckpointing:
    """Tests for model checkpointing."""
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading model."""
        config = get_minimal_model_config()
        model = bert_model.IntSeqBERT(**config)
        
        # Save
        checkpoint_path = tmp_path / "test_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, checkpoint_path)
        
        # Load
        loaded_model, checkpoint = bert_model.IntSeqBERT.load_from_checkpoint(
            str(checkpoint_path), device='cpu'
        )
        
        assert loaded_model.d_model == config['d_model']
        assert loaded_model.multitask == config['multitask']
    
    def test_load_old_checkpoint_without_multitask(self, tmp_path):
        """Test loading old checkpoint without multitask heads."""
        # Create model without multitask
        old_config = get_minimal_model_config()
        old_config['multitask'] = False
        old_model = bert_model.IntSeqBERT(**old_config)
        
        # Save without multitask heads
        checkpoint_path = tmp_path / "old_model.pt"
        torch.save({
            'model_state_dict': old_model.state_dict(),
            'config': old_config
        }, checkpoint_path)
        
        # Load - should preserve multitask=False from config
        loaded_model, _ = bert_model.IntSeqBERT.load_from_checkpoint(
            str(checkpoint_path), device='cpu'
        )
        
        # multitask=False is preserved from config
        assert loaded_model.multitask == False
        
        inputs = create_mock_inputs()
        output = loaded_model(**inputs)
        
        # No multitask outputs
        assert 'mod100' not in output


# ==========================================
# 8. Edge Cases
# ==========================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_batch_size_one(self):
        """Test with batch size 1."""
        model = bert_model.IntSeqBERT(**get_minimal_model_config())
        inputs = create_mock_inputs(batch_size=1, seq_len=10)
        
        output = model(**inputs)
        
        assert output['encoded_state'].shape[0] == 1
    
    def test_very_short_sequence(self):
        """Test with very short sequence."""
        model = bert_model.IntSeqBERT(**get_minimal_model_config())
        inputs = create_mock_inputs(batch_size=2, seq_len=2)
        
        output = model(**inputs)
        
        assert output['pred_mag'].shape[1] == 2
    
    def test_eval_mode(self):
        """Test model in eval mode."""
        model = bert_model.IntSeqBERT(**get_minimal_model_config())
        model.eval()
        
        with torch.no_grad():
            inputs = create_mock_inputs()
            output = model(**inputs)
        
        assert output['encoded_state'] is not None
    
    def test_padding_mask(self):
        """Test with padding mask."""
        model = bert_model.IntSeqBERT(**get_minimal_model_config())
        inputs = create_mock_inputs(batch_size=2, seq_len=10)
        
        # Mask last 5 positions
        inputs['attention_mask'][:, 5:] = 0
        
        output = model(**inputs)
        
        assert output['encoded_state'].shape == (2, 10, 32)


# ==========================================
# 9. Parameter Count Tests
# ==========================================

class TestParameterCount:
    """Tests for parameter counting."""
    
    def test_multitask_increases_params(self):
        """Test that multitask heads increase parameter count."""
        config = get_minimal_model_config()
        
        config['multitask'] = False
        model_no_mt = bert_model.IntSeqBERT(**config)
        params_no_mt = sum(p.numel() for p in model_no_mt.parameters())
        
        config['multitask'] = True
        model_mt = bert_model.IntSeqBERT(**config)
        params_mt = sum(p.numel() for p in model_mt.parameters())
        
        assert params_mt > params_no_mt
