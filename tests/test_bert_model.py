"""
Tests for IntSeqBERT Dual Stream model.
Tests the Transformer encoder with Magnitude + Mod Spectrum fusion.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path

from intseq_bert.bert_model import IntSeqBERT, PositionalEncoding


# ==========================================
# Helper Functions
# ==========================================

def create_mock_inputs(batch_size: int = 2, seq_len: int = 10):
    """Create mock inputs for testing."""
    return {
        'mag_inputs': torch.randn(batch_size, seq_len, 5),
        'mod_inputs': torch.randn(batch_size, seq_len, 200),
        'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long),
        'mag_labels': torch.randn(batch_size, seq_len, 5),
        'mod_labels': torch.randn(batch_size, seq_len, 200),
        'mask_matrix': torch.zeros(batch_size, seq_len, dtype=torch.bool)
    }


@pytest.fixture
def sample_model():
    """Create a small IntSeqBERT model for testing."""
    return IntSeqBERT(
        mag_dim=5,
        mod_dim=200,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1
    )


# ==========================================
# 1. PositionalEncoding Tests
# ==========================================

class TestPositionalEncoding:
    """Tests for PositionalEncoding class."""
    
    def test_initialization(self):
        """Test positional encoding can be initialized."""
        pe = PositionalEncoding(d_model=64)
        assert pe.pe.shape == (5000, 64)  # Default max_len
    
    def test_output_shape(self):
        """Test output shape matches input."""
        pe = PositionalEncoding(d_model=64)
        x = torch.randn(2, 10, 64)
        
        output = pe(x)
        assert output.shape == x.shape
    
    def test_long_sequence_truncation(self):
        """Test that long sequences are handled gracefully."""
        pe = PositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(2, 150, 64)  # Longer than max_len
        
        output = pe(x)
        # Should truncate to max_len
        assert output.shape == (2, 100, 64)


# ==========================================
# 2. Model Initialization Tests
# ==========================================

class TestModelInitialization:
    """Tests for IntSeqBERT initialization."""
    
    def test_default_initialization(self):
        """Test model initializes with default parameters."""
        model = IntSeqBERT()
        
        assert model.mag_dim == 5
        assert model.mod_dim == 200
        assert model.d_model == 128
    
    def test_custom_initialization(self):
        """Test model initializes with custom parameters."""
        model = IntSeqBERT(
            mag_dim=10,
            mod_dim=300,
            d_model=256,
            nhead=8,
            num_layers=4
        )
        
        assert model.mag_dim == 10
        assert model.mod_dim == 300
        assert model.d_model == 256
    
    def test_has_required_components(self, sample_model):
        """Test model has all required components."""
        model = sample_model
        
        # Projection layers
        assert hasattr(model, 'mag_proj')
        assert hasattr(model, 'mod_proj')
        assert hasattr(model, 'fusion_norm')
        
        # Transformer components
        assert hasattr(model, 'pos_encoder')
        assert hasattr(model, 'encoder')
        
        # Prediction heads
        assert hasattr(model, 'mag_head')
        assert hasattr(model, 'mod_head')
    
    def test_parameter_count(self, sample_model):
        """Test that model has learnable parameters."""
        num_params = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
        assert num_params > 0


# ==========================================
# 3. Forward Pass Tests
# ==========================================

class TestForwardPass:
    """Tests for forward pass behavior."""
    
    def test_output_keys(self, sample_model):
        """Test forward returns correct keys."""
        inputs = create_mock_inputs()
        
        output = sample_model(
            inputs['mag_inputs'],
            inputs['mod_inputs'],
            inputs['attention_mask']
        )
        
        assert 'encoded_state' in output
        assert 'pred_mag' in output
        assert 'pred_mod' in output
        assert 'loss' in output
    
    def test_output_shapes(self, sample_model):
        """Test output shapes are correct."""
        batch_size, seq_len = 2, 10
        inputs = create_mock_inputs(batch_size, seq_len)
        
        output = sample_model(
            inputs['mag_inputs'],
            inputs['mod_inputs'],
            inputs['attention_mask']
        )
        
        assert output['encoded_state'].shape == (batch_size, seq_len, 64)  # d_model=64
        assert output['pred_mag'].shape == (batch_size, seq_len, 5)
        assert output['pred_mod'].shape == (batch_size, seq_len, 200)
    
    def test_loss_none_without_labels(self, sample_model):
        """Test loss is None when labels not provided."""
        inputs = create_mock_inputs()
        
        output = sample_model(
            inputs['mag_inputs'],
            inputs['mod_inputs'],
            inputs['attention_mask']
        )
        
        assert output['loss'] is None
    
    def test_loss_computed_with_labels(self, sample_model):
        """Test loss is computed when labels are provided."""
        inputs = create_mock_inputs()
        # Set some positions as masked
        inputs['mask_matrix'][:, :3] = True
        
        output = sample_model(
            inputs['mag_inputs'],
            inputs['mod_inputs'],
            inputs['attention_mask'],
            inputs['mag_labels'],
            inputs['mod_labels'],
            inputs['mask_matrix']
        )
        
        assert output['loss'] is not None
        assert output['loss'].shape == ()  # Scalar
        assert output['loss'].item() >= 0


# ==========================================
# 4. Gradient Flow Tests
# ==========================================

class TestGradientFlow:
    """Tests for gradient flow through the model."""
    
    def test_gradients_flow(self, sample_model):
        """Test that gradients flow through all components."""
        inputs = create_mock_inputs()
        inputs['mask_matrix'][:, :5] = True
        
        output = sample_model(
            inputs['mag_inputs'],
            inputs['mod_inputs'],
            inputs['attention_mask'],
            inputs['mag_labels'],
            inputs['mod_labels'],
            inputs['mask_matrix']
        )
        
        output['loss'].backward()
        
        # Check gradients exist for key parameters
        assert sample_model.mag_proj.weight.grad is not None
        assert sample_model.mod_proj.weight.grad is not None
        assert sample_model.mag_head[0].weight.grad is not None
        assert sample_model.mod_head[0].weight.grad is not None


# ==========================================
# 5. Attention Mask Tests
# ==========================================

class TestAttentionMask:
    """Tests for attention mask handling."""
    
    def test_padding_mask_applied(self, sample_model):
        """Test that padding mask affects output."""
        batch_size, seq_len = 2, 10
        inputs = create_mock_inputs(batch_size, seq_len)
        
        # Full sequence
        output1 = sample_model(
            inputs['mag_inputs'],
            inputs['mod_inputs'],
            inputs['attention_mask']
        )
        
        # Partially padded
        inputs['attention_mask'][0, 5:] = 0
        output2 = sample_model(
            inputs['mag_inputs'],
            inputs['mod_inputs'],
            inputs['attention_mask']
        )
        
        # Encoded states should differ
        assert not torch.allclose(output1['encoded_state'], output2['encoded_state'])


# ==========================================
# 6. Checkpoint Loading Tests
# ==========================================

class TestCheckpointLoading:
    """Tests for checkpoint save/load functionality."""
    
    def test_save_and_load(self, sample_model, tmp_path):
        """Test model can be saved and loaded."""
        # Save checkpoint
        checkpoint_path = tmp_path / "model.pt"
        torch.save({
            'model_state_dict': sample_model.state_dict(),
            'config': {
                'mag_dim': 5,
                'mod_dim': 200,
                'd_model': 64,
                'nhead': 4,
                'num_layers': 2,
                'dim_feedforward': 128,
                'dropout': 0.1
            }
        }, checkpoint_path)
        
        # Load checkpoint
        loaded_model, checkpoint = IntSeqBERT.load_from_checkpoint(
            str(checkpoint_path),
            device='cpu'
        )
        
        # Verify architecture matches
        assert loaded_model.mag_dim == 5
        assert loaded_model.mod_dim == 200
        assert loaded_model.d_model == 64
    
    def test_loaded_model_produces_same_output(self, sample_model, tmp_path):
        """Test loaded model produces same output as original."""
        sample_model.eval()
        
        # Save
        checkpoint_path = tmp_path / "model.pt"
        torch.save({
            'model_state_dict': sample_model.state_dict(),
            'config': {
                'mag_dim': 5,
                'mod_dim': 200,
                'd_model': 64,
                'nhead': 4,
                'num_layers': 2,
                'dim_feedforward': 128,
                'dropout': 0.1
            }
        }, checkpoint_path)
        
        # Load
        loaded_model, _ = IntSeqBERT.load_from_checkpoint(str(checkpoint_path), device='cpu')
        loaded_model.eval()
        
        # Compare outputs
        inputs = create_mock_inputs()
        
        with torch.no_grad():
            output1 = sample_model(inputs['mag_inputs'], inputs['mod_inputs'], inputs['attention_mask'])
            output2 = loaded_model(inputs['mag_inputs'], inputs['mod_inputs'], inputs['attention_mask'])
        
        assert torch.allclose(output1['pred_mag'], output2['pred_mag'])
        assert torch.allclose(output1['pred_mod'], output2['pred_mod'])


# ==========================================
# 7. Edge Cases
# ==========================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_item_batch(self, sample_model):
        """Test with batch size of 1."""
        inputs = create_mock_inputs(batch_size=1, seq_len=10)
        
        output = sample_model(
            inputs['mag_inputs'],
            inputs['mod_inputs'],
            inputs['attention_mask']
        )
        
        assert output['pred_mag'].shape == (1, 10, 5)
    
    def test_variable_sequence_lengths(self, sample_model):
        """Test model handles different sequence lengths."""
        for seq_len in [5, 10, 50, 100]:
            inputs = create_mock_inputs(batch_size=2, seq_len=seq_len)
            
            output = sample_model(
                inputs['mag_inputs'],
                inputs['mod_inputs'],
                inputs['attention_mask']
            )
            
            assert output['pred_mag'].shape == (2, seq_len, 5)
    
    def test_no_masked_positions(self, sample_model):
        """Test loss computation when no positions are masked."""
        inputs = create_mock_inputs()
        # No positions masked (all False)
        inputs['mask_matrix'] = torch.zeros(2, 10, dtype=torch.bool)
        
        output = sample_model(
            inputs['mag_inputs'],
            inputs['mod_inputs'],
            inputs['attention_mask'],
            inputs['mag_labels'],
            inputs['mod_labels'],
            inputs['mask_matrix']
        )
        
        # Loss should be 0 when no positions are masked
        assert output['loss'].item() == 0.0


# ==========================================
# 8. Device Compatibility Tests
# ==========================================

class TestDeviceCompatibility:
    """Tests for device compatibility."""
    
    def test_cpu_execution(self, sample_model):
        """Test model runs on CPU."""
        model = sample_model.cpu()
        inputs = create_mock_inputs()
        
        output = model(
            inputs['mag_inputs'],
            inputs['mod_inputs'],
            inputs['attention_mask']
        )
        
        assert output['pred_mag'].device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_execution(self, sample_model):
        """Test model runs on CUDA."""
        model = sample_model.cuda()
        inputs = create_mock_inputs()
        
        mag = inputs['mag_inputs'].cuda()
        mod = inputs['mod_inputs'].cuda()
        attn = inputs['attention_mask'].cuda()
        
        output = model(mag, mod, attn)
        
        assert output['pred_mag'].device.type == 'cuda'
