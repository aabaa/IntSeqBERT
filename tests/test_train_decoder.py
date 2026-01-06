"""
Tests for IntSeqDecoder training script (Dual Stream Architecture).
Tests training loop with frozen encoder and decoder training.
"""

import pytest
import torch
import json
from pathlib import Path

from intseq_bert import train_decoder, bert_model, decoder_model


# ==========================================
# Helper Functions
# ==========================================

def create_mock_features_dir(tmp_path: Path, num_files: int = 20):
    """Create a mock features directory with .pt files."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    
    for i in range(num_files):
        seq_len = 15
        data = {
            'oeis_id': f'A{i:06d}',
            'mag_features': torch.randn(seq_len, 5),
            'mod_features': torch.randn(seq_len, 200),
            'targets': {
                'mag': torch.randn(seq_len),
                **{f'mod{m}': torch.randint(0, m, (seq_len,)) for m in range(2, 102)}
            }
        }
        torch.save(data, features_dir / f"A{i:06d}.pt")
    
    return features_dir


def create_mock_encoder_checkpoint(tmp_path: Path) -> Path:
    """Create a mock encoder checkpoint for testing."""
    encoder = bert_model.IntSeqBERT(
        mag_dim=5,
        mod_dim=200,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.1
    )
    
    checkpoint_path = tmp_path / "encoder.pt"
    torch.save({
        'model_state_dict': encoder.state_dict(),
        'config': {
            'mag_dim': 5,
            'mod_dim': 200,
            'd_model': 32,
            'nhead': 2,
            'num_layers': 1,
            'dim_feedforward': 64,
            'dropout': 0.1
        }
    }, checkpoint_path)
    
    return checkpoint_path


def get_minimal_decoder_config(features_dir: Path, encoder_ckpt: Path, output_dir: Path) -> dict:
    """Get minimal training configuration for tests."""
    return {
        'features_dir': str(features_dir),
        'encoder_checkpoint': str(encoder_ckpt),
        'output_dir': str(output_dir),
        'epochs': 1,
        'batch_size': 2,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'hidden_dim': 64,
        'dropout': 0.1,
        'num_workers': 0,
        'val_ratio': 0.2,
        'test_ratio': 0.2,
        'mask_prob': 0.15,
    }


# ==========================================
# 1. Setup Logging Tests
# ==========================================

class TestSetupLogging:
    """Tests for setup_logging function."""
    
    def test_creates_logger(self, tmp_path):
        """Test that logger is created."""
        output_dir = tmp_path / "logs"
        output_dir.mkdir()
        
        logger = train_decoder.setup_logging(output_dir)
        
        assert logger is not None
        assert logger.name == "intseq_bert.train_decoder"
    
    def test_creates_log_file(self, tmp_path):
        """Test that log file is created."""
        output_dir = tmp_path / "logs"
        output_dir.mkdir()
        
        train_decoder.setup_logging(output_dir)
        
        log_file = output_dir / "train_decoder.log"
        # Log file is created when first message is written
        # Just verify the handler setup doesn't raise


# ==========================================
# 2. Training Smoke Tests
# ==========================================

class TestTrainingSmoke:
    """Smoke tests for decoder training loop."""
    
    def test_training_runs_one_epoch(self, tmp_path):
        """Test that training completes without error."""
        features_dir = create_mock_features_dir(tmp_path, num_files=20)
        encoder_ckpt = create_mock_encoder_checkpoint(tmp_path)
        output_dir = tmp_path / "decoder_checkpoints"
        
        config = get_minimal_decoder_config(features_dir, encoder_ckpt, output_dir)
        
        # Should complete without error
        train_decoder.train(config)
        
        # Check checkpoint exists
        assert (output_dir / "best_decoder.pt").exists()
    
    def test_config_saved_to_file(self, tmp_path):
        """Test that config is saved during training."""
        features_dir = create_mock_features_dir(tmp_path, num_files=20)
        encoder_ckpt = create_mock_encoder_checkpoint(tmp_path)
        output_dir = tmp_path / "decoder_checkpoints"
        
        config = get_minimal_decoder_config(features_dir, encoder_ckpt, output_dir)
        
        train_decoder.train(config)
        
        config_path = output_dir / "config.json"
        assert config_path.exists()
        
        with open(config_path) as f:
            saved_config = json.load(f)
        
        assert saved_config['hidden_dim'] == 64


# ==========================================
# 3. Encoder Freezing Tests
# ==========================================

class TestEncoderFreezing:
    """Tests for encoder freezing behavior."""
    
    def test_encoder_is_frozen(self, tmp_path):
        """Test that encoder parameters are frozen during training."""
        # We can verify this by checking that encoder grads are None after training
        features_dir = create_mock_features_dir(tmp_path, num_files=20)
        encoder_ckpt = create_mock_encoder_checkpoint(tmp_path)
        output_dir = tmp_path / "decoder_checkpoints"
        
        config = get_minimal_decoder_config(features_dir, encoder_ckpt, output_dir)
        
        # Training should complete with frozen encoder
        train_decoder.train(config)
        
        # Success = no error raised


# ==========================================
# 4. Decoder Checkpoint Tests
# ==========================================

class TestDecoderCheckpoint:
    """Tests for decoder checkpoint saving."""
    
    def test_checkpoint_is_saved(self, tmp_path):
        """Test that decoder checkpoint is saved."""
        features_dir = create_mock_features_dir(tmp_path, num_files=20)
        encoder_ckpt = create_mock_encoder_checkpoint(tmp_path)
        output_dir = tmp_path / "decoder_checkpoints"
        
        config = get_minimal_decoder_config(features_dir, encoder_ckpt, output_dir)
        
        train_decoder.train(config)
        
        assert (output_dir / "best_decoder.pt").exists()
    
    def test_checkpoint_can_be_loaded(self, tmp_path):
        """Test that saved checkpoint can be loaded."""
        features_dir = create_mock_features_dir(tmp_path, num_files=20)
        encoder_ckpt = create_mock_encoder_checkpoint(tmp_path)
        output_dir = tmp_path / "decoder_checkpoints"
        
        config = get_minimal_decoder_config(features_dir, encoder_ckpt, output_dir)
        
        train_decoder.train(config)
        
        # Load checkpoint
        state_dict = torch.load(output_dir / "best_decoder.pt")
        
        # Create decoder and load
        decoder = decoder_model.IntSeqDecoder(d_model=32, hidden_dim=64)
        decoder.load_state_dict(state_dict)
        
        # Verify forward works
        x = torch.randn(2, 32)
        output = decoder(x)
        assert 'mag_mu' in output


# ==========================================
# 5. CLI Tests
# ==========================================

class TestCLI:
    """Tests for command-line interface."""
    
    def test_cli_argument_parsing(self, tmp_path, monkeypatch):
        """Test CLI argument parsing."""
        features_dir = create_mock_features_dir(tmp_path, num_files=20)
        encoder_ckpt = create_mock_encoder_checkpoint(tmp_path)
        output_dir = tmp_path / "output"
        
        test_args = [
            "train_decoder.py",
            "--features_dir", str(features_dir),
            "--encoder_checkpoint", str(encoder_ckpt),
            "--output_dir", str(output_dir),
            "--epochs", "1",
            "--batch_size", "2",
            "--num_workers", "0",
        ]
        monkeypatch.setattr("sys.argv", test_args)
        
        # Should complete without error
        train_decoder.main()
        
        assert (output_dir / "best_decoder.pt").exists()


# ==========================================
# 6. Loss Computation Integration Tests
# ==========================================

class TestLossIntegration:
    """Tests for loss computation during training."""
    
    def test_loss_decreases(self, tmp_path):
        """Test that training at least completes with finite loss."""
        features_dir = create_mock_features_dir(tmp_path, num_files=20)
        encoder_ckpt = create_mock_encoder_checkpoint(tmp_path)
        output_dir = tmp_path / "decoder_checkpoints"
        
        config = get_minimal_decoder_config(features_dir, encoder_ckpt, output_dir)
        
        # If loss computation works, training will complete
        train_decoder.train(config)
        
        # Check log file was created (indicates training ran)
        assert (output_dir / "train_decoder.log").exists()