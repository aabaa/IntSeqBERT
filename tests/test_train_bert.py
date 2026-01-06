"""
Tests for IntSeqBERT training script (Dual Stream Architecture).
Tests LR scheduler, training loop, and CLI.
"""

import pytest
import torch
import argparse
import json
from pathlib import Path

from intseq_bert import train_bert, bert_model


# ==========================================
# Helper Functions
# ==========================================

def create_mock_features_dir(tmp_path: Path, num_files: int = 10):
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


def get_minimal_training_config(features_dir: Path, output_dir: Path) -> dict:
    """Get minimal training configuration for tests."""
    return {
        'features_dir': str(features_dir),
        'output_dir': str(output_dir),
        'epochs': 1,
        'batch_size': 2,
        'lr': 1e-4,
        'd_model': 32,
        'nhead': 2,
        'num_layers': 1,
        'dim_feedforward': 64,
        'mag_dim': 5,
        'mod_dim': 200,
        'num_workers': 0,
        'val_ratio': 0.2,
        'test_ratio': 0.2,
    }


# ==========================================
# 1. LR Scheduler Tests
# ==========================================

class TestLRScheduler:
    """Tests for get_cosine_schedule_with_warmup function."""
    
    def test_scheduler_creation(self):
        """Test scheduler can be created."""
        model = bert_model.IntSeqBERT(d_model=64, num_layers=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        scheduler = train_bert.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=1000
        )
        
        assert scheduler is not None
    
    def test_warmup_phase(self):
        """Test LR increases during warmup."""
        model = bert_model.IntSeqBERT(d_model=64, num_layers=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        scheduler = train_bert.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,
            num_training_steps=100
        )
        
        # LR should increase during warmup
        lrs = []
        for step in range(10):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()
        
        # Each subsequent LR should be >= previous during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i-1]
    
    def test_decay_phase(self):
        """Test LR decreases after warmup (cosine decay)."""
        model = bert_model.IntSeqBERT(d_model=64, num_layers=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        scheduler = train_bert.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,
            num_training_steps=100
        )
        
        # Skip warmup
        for _ in range(10):
            optimizer.step()
            scheduler.step()
        
        # LR should decrease during decay
        peak_lr = scheduler.get_last_lr()[0]
        
        for _ in range(50):
            optimizer.step()
            scheduler.step()
        
        mid_lr = scheduler.get_last_lr()[0]
        assert mid_lr < peak_lr


# ==========================================
# 2. Training Smoke Tests
# ==========================================

class TestTrainingSmoke:
    """Smoke tests for training loop."""
    
    def test_training_runs_one_epoch(self, tmp_path):
        """Test that training completes without error."""
        features_dir = create_mock_features_dir(tmp_path, num_files=20)
        config = get_minimal_training_config(features_dir, tmp_path / "checkpoints")
        
        # Should complete without error
        train_bert.train(config)
        
        # Check checkpoints exist
        assert (tmp_path / "checkpoints" / "best_model.pt").exists()
        assert (tmp_path / "checkpoints" / "last_model.pt").exists()
    
    def test_checkpoint_structure(self, tmp_path):
        """Test that checkpoint has correct structure."""
        features_dir = create_mock_features_dir(tmp_path, num_files=20)
        config = get_minimal_training_config(features_dir, tmp_path / "checkpoints")
        
        train_bert.train(config)
        
        checkpoint = torch.load(tmp_path / "checkpoints" / "best_model.pt")
        
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'epoch' in checkpoint
        assert 'config' in checkpoint
    
    def test_config_saved_to_file(self, tmp_path):
        """Test that config is saved during training."""
        features_dir = create_mock_features_dir(tmp_path, num_files=20)
        config = get_minimal_training_config(features_dir, tmp_path / "checkpoints")
        
        train_bert.train(config)
        
        # Check config was saved
        config_path = tmp_path / "checkpoints" / "config.json"
        assert config_path.exists()
        
        with open(config_path) as f:
            saved_config = json.load(f)
        
        assert saved_config['d_model'] == 32


# ==========================================
# 3. CLI Tests
# ==========================================

class TestCLI:
    """Tests for command-line interface."""
    
    def test_cli_argument_parsing(self, tmp_path, monkeypatch):
        """Test CLI argument parsing."""
        features_dir = create_mock_features_dir(tmp_path, num_files=20)
        
        test_args = [
            "train_bert.py",
            "--features_dir", str(features_dir),
            "--output_dir", str(tmp_path / "output"),
            "--epochs", "1",
            "--batch_size", "2",
            "--d_model", "32",
            "--nhead", "2",
            "--num_layers", "1",
            "--num_workers", "0",
        ]
        monkeypatch.setattr("sys.argv", test_args)
        
        # Should complete without error
        train_bert.main()
        
        assert (tmp_path / "output" / "best_model.pt").exists()


# ==========================================
# 4. Model Loading Tests
# ==========================================

class TestModelLoading:
    """Tests for loading trained models."""
    
    def test_load_trained_model(self, tmp_path):
        """Test loading a trained checkpoint."""
        features_dir = create_mock_features_dir(tmp_path, num_files=20)
        config = get_minimal_training_config(features_dir, tmp_path / "checkpoints")
        
        train_bert.train(config)
        
        # Load the model
        model, checkpoint = bert_model.IntSeqBERT.load_from_checkpoint(
            str(tmp_path / "checkpoints" / "best_model.pt"),
            device='cpu'
        )
        
        assert model.d_model == 32
        assert checkpoint['epoch'] == 1
