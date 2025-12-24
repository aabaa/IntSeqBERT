"""
Integration tests for training script.
"""

import pytest
import torch
from pathlib import Path

from intseq_bert.train import train


def test_training_smoke(tmp_path):
    """
    Smoke test: verify training completes for 1 epoch with minimal data.
    
    This is an integration test that ensures all components work together:
    - Data loading
    - Collator
    - Model
    - Training loop
    - Checkpointing
    """
    # Create dummy features.pt with 4 sequences
    features = {
        f"A{i:06d}": torch.randn(10, 27, dtype=torch.float32)
        for i in range(4)
    }
    features_path = tmp_path / "features.pt"
    torch.save(features, features_path)
    
    # Create minimal config
    config = {
        # Paths
        "features_path": str(features_path),
        "metadata_path": None,
        "output_dir": str(tmp_path / "checkpoints"),
        
        # Training (minimal for speed)
        "epochs": 1,
        "batch_size": 2,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "warmup_steps": 2,
        "max_grad_norm": 1.0,
        "log_interval": 10,
        
        # Model (small for speed)
        "input_dim": 27,
        "d_model": 32,  # Small
        "nhead": 2,  # Small
        "num_layers": 1,  # Minimal
        "dim_feedforward": 64,  # Small
        "dropout": 0.1,
        
        # Data
        "val_ratio": 0.25,  # 1 sequence
        "test_ratio": 0.25,  # 1 sequence
        "mask_prob": 0.15,
        "min_len": 5,
        "seed": 42
    }
    
    # Run training (should complete without errors)
    train(config)
    
    # Verify outputs exist
    output_dir = tmp_path / "checkpoints"
    assert output_dir.exists()
    
    # Check checkpoint files
    best_model_path = output_dir / "best_model.pt"
    last_model_path = output_dir / "last_model.pt"
    config_path = output_dir / "config.json"
    log_path = output_dir / "train.log"
    
    assert best_model_path.exists(), "best_model.pt should exist"
    assert last_model_path.exists(), "last_model.pt should exist"
    assert config_path.exists(), "config.json should exist"
    assert log_path.exists(), "train.log should exist"
    
    # Verify checkpoint can be loaded
    checkpoint = torch.load(best_model_path)
    assert "epoch" in checkpoint
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "scheduler_state_dict" in checkpoint
    assert "train_loss" in checkpoint
    assert "val_loss" in checkpoint
    
    # Verify epoch is correct
    assert checkpoint["epoch"] == 1


def test_training_with_validation_improvement(tmp_path):
    """Test that best model is saved when validation improves."""
    # Create dummy features
    features = {
        f"A{i:06d}": torch.randn(15, 27, dtype=torch.float32)
        for i in range(8)
    }
    features_path = tmp_path / "features.pt"
    torch.save(features, features_path)
    
    config = {
        "features_path": str(features_path),
        "output_dir": str(tmp_path / "checkpoints"),
        "epochs": 2,  # Run 2 epochs
        "batch_size": 2,
        "lr": 1e-3,
        "d_model": 32,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 64,
        "val_ratio": 0.25,
        "test_ratio": 0.25,
        "seed": 42
    }
    
    train(config)
    
    # Both checkpoints should exist
    best_path = tmp_path / "checkpoints" / "best_model.pt"
    last_path = tmp_path / "checkpoints" / "last_model.pt"
    
    assert best_path.exists()
    assert last_path.exists()
    
    # Last checkpoint should be from epoch 2
    last_checkpoint = torch.load(last_path)
    assert last_checkpoint["epoch"] == 2


def test_training_device_handling(tmp_path):
    """Test that training correctly handles device selection."""
    # Create minimal dataset
    features = {
        f"A{i:06d}": torch.randn(10, 27, dtype=torch.float32)
        for i in range(4)
    }
    features_path = tmp_path / "features.pt"
    torch.save(features, features_path)
    
    config = {
        "features_path": str(features_path),
        "output_dir": str(tmp_path / "checkpoints"),
        "epochs": 1,
        "batch_size": 2,
        "lr": 1e-3,
        "d_model": 16,
        "nhead": 2,
        "num_layers": 1,
        "val_ratio": 0.25,
        "test_ratio": 0.25,
        "seed": 42
    }
    
    # Should complete regardless of device (CPU/CUDA/MPS)
    train(config)
    
    # Verify checkpoint was saved
    assert (tmp_path / "checkpoints" / "best_model.pt").exists()
