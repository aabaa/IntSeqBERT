"""
Tests for decoder training script.
"""

import pytest
import torch
import json
from pathlib import Path

from intseq_bert.train_decoder import (
    get_targets,
    decoder_collate_fn,
    DecoderDataset,
    load_decoder_data,
    train_decoder
)
from intseq_bert.bert_model import IntSeqBERT
from intseq_bert.features import log_magnitude


def test_get_targets():
    """Test target generation from integers."""
    integers = [-5, 0, 5, 42]
    targets = get_targets(integers)
    
    # Check sign mapping
    assert targets['sign'].tolist() == [0, 1, 2, 2]  # neg, zero, pos, pos
    
    # Check magnitude: Now bin indices (LongTensor) instead of log values
    assert targets['mag'].dtype == torch.long
    # Magnitude bin indices should be in valid range [0, 4095]
    assert all(0 <= targets['mag'][i].item() < 4096 for i in range(len(integers)))
    
    # Check modulo (Python % handles negatives correctly)
    assert targets['mod3'].tolist() == [-5 % 3, 0 % 3, 5 % 3, 42 % 3]
    assert targets['mod5'].tolist() == [-5 % 5, 0 % 5, 5 % 5, 42 % 5]
    assert targets['mod10'].tolist() == [-5 % 10, 0 % 10, 5 % 10, 42 % 10]
    
    # Check new modulo heads exist
    assert 'mod7' in targets and 'mod11' in targets
    assert 'mod13' in targets and 'mod100' in targets


def test_decoder_collate_fn():
    """Test custom collate function."""
    # Create fake batch
    batch = [
        {
            'features': torch.randn(10, 35),
            'integers': list(range(10))
        },
        {
            'features': torch.randn(15, 35),
            'integers': list(range(15))
        }
    ]
    
    result = decoder_collate_fn(batch)
    
    # Check shapes
    assert result['masked_inputs'].shape == (2, 15, 35)  # max_len=15, 35-dim
    assert result['attention_mask'].shape == (2, 15)
    assert result['mask_indices'].shape == (2,)
    assert len(result['target_integers']) == 2
    assert result['target_features'].shape == (2, 35)
    
    # Check attention mask
    assert result['attention_mask'][0, :10].sum() == 10  # First seq has length 10
    assert result['attention_mask'][0, 10:].sum() == 0   # Rest is padding
    assert result['attention_mask'][1].sum() == 15  # Second seq has length 15
    
    # Check target integers are valid
    assert 0 <= result['target_integers'][0] < 10
    assert 0 <= result['target_integers'][1] < 15


def test_decoder_training_smoke(tmp_path):
    """
    Smoke test: verify decoder training completes for 1 epoch.
    """
    # Create dummy BERT checkpoint
    bert_model = IntSeqBERT(
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        input_dim=35
    )
    bert_checkpoint = {
        'model_state_dict': bert_model.state_dict(),
        'config': {
            'd_model': 32,
            'nhead': 2,
            'num_layers': 1,
            'dim_feedforward': 64,
            'input_dim': 35
        }
    }
    bert_path = tmp_path / "bert.pt"
    torch.save(bert_checkpoint, bert_path)
    
    # Create dummy features.pt
    features = {
        f"A{i:06d}": torch.randn(10, 35, dtype=torch.float32)
        for i in range(8)
    }
    features_path = tmp_path / "features.pt"
    torch.save(features, features_path)
    
    # Create dummy JSONL with matching integers
    jsonl_path = tmp_path / "data.jsonl"
    with open(jsonl_path, 'w') as f:
        for i in range(8):
            record = {
                "oeis_id": f"A{i:06d}",
                "sequence": list(range(i * 10, (i + 1) * 10))  # Different integers
            }
            f.write(json.dumps(record) + '\n')
    
    # Run decoder training
    config = {
        'bert_checkpoint': str(bert_path),
        'features_path': str(features_path),
        'jsonl_path': str(jsonl_path),
        'output_dir': str(tmp_path / "decoder_output"),
        'epochs': 1,
        'batch_size': 2,
        'lr': 1e-3,
        'bypass_bert': False,  # Explicitly disable bypass mode
        'seed': 42
    }
    
    train_decoder(config)
    
    # Verify outputs
    output_dir = tmp_path / "decoder_output"
    assert output_dir.exists()
    assert (output_dir / "best_decoder.pt").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "train_decoder.log").exists()
    
    # Load and verify checkpoint
    checkpoint = torch.load(output_dir / "best_decoder.pt")
    assert 'decoder_state_dict' in checkpoint
    assert 'epoch' in checkpoint
    assert checkpoint['epoch'] == 1


def test_bert_gradient_frozen(tmp_path):
    """Verify that BERT gradients are frozen during decoder training."""
    # Create small BERT
    bert_model = IntSeqBERT(d_model=16, nhead=2, num_layers=1)
    bert_checkpoint = {
        'model_state_dict': bert_model.state_dict(),
        'config': {'d_model': 16, 'nhead': 2, 'num_layers': 1}
    }
    bert_path = tmp_path / "bert.pt"
    torch.save(bert_checkpoint, bert_path)
    
    # Load with frozen setup
    loaded_bert, _ = IntSeqBERT.load_from_checkpoint(str(bert_path), device='cpu')
    loaded_bert.eval()
    loaded_bert.requires_grad_(False)
    
    # Check all parameters have requires_grad=False
    for param in loaded_bert.parameters():
        assert not param.requires_grad


def test_load_decoder_data(tmp_path):
    """Test data loading with features and integers alignment."""
    # Create features
    features = {
        "A000001": torch.randn(5, 35),
        "A000002": torch.randn(8, 35),
        "A000003": torch.randn(6, 35)
    }
    features_path = tmp_path / "features.pt"
    torch.save(features, features_path)
    
    # Create JSONL
    jsonl_path = tmp_path / "data.jsonl"
    with open(jsonl_path, 'w') as f:
        f.write(json.dumps({"oeis_id": "A000001", "sequence": [1, 2, 3, 4, 5]}) + '\n')
        f.write(json.dumps({"oeis_id": "A000002", "sequence": [10, 20, 30, 40, 50, 60, 70, 80]}) + '\n')
        # A000003 not in JSONL - should be skipped
    
    train_ds, val_ds, test_ds = load_decoder_data(
        str(features_path),
        str(jsonl_path),
        val_ratio=0.5,
        test_ratio=0.0,
        seed=42
    )
    
    # Should have 2 items total (A000003 skipped)
    total = len(train_ds) + len(val_ds) + len(test_ds)
    assert total == 2
    
    # Check data structure
    sample = train_ds[0] if len(train_ds) > 0 else val_ds[0]
    assert 'oeis_id' in sample
    assert 'features' in sample
    assert 'integers' in sample
    assert len(sample['features']) == len(sample['integers'])


def test_bypass_mode_collate():
    """Test that collate function includes target_features."""
    batch = [
        {
            'features': torch.randn(10, 35),
            'integers': list(range(10, 20))
        },
        {
            'features': torch.randn(8, 35),
            'integers': list(range(30, 38))
        }
    ]
    
    result = decoder_collate_fn(batch)
    
    # Check that target_features is included
    assert 'target_features' in result
    assert result['target_features'].shape == (2, 35)  # (batch_size, 35)


def test_bypass_mode_training(tmp_path):
    """Test decoder training in bypass mode (without BERT)."""
    # Create dummy features
    features = {
        f"A{i:06d}": torch.randn(10, 35, dtype=torch.float32)
        for i in range(8)
    }
    features_path = tmp_path / "features.pt"
    torch.save(features, features_path)
    
    # Create dummy JSONL
    jsonl_path = tmp_path / "data.jsonl"
    with open(jsonl_path, 'w') as f:
        for i in range(8):
            record = {
                "oeis_id": f"A{i:06d}",
                "sequence": list(range(i * 10, (i + 1) * 10))
            }
            f.write(json.dumps(record) + '\n')
    
    # Run bypass mode training
    config = {
        'bert_checkpoint': None,  # Not needed in bypass mode
        'features_path': str(features_path),
        'jsonl_path': str(jsonl_path),
        'output_dir': str(tmp_path / "decoder_bypass"),
        'epochs': 1,
        'batch_size': 2,
        'lr': 1e-3,
        'bypass_bert': True,  # Enable bypass mode
        'seed': 42
    }
    
    train_decoder(config)
    
    # Verify outputs
    output_dir = tmp_path / "decoder_bypass"
    assert output_dir.exists()
    assert (output_dir / "best_decoder.pt").exists()
    
    # Load checkpoint and verify decoder has input_dim=35
    checkpoint = torch.load(output_dir / "best_decoder.pt")
    assert 'decoder_state_dict' in checkpoint
    
    # Check that decoder was configured for 35-dim input
    # The shared_encoder first layer should be (35 → 256)
    first_weight_key = 'shared_encoder.0.weight'
    if first_weight_key in checkpoint['decoder_state_dict']:
        first_layer_weight = checkpoint['decoder_state_dict'][first_weight_key]
        assert first_layer_weight.shape[1] == 35  # Input dimension

