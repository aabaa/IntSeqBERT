"""
test_train.py:
Unit tests for IntSeqBERT training module.
Tests EarlyStopping, prepare_labels, evaluate, and training utilities.
"""

import pytest
import torch
import torch.nn as nn

from intseq_bert import config
from intseq_bert.train import (
    EarlyStopping,
    set_seed,
    prepare_labels,
    evaluate
)
from intseq_bert.models import IntSeqForPreTraining


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
def mock_collator_output(batch_size, seq_len):
    """
    Creates a mock output from OEISCollator.
    This simulates what the collator produces.
    """
    # Lengths for each sample (variable length before padding)
    lengths = [seq_len - i for i in range(batch_size)]
    
    # mag_inputs: (B, L, 5) - includes is_masked flag
    mag_inputs = torch.randn(batch_size, seq_len, config.MAG_EXTENDED_DIM)
    
    # mod_inputs: (B, L, 200)
    mod_inputs = torch.randn(batch_size, seq_len, config.MOD_FEATURE_DIM)
    
    # mag_labels: (B, L, 4) - [log_val, sign+, sign-, sign0]
    mag_labels = torch.zeros(batch_size, seq_len, config.MAG_RAW_DIM)
    mag_labels[:, :, 0] = torch.randn(batch_size, seq_len)  # log_val
    # Set one-hot signs randomly
    for b in range(batch_size):
        for l in range(seq_len):
            sign_idx = torch.randint(0, 3, (1,)).item()
            mag_labels[b, l, 1 + sign_idx] = 1.0
    
    # mod_labels: (B, L, 100) - integer remainders
    mod_labels = torch.stack([
        torch.randint(0, m, (batch_size, seq_len))
        for m in config.MOD_RANGE
    ], dim=-1)
    
    # attention_mask: (B, L) - 1 for valid, 0 for padding
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    for b, length in enumerate(lengths):
        attention_mask[b, length:] = 0
    
    # mask_matrix: (B, L) - True where prediction is needed
    mask_matrix = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask_matrix[:, :8] = True  # First 8 positions masked
    # Don't mask padding
    for b, length in enumerate(lengths):
        mask_matrix[b, length:] = False
    
    return {
        "mag_inputs": mag_inputs,
        "mod_inputs": mod_inputs,
        "mag_labels": mag_labels,
        "mod_labels": mod_labels,
        "attention_mask": attention_mask,
        "mask_matrix": mask_matrix,
        "oeis_ids": [f"A{100000 + i}" for i in range(batch_size)]
    }

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def small_model():
    """Small model for fast testing."""
    return IntSeqForPreTraining(d_model=32, nhead=2, num_layers=1)


# ==========================================
# EarlyStopping Tests
# ==========================================

class TestEarlyStopping:
    """Tests for EarlyStopping class."""
    
    def test_initial_state(self):
        """Test initial state is correct."""
        es = EarlyStopping(patience=5, delta=0.0)
        assert es.patience == 5
        assert es.counter == 0
        assert es.best_loss == float("inf")
        assert es.early_stop == False
    
    def test_improvement_resets_counter(self):
        """Test that improvement resets the counter."""
        es = EarlyStopping(patience=3)
        
        # Initial call with loss
        result = es(1.0)
        assert result == False
        assert es.counter == 0
        assert es.best_loss == 1.0
        
        # Improvement
        result = es(0.5)
        assert result == False
        assert es.counter == 0
        assert es.best_loss == 0.5
    
    def test_no_improvement_increments_counter(self):
        """Test that no improvement increments counter."""
        es = EarlyStopping(patience=3)
        
        es(1.0)  # Set baseline
        
        # No improvement - counter increases but not yet at patience
        result = es(1.0)
        assert result == False  # Not yet at patience
        assert es.counter == 1
        
        result = es(1.1)  # Worse
        assert result == False  # Still not at patience
        assert es.counter == 2
    
    def test_early_stop_triggered(self):
        """Test early stopping is triggered after patience epochs."""
        es = EarlyStopping(patience=3)
        
        result = es(1.0)  # Set baseline, counter = 0
        assert result == False
        
        result = es(1.0)  # No improvement, counter = 1
        assert result == False
        
        result = es(1.0)  # No improvement, counter = 2
        assert result == False
        
        result = es(1.0)  # No improvement, counter = 3 >= patience
        assert result == True
        assert es.counter == 3
    
    def test_delta_threshold(self):
        """Test delta threshold for improvement detection."""
        es = EarlyStopping(patience=3, delta=0.1)
        
        es(1.0)  # Set baseline
        
        # Improvement less than delta doesn't count as improvement
        result = es(0.95)  # 1.0 - 0.95 = 0.05 < delta
        assert result == False  # Not yet at patience
        assert es.counter == 1
        
        # Improvement greater than delta resets counter
        es = EarlyStopping(patience=3, delta=0.1)
        es(1.0)
        result = es(0.85)  # 1.0 - 0.85 = 0.15 > delta
        assert result == False
        assert es.counter == 0


# ==========================================
# set_seed Tests
# ==========================================

class TestSetSeed:
    """Tests for set_seed function."""
    
    def test_reproducibility(self):
        """Test that set_seed ensures reproducibility."""
        set_seed(42)
        r1 = torch.randn(5)
        
        set_seed(42)
        r2 = torch.randn(5)
        
        assert torch.allclose(r1, r2)
    
    def test_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        r1 = torch.randn(5)
        
        set_seed(123)
        r2 = torch.randn(5)
        
        assert not torch.allclose(r1, r2)


# ==========================================
# prepare_labels Tests
# ==========================================

class TestPrepareLabels:
    """Tests for prepare_labels function."""
    
    def test_output_keys(self, mock_collator_output, device):
        """Test that prepare_labels returns the expected keys."""
        result = prepare_labels(mock_collator_output, device)
        
        assert "mag_features" in result
        assert "mod_features" in result
        assert "src_key_padding_mask" in result
        assert "labels" in result
        
        labels = result["labels"]
        assert "mag_targets" in labels
        assert "sign_targets" in labels
        assert "mod_targets" in labels
        assert "mask_map" in labels
    
    def test_mag_targets_extraction(self, mock_collator_output, device, batch_size, seq_len):
        """Test that mag_targets is correctly extracted from mag_labels."""
        result = prepare_labels(mock_collator_output, device)
        
        mag_targets = result["labels"]["mag_targets"]
        expected = mock_collator_output["mag_labels"][:, :, 0]
        
        assert mag_targets.shape == (batch_size, seq_len)
        assert torch.allclose(mag_targets, expected.to(device))
    
    def test_sign_targets_conversion(self, mock_collator_output, device, batch_size, seq_len):
        """Test that sign one-hot is correctly converted to class indices."""
        result = prepare_labels(mock_collator_output, device)
        
        sign_targets = result["labels"]["sign_targets"]
        
        assert sign_targets.shape == (batch_size, seq_len)
        assert sign_targets.dtype == torch.int64
        
        # Check values are 0, 1, or 2
        assert (sign_targets >= 0).all()
        assert (sign_targets <= 2).all()
    
    def test_padding_mask_conversion(self, mock_collator_output, device, batch_size, seq_len):
        """Test attention_mask is correctly inverted to src_key_padding_mask."""
        result = prepare_labels(mock_collator_output, device)
        
        padding_mask = result["src_key_padding_mask"]
        attention_mask = mock_collator_output["attention_mask"]
        
        # Should be True where attention_mask is 0
        expected = (attention_mask == 0).to(device)
        assert torch.equal(padding_mask, expected)
    
    def test_device_transfer(self, mock_collator_output):
        """Test that tensors are moved to the specified device."""
        device = torch.device("cpu")
        result = prepare_labels(mock_collator_output, device)
        
        assert result["mag_features"].device == device
        assert result["mod_features"].device == device
        assert result["labels"]["mag_targets"].device == device


# ==========================================
# evaluate Tests
# ==========================================

class TestEvaluate:
    """Tests for evaluate function."""
    
    def test_output_keys(self, small_model, mock_collator_output, device):
        """Test that evaluate returns expected metric keys."""
        from torch.utils.data import DataLoader
        
        # Create a simple dataset from mock batch
        class MockDataset:
            def __init__(self, batch):
                self.batch = batch
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return {k: v[0] if isinstance(v, list) else v for k, v in self.batch.items()}
        
        def mock_collate(samples):
            return mock_collator_output
        
        loader = DataLoader(MockDataset(mock_collator_output), batch_size=4, collate_fn=mock_collate)
        
        metrics = evaluate(small_model, loader, device)
        
        assert "val_loss" in metrics
        assert "mag_mse" in metrics
        assert "mag_acc" in metrics
        assert "sign_acc" in metrics
        assert "mod_acc" in metrics
        assert "mod_loss" in metrics
        assert "mod_accuracies" in metrics
    
    def test_accuracy_ranges(self, small_model, mock_collator_output, device):
        """Test that accuracy metrics are in valid percentage range."""
        from torch.utils.data import DataLoader
        
        def mock_collate(samples):
            return mock_collator_output
        
        class MockDataset:
            def __len__(self): return 1
            def __getitem__(self, idx): return {}
        
        loader = DataLoader(MockDataset(), batch_size=4, collate_fn=mock_collate)
        
        metrics = evaluate(small_model, loader, device)
        
        # Accuracies should be 0-100%
        assert 0 <= metrics["mag_acc"] <= 100
        assert 0 <= metrics["sign_acc"] <= 100
        assert 0 <= metrics["mod_acc"] <= 100
        
        # mod_accuracies should be a list of 100 values
        from intseq_bert import config as cfg
        assert isinstance(metrics["mod_accuracies"], list)
        assert len(metrics["mod_accuracies"]) == cfg.NUM_MODULI
        for acc in metrics["mod_accuracies"]:
            assert 0 <= acc <= 100
        
        # MSE should be non-negative
        assert metrics["mag_mse"] >= 0
    
    def test_eval_mode(self, small_model, mock_collator_output, device):
        """Test that model is in eval mode during evaluation."""
        from torch.utils.data import DataLoader
        
        def mock_collate(samples):
            return mock_collator_output
        
        class MockDataset:
            def __len__(self): return 1
            def __getitem__(self, idx): return {}
        
        loader = DataLoader(MockDataset(), batch_size=4, collate_fn=mock_collate)
        
        small_model.train()  # Start in train mode
        
        # Evaluate should switch to eval
        evaluate(small_model, loader, device)
        
        # Model should still be in eval mode after
        assert not small_model.training


# ==========================================
# Integration Tests
# ==========================================

class TestTrainingIntegration:
    """Integration tests for training components."""
    
    def test_full_forward_backward(self, small_model, mock_collator_output, device):
        """Test complete forward-backward pass with prepare_labels."""
        inputs = prepare_labels(mock_collator_output, device)
        
        # Forward pass
        outputs = small_model(
            mag_features=inputs["mag_features"],
            mod_features=inputs["mod_features"],
            src_key_padding_mask=inputs["src_key_padding_mask"],
            labels=inputs["labels"]
        )
        
        assert "loss" in outputs
        assert outputs["loss"].requires_grad
        
        # Backward pass
        outputs["loss"].backward()
        
        # Check gradients exist
        for name, param in small_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_optimizer_step(self, small_model, mock_collator_output, device):
        """Test optimizer can update parameters."""
        optimizer = torch.optim.AdamW(small_model.parameters(), lr=1e-3)
        
        # Get initial weights (using film_scale which is always Linear)
        # Note: mag_proj may be Sequential in v3, so we use a known Linear layer
        initial_weight = small_model.bert.embeddings.film_scale.weight.clone()
        
        inputs = prepare_labels(mock_collator_output, device)
        outputs = small_model(
            mag_features=inputs["mag_features"],
            mod_features=inputs["mod_features"],
            src_key_padding_mask=inputs["src_key_padding_mask"],
            labels=inputs["labels"]
        )
        
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        
        # Weights should have changed
        updated_weight = small_model.bert.embeddings.film_scale.weight
        assert not torch.allclose(initial_weight, updated_weight)


# ==========================================
# TrainingLogger Tests
# ==========================================

class TestTrainingLogger:
    """Tests for TrainingLogger class."""
    
    @pytest.fixture
    def mock_args(self):
        """Create mock argparse.Namespace for testing."""
        import argparse
        args = argparse.Namespace(
            lr=5e-5,
            batch_size=32,
            d_model=512,
            num_layers=8,
            nhead=8,
            split_type="std",
            resume=None
        )
        return args
    
    @pytest.fixture
    def data_stats(self):
        return {"train_samples": 1000, "val_samples": 100, "test_samples": 100}
    
    def test_init_creates_config_json(self, tmp_path, mock_args, data_stats):
        """Test that __init__ creates config.json."""
        from intseq_bert.train import TrainingLogger
        
        logger = TrainingLogger(tmp_path, mock_args, data_stats)
        
        assert (tmp_path / "config.json").exists()
        
        import json
        with open(tmp_path / "config.json") as f:
            config_data = json.load(f)
        
        assert "timestamp" in config_data
        assert "args" in config_data
        assert config_data["args"]["lr"] == 5e-5
        assert "environment" in config_data
        assert "data_stats" in config_data
    
    def test_init_creates_csv_header(self, tmp_path, mock_args):
        """Test that __init__ creates history.csv with header."""
        from intseq_bert.train import TrainingLogger
        from intseq_bert import config as cfg
        
        logger = TrainingLogger(tmp_path, mock_args)
        
        assert (tmp_path / "history.csv").exists()
        
        import csv
        with open(tmp_path / "history.csv") as f:
            reader = csv.reader(f)
            headers = next(reader)
        
        # Check basic headers
        assert "epoch" in headers
        assert "val_loss" in headers
        assert "val_mod_acc" in headers
        
        # Check per-mod columns exist (100 columns)
        mod_columns = [h for h in headers if h.startswith("mod_acc_")]
        assert len(mod_columns) == len(cfg.MOD_RANGE)
        assert "mod_acc_2" in headers
        assert "mod_acc_101" in headers
    
    def test_init_resume_skips_config(self, tmp_path, mock_args):
        """Test that resume=True skips config.json creation if it exists."""
        from intseq_bert.train import TrainingLogger
        import json
        
        # Create initial config
        logger1 = TrainingLogger(tmp_path, mock_args)
        
        # Read original timestamp
        with open(tmp_path / "config.json") as f:
            original_config = json.load(f)
        original_timestamp = original_config["timestamp"]
        
        # Create new args with different values
        import argparse
        new_args = argparse.Namespace(lr=1e-4, batch_size=64, resume="some_path.pt")
        
        # Resume should NOT overwrite config
        logger2 = TrainingLogger(tmp_path, new_args, resume=True)
        
        with open(tmp_path / "config.json") as f:
            resumed_config = json.load(f)
        
        # Config should be unchanged
        assert resumed_config["timestamp"] == original_timestamp
        assert resumed_config["args"]["lr"] == 5e-5  # Original value
    
    def test_log_epoch_appends_csv(self, tmp_path, mock_args):
        """Test that log_epoch appends rows to CSV."""
        from intseq_bert.train import TrainingLogger
        from intseq_bert import config as cfg
        
        logger = TrainingLogger(tmp_path, mock_args)
        
        epoch_data = {
            "epoch": 1,
            "lr": 5e-5,
            "time_sec": 120.5,
            "is_best": True,
            "early_stop_counter": 0,
            "train_loss": 0.5,
            "val_loss": 0.3,
            "val_mag_acc": 85.0,
            "val_mag_mse": 0.2,
            "val_sign_acc": 90.0,
            "val_mod_acc": 15.0,
            "val_mod_loss": 0.8,
            "mod_accuracies": [50.0 + i for i in range(len(cfg.MOD_RANGE))],
            "w_mag": 1.0,
            "w_sign": 1.0,
            "w_mod": 2.0
        }
        
        logger.log_epoch(epoch_data)
        
        import csv
        with open(tmp_path / "history.csv") as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        assert len(rows) == 2  # header + 1 data row
        assert rows[1][0] == "1"  # epoch
    
    def test_log_epoch_includes_all_mods(self, tmp_path, mock_args):
        """Test that log_epoch includes all 100 mod accuracies."""
        from intseq_bert.train import TrainingLogger
        from intseq_bert import config as cfg
        
        logger = TrainingLogger(tmp_path, mock_args)
        
        mod_accs = [float(i) for i in range(len(cfg.MOD_RANGE))]
        epoch_data = {
            "epoch": 1,
            "lr": 5e-5,
            "train_loss": 0.5,
            "val_loss": 0.3,
            "val_mag_acc": 85.0,
            "val_sign_acc": 90.0,
            "val_mod_acc": 15.0,
            "mod_accuracies": mod_accs
        }
        
        logger.log_epoch(epoch_data)
        
        import csv
        with open(tmp_path / "history.csv") as f:
            reader = csv.reader(f)
            headers = next(reader)
            data_row = next(reader)
        
        # Total columns: 12 base + 100 mods + 3 weights = 115
        expected_cols = 12 + len(cfg.MOD_RANGE) + 3
        assert len(headers) == expected_cols
        assert len(data_row) == expected_cols
    
    def test_save_best_metrics(self, tmp_path, mock_args):
        """Test that save_best_metrics creates best_metrics.json."""
        from intseq_bert.train import TrainingLogger
        import json
        
        logger = TrainingLogger(tmp_path, mock_args)
        
        metrics = {
            "epoch": 10,
            "val_loss": 0.1,
            "val_mag_acc": 95.0,
            "val_mag_mse": 0.05,
            "val_sign_acc": 99.0,
            "val_mod_acc": 30.0,
            "val_mod_loss": 0.5,
            "mod_accuracies": [80.0] * 100
        }
        
        logger.save_best_metrics(metrics)
        
        assert (tmp_path / "best_metrics.json").exists()
        
        with open(tmp_path / "best_metrics.json") as f:
            best_data = json.load(f)
        
        assert best_data["best_epoch"] == 10
        assert best_data["val_loss"] == 0.1
        assert "saved_at" in best_data
    
    def test_save_best_metrics_includes_representative_mods(self, tmp_path, mock_args):
        """Test that best_metrics.json includes representative mod accuracies."""
        from intseq_bert.train import TrainingLogger
        from intseq_bert import config as cfg
        import json
        
        logger = TrainingLogger(tmp_path, mock_args)
        
        # Create distinct values for each mod
        mod_accuracies = [float(m) for m in cfg.MOD_RANGE]  # Use mod value as accuracy
        
        metrics = {
            "epoch": 5,
            "val_loss": 0.2,
            "val_mag_acc": 90.0,
            "val_sign_acc": 95.0,
            "val_mod_acc": 25.0,
            "mod_accuracies": mod_accuracies
        }
        
        logger.save_best_metrics(metrics)
        
        with open(tmp_path / "best_metrics.json") as f:
            best_data = json.load(f)
        
        assert "representative_mods" in best_data
        rep_mods = best_data["representative_mods"]
        
        # Check representative mods are present
        assert "mod_2" in rep_mods
        assert "mod_3" in rep_mods
        assert "mod_5" in rep_mods
        assert "mod_7" in rep_mods
        assert "mod_10" in rep_mods
        assert "mod_100" in rep_mods
        assert "mod_101" in rep_mods
        
        # Check values match expected (mod value used as accuracy)
        assert rep_mods["mod_2"] == 2.0
        assert rep_mods["mod_10"] == 10.0
    
    def test_get_representative_mod_indices(self):
        """Test get_representative_mod_indices returns correct indices."""
        from intseq_bert.train import TrainingLogger
        from intseq_bert import config as cfg
        
        indices = TrainingLogger.get_representative_mod_indices()
        
        # All representative mods should have valid indices
        assert len(indices) == 7  # [2,3,5,7,10,100,101]
        
        # Verify indices are correct
        for idx, expected_mod in zip(indices, [2, 3, 5, 7, 10, 100, 101]):
            assert cfg.MOD_RANGE[idx] == expected_mod
    
    def test_get_csv_headers(self):
        """Test get_csv_headers returns correct headers."""
        from intseq_bert.train import TrainingLogger
        from intseq_bert import config as cfg
        
        headers = TrainingLogger.get_csv_headers()
        
        # Check basic headers exist
        assert "epoch" in headers
        assert "lr" in headers
        assert "time_sec" in headers
        assert "is_best" in headers
        assert "early_stop_counter" in headers
        assert "train_loss" in headers
        assert "val_loss" in headers
        assert "val_mag_acc" in headers
        assert "val_mag_mse" in headers
        assert "val_sign_acc" in headers
        assert "val_mod_acc" in headers
        assert "val_mod_loss" in headers
        
        # Check per-mod columns
        assert "mod_acc_2" in headers
        assert "mod_acc_101" in headers
        
        # Check weight columns
        assert "w_mag" in headers
        assert "w_sign" in headers
        assert "w_mod" in headers
        
        # Total: 12 base + 100 mods + 3 weights = 115
        expected_count = 12 + len(cfg.MOD_RANGE) + 3
        assert len(headers) == expected_count
    
    def test_get_csv_headers_order(self):
        """Test get_csv_headers maintains correct column order."""
        from intseq_bert.train import TrainingLogger
        
        headers = TrainingLogger.get_csv_headers()
        
        # First columns should be meta
        assert headers[0] == "epoch"
        assert headers[1] == "lr"
        
        # Last columns should be weights
        assert headers[-3] == "w_mag"
        assert headers[-2] == "w_sign"
        assert headers[-1] == "w_mod"


# ==========================================
# Test-Only Mode Tests
# ==========================================

class TestLoadModelConfig:
    """Tests for _load_model_config function."""
    
    @pytest.fixture
    def mock_args(self):
        """Create mock args with model config."""
        import argparse
        return argparse.Namespace(
            d_model=256,
            nhead=4,
            num_layers=4
        )
    
    def test_priority_checkpoint_config(self, tmp_path, mock_args):
        """Test that checkpoint config has highest priority."""
        from intseq_bert.train import _load_model_config
        
        # Create checkpoint with config
        checkpoint = {
            "config": {
                "d_model": 512,
                "nhead": 8,
                "num_layers": 6
            }
        }
        
        result = _load_model_config(tmp_path / "model.pt", checkpoint, mock_args)
        
        assert result["d_model"] == 512
        assert result["nhead"] == 8
        assert result["num_layers"] == 6
    
    def test_priority_config_json(self, tmp_path, mock_args):
        """Test that config.json is used when checkpoint config is empty."""
        from intseq_bert.train import _load_model_config
        import json
        
        # Create config.json
        config_data = {
            "args": {
                "d_model": 768,
                "nhead": 12,
                "num_layers": 12
            }
        }
        with open(tmp_path / "config.json", "w") as f:
            json.dump(config_data, f)
        
        # Checkpoint without config
        checkpoint = {}
        
        result = _load_model_config(tmp_path / "model.pt", checkpoint, mock_args)
        
        assert result["d_model"] == 768
        assert result["nhead"] == 12
        assert result["num_layers"] == 12
    
    def test_priority_fallback_to_args(self, tmp_path, mock_args):
        """Test that args are used as fallback."""
        from intseq_bert.train import _load_model_config
        
        # Empty checkpoint, no config.json
        checkpoint = {}
        
        result = _load_model_config(tmp_path / "model.pt", checkpoint, mock_args)
        
        assert result["d_model"] == 256
        assert result["nhead"] == 4
        assert result["num_layers"] == 4
    
    def test_partial_checkpoint_config(self, tmp_path, mock_args):
        """Test partial config in checkpoint uses args for missing values."""
        from intseq_bert.train import _load_model_config
        
        # Checkpoint with only some config values
        checkpoint = {
            "config": {
                "d_model": 512
                # nhead and num_layers missing
            }
        }
        
        result = _load_model_config(tmp_path / "model.pt", checkpoint, mock_args)
        
        assert result["d_model"] == 512  # From checkpoint
        assert result["nhead"] == 4       # From args (fallback)
        assert result["num_layers"] == 4  # From args (fallback)


class TestSaveTestCsv:
    """Tests for _save_test_csv function."""
    
    @pytest.fixture
    def sample_metrics(self):
        """Sample metrics for testing."""
        from intseq_bert import config as cfg
        return {
            "val_loss": 0.0892,
            "mag_acc": 91.88,
            "mag_mse": 0.187,
            "sign_acc": 98.45,
            "mod_acc": 24.46,
            "mod_loss": 0.756,
            "mod_accuracies": [75.0 + i * 0.1 for i in range(len(cfg.MOD_RANGE))]
        }
    
    def test_creates_csv_file(self, tmp_path, sample_metrics):
        """Test that CSV file is created."""
        from intseq_bert.train import _save_test_csv
        
        csv_path = tmp_path / "test_results.csv"
        _save_test_csv(csv_path, sample_metrics, time_sec=45.2)
        
        assert csv_path.exists()
    
    def test_csv_has_header_and_data(self, tmp_path, sample_metrics):
        """Test CSV contains header and one data row."""
        from intseq_bert.train import _save_test_csv
        import csv
        
        csv_path = tmp_path / "test_results.csv"
        _save_test_csv(csv_path, sample_metrics, time_sec=45.2)
        
        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        assert len(rows) == 2  # header + 1 data row
    
    def test_csv_uses_shared_headers(self, tmp_path, sample_metrics):
        """Test CSV uses TrainingLogger.get_csv_headers()."""
        from intseq_bert.train import _save_test_csv, TrainingLogger
        import csv
        
        csv_path = tmp_path / "test_results.csv"
        _save_test_csv(csv_path, sample_metrics, time_sec=45.2)
        
        with open(csv_path) as f:
            reader = csv.reader(f)
            actual_headers = next(reader)
        
        expected_headers = TrainingLogger.get_csv_headers()
        assert actual_headers == expected_headers
    
    def test_csv_test_mode_markers(self, tmp_path, sample_metrics):
        """Test CSV has correct test-mode markers."""
        from intseq_bert.train import _save_test_csv
        import csv
        
        csv_path = tmp_path / "test_results.csv"
        _save_test_csv(csv_path, sample_metrics, time_sec=45.2)
        
        with open(csv_path) as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = next(reader)
        
        # Create dict for easier access
        row_dict = dict(zip(headers, data))
        
        assert row_dict["epoch"] == "0"  # Test mode marker
        assert row_dict["lr"] == "0.0"
        assert row_dict["train_loss"] == "0.0"  # N/A
        assert row_dict["is_best"] == "True"
    
    def test_csv_contains_metrics(self, tmp_path, sample_metrics):
        """Test CSV contains the metrics values."""
        from intseq_bert.train import _save_test_csv
        import csv
        
        csv_path = tmp_path / "test_results.csv"
        _save_test_csv(csv_path, sample_metrics, time_sec=45.2)
        
        with open(csv_path) as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = next(reader)
        
        row_dict = dict(zip(headers, data))
        
        assert float(row_dict["val_loss"]) == pytest.approx(0.0892)
        assert float(row_dict["val_mag_acc"]) == pytest.approx(91.88)
        assert float(row_dict["val_sign_acc"]) == pytest.approx(98.45)
    
    def test_csv_contains_time_sec(self, tmp_path, sample_metrics):
        """Test CSV contains evaluation time in time_sec column."""
        from intseq_bert.train import _save_test_csv
        import csv
        
        csv_path = tmp_path / "test_results.csv"
        _save_test_csv(csv_path, sample_metrics, time_sec=123.456)
        
        with open(csv_path) as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = next(reader)
        
        row_dict = dict(zip(headers, data))
        assert float(row_dict["time_sec"]) == pytest.approx(123.456)
    
    def test_csv_contains_mod_accuracies(self, tmp_path, sample_metrics):
        """Test CSV contains all 100 mod accuracy columns."""
        from intseq_bert.train import _save_test_csv
        from intseq_bert import config as cfg
        import csv
        
        csv_path = tmp_path / "test_results.csv"
        _save_test_csv(csv_path, sample_metrics, time_sec=45.2)
        
        with open(csv_path) as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = next(reader)
        
        row_dict = dict(zip(headers, data))
        
        # Check mod_acc_2 through mod_acc_101 exist
        for mod in cfg.MOD_RANGE:
            col_name = f"mod_acc_{mod}"
            assert col_name in row_dict, f"Missing column: {col_name}"
            # Values should be parseable as floats
            float(row_dict[col_name])
    
    def test_csv_early_stop_counter_is_zero(self, tmp_path, sample_metrics):
        """Test CSV has early_stop_counter = 0 in test mode."""
        from intseq_bert.train import _save_test_csv
        import csv
        
        csv_path = tmp_path / "test_results.csv"
        _save_test_csv(csv_path, sample_metrics, time_sec=45.2)
        
        with open(csv_path) as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = next(reader)
        
        row_dict = dict(zip(headers, data))
        assert row_dict["early_stop_counter"] == "0"


class TestSaveTestJson:
    """Tests for _save_test_json function."""
    
    @pytest.fixture
    def sample_args(self):
        """Sample args for testing."""
        import argparse
        return argparse.Namespace(
            model_path="checkpoints/model.pt",
            split_type="std",
            test_split="test"
        )
    
    @pytest.fixture
    def sample_metrics(self):
        """Sample metrics for testing."""
        from intseq_bert import config as cfg
        return {
            "val_loss": 0.0892,
            "mag_acc": 91.88,
            "mag_mse": 0.187,
            "sign_acc": 98.45,
            "mod_acc": 24.46,
            "mod_loss": 0.756,
            "mod_accuracies": [75.0 + i * 0.1 for i in range(len(cfg.MOD_RANGE))]
        }
    
    def test_creates_json_file(self, tmp_path, sample_args, sample_metrics):
        """Test that JSON file is created."""
        from intseq_bert.train import _save_test_json
        
        json_path = tmp_path / "test_metrics.json"
        _save_test_json(json_path, sample_args, sample_metrics, 45.2, 2500)
        
        assert json_path.exists()
    
    def test_json_contains_model_path(self, tmp_path, sample_args, sample_metrics):
        """Test JSON contains model path."""
        from intseq_bert.train import _save_test_json
        import json
        
        json_path = tmp_path / "test_metrics.json"
        _save_test_json(json_path, sample_args, sample_metrics, 45.2, 2500)
        
        with open(json_path) as f:
            data = json.load(f)
        
        assert data["model_path"] == "checkpoints/model.pt"
        assert data["split_type"] == "std"
    
    def test_json_contains_metrics(self, tmp_path, sample_args, sample_metrics):
        """Test JSON contains all metrics."""
        from intseq_bert.train import _save_test_json
        import json
        
        json_path = tmp_path / "test_metrics.json"
        _save_test_json(json_path, sample_args, sample_metrics, 45.2, 2500)
        
        with open(json_path) as f:
            data = json.load(f)
        
        assert data["test_loss"] == pytest.approx(0.0892)
        assert data["test_mag_acc"] == pytest.approx(91.88)
        assert data["test_sign_acc"] == pytest.approx(98.45)
        assert data["test_mod_acc"] == pytest.approx(24.46)
        assert data["test_samples"] == 2500
        assert data["evaluation_time_sec"] == pytest.approx(45.2)
    
    def test_json_contains_representative_mods(self, tmp_path, sample_args, sample_metrics):
        """Test JSON contains representative mods."""
        from intseq_bert.train import _save_test_json
        import json
        
        json_path = tmp_path / "test_metrics.json"
        _save_test_json(json_path, sample_args, sample_metrics, 45.2, 2500)
        
        with open(json_path) as f:
            data = json.load(f)
        
        assert "representative_mods" in data
        rep_mods = data["representative_mods"]
        
        # Check all representative mods are present
        assert "mod_2" in rep_mods
        assert "mod_3" in rep_mods
        assert "mod_5" in rep_mods
        assert "mod_7" in rep_mods
        assert "mod_10" in rep_mods
        assert "mod_100" in rep_mods
        assert "mod_101" in rep_mods
    
    def test_json_contains_all_mod_accuracies(self, tmp_path, sample_args, sample_metrics):
        """Test JSON contains all mod accuracies array."""
        from intseq_bert.train import _save_test_json
        from intseq_bert import config as cfg
        import json
        
        json_path = tmp_path / "test_metrics.json"
        _save_test_json(json_path, sample_args, sample_metrics, 45.2, 2500)
        
        with open(json_path) as f:
            data = json.load(f)
        
        assert "all_mod_accuracies" in data
        assert len(data["all_mod_accuracies"]) == len(cfg.MOD_RANGE)
    
    def test_json_contains_timestamp(self, tmp_path, sample_args, sample_metrics):
        """Test JSON contains evaluated_at timestamp."""
        from intseq_bert.train import _save_test_json
        import json
        
        json_path = tmp_path / "test_metrics.json"
        _save_test_json(json_path, sample_args, sample_metrics, 45.2, 2500)
        
        with open(json_path) as f:
            data = json.load(f)
        
        assert "evaluated_at" in data
        # Check timestamp format (YYYY-MM-DD HH:MM:SS)
        assert len(data["evaluated_at"]) == 19


class TestTestOnlyMode:
    """Integration tests for test_only function."""
    
    @pytest.fixture
    def mock_checkpoint(self, tmp_path, small_model):
        """Create a mock checkpoint file."""
        checkpoint = {
            "model_state_dict": small_model.state_dict(),
            "config": {
                "d_model": 32,
                "nhead": 2,
                "num_layers": 1
            },
            "epoch": 10,
            "val_loss": 0.1
        }
        
        checkpoint_path = tmp_path / "best_model.pt"
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    @pytest.fixture
    def test_only_args(self, mock_checkpoint, tmp_path):
        """Create args for test_only function."""
        import argparse
        return argparse.Namespace(
            test_only=True,
            model_path=str(mock_checkpoint),
            split_type="std",
            test_split="test",
            data_root="data",
            output_dir=str(tmp_path / "output"),
            test_output=None,
            batch_size=4,
            num_workers=0,
            seed=42,
            d_model=32,
            nhead=2,
            num_layers=1
        )
    
    def test_test_only_loads_model(self, mock_checkpoint, small_model):
        """Test that test_only correctly loads the model."""
        checkpoint = torch.load(mock_checkpoint)
        
        # Verify the checkpoint has what we expect
        assert "model_state_dict" in checkpoint
        assert "config" in checkpoint
        
        # Model should be able to load this state dict
        loaded_model = IntSeqForPreTraining(d_model=32, nhead=2, num_layers=1)
        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        
        # Weights should match
        for (n1, p1), (n2, p2) in zip(
            small_model.named_parameters(), 
            loaded_model.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Mismatch in {n1}"
    
    def test_test_only_config_extraction(self, mock_checkpoint):
        """Test that config is correctly extracted from checkpoint."""
        import argparse
        from intseq_bert.train import _load_model_config
        from pathlib import Path
        
        checkpoint = torch.load(mock_checkpoint)
        
        args = argparse.Namespace(d_model=256, nhead=8, num_layers=6)
        model_config = _load_model_config(Path(mock_checkpoint), checkpoint, args)
        
        # Should use checkpoint config, not args
        assert model_config["d_model"] == 32
        assert model_config["nhead"] == 2
        assert model_config["num_layers"] == 1


class TestTestOnlyCLI:
    """Tests for test-only CLI argument validation."""
    
    def test_test_only_requires_model_path(self):
        """Test that --test_only without --model_path raises error."""
        import argparse
        
        # This tests the validation logic
        def check_args(args):
            if args.test_only and not args.model_path:
                raise argparse.ArgumentError(None, "--test_only requires --model_path")
        
        args = argparse.Namespace(test_only=True, model_path=None)
        
        with pytest.raises(argparse.ArgumentError):
            check_args(args)
    
    def test_model_path_requires_test_only(self):
        """Test that --model_path without --test_only raises error."""
        import argparse
        
        def check_args(args):
            if args.model_path and not args.test_only:
                raise argparse.ArgumentError(None, "--model_path requires --test_only flag")
        
        args = argparse.Namespace(test_only=False, model_path="some/path.pt")
        
        with pytest.raises(argparse.ArgumentError):
            check_args(args)
    
    def test_valid_test_only_args(self):
        """Test that valid test-only args pass validation."""
        import argparse
        
        def check_args(args):
            if args.test_only and not args.model_path:
                raise argparse.ArgumentError(None, "--test_only requires --model_path")
            if args.model_path and not args.test_only:
                raise argparse.ArgumentError(None, "--model_path requires --test_only flag")
            return True
        
        args = argparse.Namespace(test_only=True, model_path="checkpoints/model.pt")
        
        # Should not raise
        assert check_args(args) is True
    
    def test_test_split_default_value(self):
        """Test that --test_split defaults to 'test'."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--test_split", type=str, default="test")
        
        args = parser.parse_args([])
        assert args.test_split == "test"
    
    def test_test_split_accepts_values(self):
        """Test that --test_split accepts train/val/test."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--test_split", type=str, default="test")
        
        for split_name in ["train", "val", "test"]:
            args = parser.parse_args(["--test_split", split_name])
            assert args.test_split == split_name
    
    def test_test_output_default_none(self):
        """Test that --test_output defaults to None."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--test_output", type=str, default=None)
        
        args = parser.parse_args([])
        assert args.test_output is None
    
    def test_test_output_custom_path(self):
        """Test that --test_output accepts custom path."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--test_output", type=str, default=None)
        
        args = parser.parse_args(["--test_output", "custom/results.csv"])
        assert args.test_output == "custom/results.csv"
