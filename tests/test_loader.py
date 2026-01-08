"""
Tests for the OEIS data loader module.

Covers:
1. OEISDataset - Dataset class for loading .pt files
2. create_splits - Admin function for generating static split files
3. load_dataset - Runtime function for loading datasets from split files
4. Integration tests
"""

import pytest
import torch
from pathlib import Path

from intseq_bert import loader, schemas, config


# ==========================================
# Helper Functions
# ==========================================

def create_feature_file(path: Path, oeis_id: str, seq_len: int = 10):
    """Create a mock .pt feature file with required keys."""
    data = {
        config.KEY_OEIS_ID: oeis_id,
        config.KEY_MAG_FEATURES: torch.randn(seq_len, 5),
        config.KEY_MOD_FEATURES: torch.randn(seq_len, 200),
        config.KEY_TARGETS: {
            'mag': torch.randn(seq_len),
            'mod3': torch.randint(0, 3, (seq_len,)),
        }
    }
    torch.save(data, path)
    return data


def create_jsonl_file(path: Path, records: list):
    """Create a JSONL file from OEISRecord list."""
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(rec.to_json_line() + '\n')


def setup_test_data_root(tmp_path: Path, oeis_ids: list, seq_len: int = 10):
    """
    Create a complete test data structure matching config layout:
    tmp_path/
      features/
        A000001.pt, A000002.pt, ...
    """
    features_dir = tmp_path / config.FEATURES_DIR_NAME
    features_dir.mkdir(parents=True)
    
    for oid in oeis_ids:
        create_feature_file(features_dir / f"{oid}.pt", oid, seq_len)
    
    return tmp_path


# ==========================================
# 1. OEISDataset Tests
# ==========================================

class TestOEISDataset:
    """Tests for OEISDataset class."""
    
    def test_initialization(self, tmp_path):
        """Test dataset can be initialized with ID list."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        ids = ["A000001", "A000002", "A000003"]
        for oid in ids:
            create_feature_file(features_dir / f"{oid}.pt", oid)
        
        dataset = loader.OEISDataset(ids, features_dir)
        assert len(dataset) == 3
    
    def test_getitem_returns_correct_structure(self, tmp_path):
        """Test __getitem__ returns dict with correct keys."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        create_feature_file(features_dir / "A000001.pt", "A000001", seq_len=15)
        
        dataset = loader.OEISDataset(["A000001"], features_dir)
        item = dataset[0]
        
        # Check required keys
        assert config.KEY_OEIS_ID in item
        assert config.KEY_MAG_FEATURES in item
        assert config.KEY_MOD_FEATURES in item
        
        # Check values
        assert item[config.KEY_OEIS_ID] == "A000001"
        assert item[config.KEY_MAG_FEATURES].shape == (15, 5)
        assert item[config.KEY_MOD_FEATURES].shape == (15, 200)
    
    def test_missing_file_raises_error(self, tmp_path):
        """Test Fail Fast: missing file raises FileNotFoundError."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        dataset = loader.OEISDataset(["A999999"], features_dir)
        
        with pytest.raises(FileNotFoundError, match="Feature file missing"):
            _ = dataset[0]
    
    def test_missing_required_key_raises_error(self, tmp_path):
        """Test that missing required keys raise ValueError."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        # Create file with missing required key
        bad_data = {"bad_key": torch.randn(10, 5)}
        torch.save(bad_data, features_dir / "A000001.pt")
        
        dataset = loader.OEISDataset(["A000001"], features_dir)
        
        with pytest.raises(ValueError, match="Missing required key"):
            _ = dataset[0]


# ==========================================
# 2. create_splits Tests (Admin Function)
# ==========================================

class TestCreateSplits:
    """Tests for create_splits function (Admin/Setup)."""
    
    def test_creates_split_files(self, tmp_path):
        """Test that split files are created correctly."""
        # Setup
        data_root = setup_test_data_root(tmp_path, [f"A{i:06d}" for i in range(20)])
        
        jsonl_path = tmp_path / "data.jsonl"
        records = [schemas.OEISRecord(f"A{i:06d}", [i]) for i in range(20)]
        create_jsonl_file(jsonl_path, records)
        
        # Execute
        loader.create_splits(
            source_jsonl=str(jsonl_path),
            output_split_type="test_split",
            data_root=str(data_root)
        )
        
        # Verify
        split_dir = data_root / config.SPLIT_DIR_NAME / "test_split"
        assert (split_dir / "train.txt").exists()
        assert (split_dir / "val.txt").exists()
        assert (split_dir / "test.txt").exists()
        
        # Check total count matches
        total = 0
        for name in ["train.txt", "val.txt", "test.txt"]:
            with open(split_dir / name) as f:
                total += len([line for line in f if line.strip()])
        assert total == 20
    
    def test_deterministic_shuffle(self, tmp_path):
        """Test that same seed produces same split (via config.SEED)."""
        # Setup two identical data roots
        ids = [f"A{i:06d}" for i in range(50)]
        
        for trial in ["trial1", "trial2"]:
            trial_root = tmp_path / trial
            setup_test_data_root(trial_root, ids)
            
            jsonl_path = trial_root / "data.jsonl"
            records = [schemas.OEISRecord(oid, [1]) for oid in ids]
            create_jsonl_file(jsonl_path, records)
            
            loader.create_splits(
                source_jsonl=str(jsonl_path),
                output_split_type="strict",
                data_root=str(trial_root)
            )
        
        # Compare results
        for split_name in ["train.txt", "val.txt", "test.txt"]:
            path1 = tmp_path / "trial1" / config.SPLIT_DIR_NAME / "strict" / split_name
            path2 = tmp_path / "trial2" / config.SPLIT_DIR_NAME / "strict" / split_name
            
            with open(path1) as f1, open(path2) as f2:
                assert f1.read() == f2.read(), f"Split {split_name} differs between runs"
    
    def test_tag_filtering_include(self, tmp_path):
        """Test include_tags filtering."""
        data_root = setup_test_data_root(tmp_path, ["A000001", "A000002", "A000003"])
        
        jsonl_path = tmp_path / "data.jsonl"
        records = [
            schemas.OEISRecord("A000001", [1], keywords=["core", "nonn"]),
            schemas.OEISRecord("A000002", [2], keywords=["nonn"]),
            schemas.OEISRecord("A000003", [3], keywords=["core"]),
        ]
        create_jsonl_file(jsonl_path, records)
        
        loader.create_splits(
            source_jsonl=str(jsonl_path),
            output_split_type="filtered",
            include_tags=["core"],
            data_root=str(data_root)
        )
        
        # Only A000001 and A000003 should be included
        split_dir = data_root / config.SPLIT_DIR_NAME / "filtered"
        total = 0
        for name in ["train.txt", "val.txt", "test.txt"]:
            with open(split_dir / name) as f:
                total += len([line for line in f if line.strip()])
        assert total == 2
    
    def test_tag_filtering_exclude(self, tmp_path):
        """Test exclude_tags filtering."""
        data_root = setup_test_data_root(tmp_path, ["A000001", "A000002", "A000003"])
        
        jsonl_path = tmp_path / "data.jsonl"
        records = [
            schemas.OEISRecord("A000001", [1], keywords=["nonn"]),
            schemas.OEISRecord("A000002", [2], keywords=["nonn", "dead"]),
            schemas.OEISRecord("A000003", [3], keywords=["nonn"]),
        ]
        create_jsonl_file(jsonl_path, records)
        
        loader.create_splits(
            source_jsonl=str(jsonl_path),
            output_split_type="filtered",
            exclude_tags=["dead"],
            data_root=str(data_root)
        )
        
        # A000002 should be excluded
        split_dir = data_root / config.SPLIT_DIR_NAME / "filtered"
        total = 0
        for name in ["train.txt", "val.txt", "test.txt"]:
            with open(split_dir / name) as f:
                total += len([line for line in f if line.strip()])
        assert total == 2
    
    def test_skips_missing_feature_files(self, tmp_path):
        """Test that IDs without feature files are skipped."""
        # Only create features for some IDs
        data_root = setup_test_data_root(tmp_path, ["A000001", "A000002"])
        
        jsonl_path = tmp_path / "data.jsonl"
        records = [
            schemas.OEISRecord("A000001", [1]),
            schemas.OEISRecord("A000002", [2]),
            schemas.OEISRecord("A000003", [3]),  # No feature file
            schemas.OEISRecord("A000004", [4]),  # No feature file
        ]
        create_jsonl_file(jsonl_path, records)
        
        loader.create_splits(
            source_jsonl=str(jsonl_path),
            output_split_type="partial",
            data_root=str(data_root)
        )
        
        # Only 2 IDs should be in splits
        split_dir = data_root / config.SPLIT_DIR_NAME / "partial"
        total = 0
        for name in ["train.txt", "val.txt", "test.txt"]:
            with open(split_dir / name) as f:
                total += len([line for line in f if line.strip()])
        assert total == 2


# ==========================================
# 3. load_dataset Tests (Runtime Function)
# ==========================================

class TestLoadDataset:
    """Tests for load_dataset function (Runtime)."""
    
    def test_loads_from_split_file(self, tmp_path):
        """Test loading dataset from pre-existing split file."""
        # Setup data root with features
        ids = ["A000001", "A000002", "A000003"]
        data_root = setup_test_data_root(tmp_path, ids)
        
        # Create split file manually
        split_dir = data_root / config.SPLIT_DIR_NAME / "strict"
        split_dir.mkdir(parents=True)
        with open(split_dir / "train.txt", 'w') as f:
            for oid in ids:
                f.write(oid + '\n')
        
        # Load
        dataset = loader.load_dataset("strict", "train", data_root=str(data_root))
        
        assert len(dataset) == 3
        item = dataset[0]
        assert config.KEY_MAG_FEATURES in item
    
    def test_missing_split_file_raises_error(self, tmp_path):
        """Test that missing split file raises FileNotFoundError."""
        data_root = setup_test_data_root(tmp_path, ["A000001"])
        
        with pytest.raises(FileNotFoundError, match="Split file not found"):
            loader.load_dataset("nonexistent", "train", data_root=str(data_root))
    
    def test_no_shuffle_during_load(self, tmp_path):
        """
        CRITICAL: Test that load_dataset does NOT shuffle.
        Order must be preserved from split file.
        """
        ids = [f"A{i:06d}" for i in range(10)]
        data_root = setup_test_data_root(tmp_path, ids)
        
        # Create split file with specific order
        split_dir = data_root / config.SPLIT_DIR_NAME / "strict"
        split_dir.mkdir(parents=True)
        with open(split_dir / "train.txt", 'w') as f:
            for oid in ids:
                f.write(oid + '\n')
        
        # Load multiple times and verify order is preserved
        for _ in range(3):
            dataset = loader.load_dataset("strict", "train", data_root=str(data_root))
            loaded_ids = [dataset[i][config.KEY_OEIS_ID] for i in range(len(dataset))]
            assert loaded_ids == ids, "Order should be preserved from split file"


# ==========================================
# 4. Integration Tests
# ==========================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_create_then_load_pipeline(self, tmp_path):
        """Test the complete Admin -> Runtime workflow."""
        # Admin phase: create splits
        ids = [f"A{i:06d}" for i in range(30)]
        data_root = setup_test_data_root(tmp_path, ids, seq_len=20)
        
        jsonl_path = tmp_path / "data.jsonl"
        records = [schemas.OEISRecord(oid, [1, 2, 3]) for oid in ids]
        create_jsonl_file(jsonl_path, records)
        
        loader.create_splits(
            source_jsonl=str(jsonl_path),
            output_split_type="strict",
            data_root=str(data_root)
        )
        
        # Runtime phase: load datasets
        train_ds = loader.load_dataset("strict", "train", data_root=str(data_root))
        val_ds = loader.load_dataset("strict", "val", data_root=str(data_root))
        test_ds = loader.load_dataset("strict", "test", data_root=str(data_root))
        
        # Verify we can iterate and access data
        assert len(train_ds) + len(val_ds) + len(test_ds) == 30
        
        item = train_ds[0]
        assert item[config.KEY_MAG_FEATURES].shape[0] == 20
    
    def test_dataloader_compatibility(self, tmp_path):
        """Test that OEISDataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        ids = ["A000001", "A000002", "A000003", "A000004", "A000005"]
        data_root = setup_test_data_root(tmp_path, ids)
        
        # Create split
        split_dir = data_root / config.SPLIT_DIR_NAME / "strict"
        split_dir.mkdir(parents=True)
        with open(split_dir / "train.txt", 'w') as f:
            for oid in ids:
                f.write(oid + '\n')
        
        dataset = loader.load_dataset("strict", "train", data_root=str(data_root))
        
        # Use DataLoader with shuffle=True (this is where shuffling should happen)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        count = 0
        for batch in dataloader:
            assert config.KEY_MAG_FEATURES in batch
            count += 1
        
        assert count == 5
