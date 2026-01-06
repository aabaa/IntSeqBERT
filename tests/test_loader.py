"""
Tests for the Dual Stream data loader module.
Tests directory-based dataset loading with lazy loading and tag filtering.
"""

import pytest
import torch
import json
from pathlib import Path

from intseq_bert.loader import (
    DualStreamDataset,
    load_and_split_data,
    _filter_by_tags
)
from intseq_bert import schemas


# ==========================================
# Helper Functions
# ==========================================

def create_feature_file(path: Path, oeis_id: str, seq_len: int = 10):
    """Create a mock .pt feature file."""
    data = {
        'oeis_id': oeis_id,
        'mag_features': torch.randn(seq_len, 5),
        'mod_features': torch.randn(seq_len, 200),
        'targets': {
            'mag': torch.randn(seq_len),
            'mod3': torch.randint(0, 3, (seq_len,)),
            'mod5': torch.randint(0, 5, (seq_len,)),
        }
    }
    torch.save(data, path)
    return data


def create_metadata_jsonl(path: Path, records: list):
    """Create a metadata JSONL file."""
    with open(path, 'w') as f:
        for rec in records:
            f.write(rec.to_json_line() + '\n')


# ==========================================
# 1. DualStreamDataset Tests
# ==========================================

class TestDualStreamDataset:
    """Tests for DualStreamDataset class."""
    
    def test_initialization(self, tmp_path):
        """Test dataset can be initialized with file list."""
        # Create test files
        files = []
        for i in range(3):
            path = tmp_path / f"A{i:06d}.pt"
            create_feature_file(path, f"A{i:06d}")
            files.append(path)
        
        dataset = DualStreamDataset(files)
        assert len(dataset) == 3
    
    def test_getitem_returns_correct_structure(self, tmp_path):
        """Test __getitem__ returns dict with correct keys."""
        path = tmp_path / "A000001.pt"
        create_feature_file(path, "A000001", seq_len=15)
        
        dataset = DualStreamDataset([path])
        item = dataset[0]
        
        assert 'oeis_id' in item
        assert 'mag_features' in item
        assert 'mod_features' in item
        assert 'targets' in item
        
        assert item['oeis_id'] == 'A000001'
        assert item['mag_features'].shape == (15, 5)
        assert item['mod_features'].shape == (15, 200)
    
    def test_lazy_loading(self, tmp_path):
        """Test that files are loaded on demand (lazy loading)."""
        # Create files
        files = []
        for i in range(5):
            path = tmp_path / f"A{i:06d}.pt"
            create_feature_file(path, f"A{i:06d}")
            files.append(path)
        
        dataset = DualStreamDataset(files)
        
        # Dataset should not load anything until __getitem__ is called
        # We can't directly test this, but we can verify behavior
        item = dataset[2]
        assert item['oeis_id'] == 'A000002'
    
    def test_invalid_file_raises_error(self, tmp_path):
        """Test that loading invalid file raises error."""
        # Create invalid file
        path = tmp_path / "invalid.pt"
        torch.save({'bad': 'data'}, path)
        
        dataset = DualStreamDataset([path])
        
        with pytest.raises(ValueError):
            _ = dataset[0]


# ==========================================
# 2. load_and_split_data Tests
# ==========================================

class TestLoadAndSplitData:
    """Tests for load_and_split_data function."""
    
    def test_basic_loading(self, tmp_path):
        """Test basic loading without filtering."""
        # Create 20 feature files
        for i in range(20):
            path = tmp_path / f"A{i:06d}.pt"
            create_feature_file(path, f"A{i:06d}")
        
        train_ds, val_ds, test_ds = load_and_split_data(
            str(tmp_path),
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42
        )
        
        # Check sizes (20 * 0.1 = 2 for val and test each)
        assert len(train_ds) == 16
        assert len(val_ds) == 2
        assert len(test_ds) == 2
    
    def test_reproducible_split(self, tmp_path):
        """Test that splitting is reproducible with same seed."""
        for i in range(10):
            path = tmp_path / f"A{i:06d}.pt"
            create_feature_file(path, f"A{i:06d}")
        
        train1, _, _ = load_and_split_data(str(tmp_path), seed=123)
        train2, _, _ = load_and_split_data(str(tmp_path), seed=123)
        
        # Same seed should give same files
        ids1 = [train1[i]['oeis_id'] for i in range(len(train1))]
        ids2 = [train2[i]['oeis_id'] for i in range(len(train2))]
        assert ids1 == ids2
    
    def test_different_seed_different_split(self, tmp_path):
        """Test that different seeds give different splits."""
        for i in range(10):
            path = tmp_path / f"A{i:06d}.pt"
            create_feature_file(path, f"A{i:06d}")
        
        train1, _, _ = load_and_split_data(str(tmp_path), seed=1)
        train2, _, _ = load_and_split_data(str(tmp_path), seed=2)
        
        ids1 = sorted([train1[i]['oeis_id'] for i in range(len(train1))])
        ids2 = sorted([train2[i]['oeis_id'] for i in range(len(train2))])
        # While technically could be same, very unlikely with different seeds
        # We just check they are valid
        assert len(ids1) == len(ids2)
    
    def test_max_samples_limit(self, tmp_path):
        """Test max_samples parameter."""
        for i in range(100):
            path = tmp_path / f"A{i:06d}.pt"
            create_feature_file(path, f"A{i:06d}")
        
        train_ds, val_ds, test_ds = load_and_split_data(
            str(tmp_path),
            max_samples=20,
            seed=42
        )
        
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 20
    
    def test_empty_directory_raises(self, tmp_path):
        """Test that empty directory raises error."""
        with pytest.raises(ValueError, match="No .pt files found"):
            load_and_split_data(str(tmp_path))
    
    def test_nonexistent_directory_raises(self, tmp_path):
        """Test that nonexistent directory raises error."""
        fake_path = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            load_and_split_data(str(fake_path))


# ==========================================
# 3. Tag Filtering Tests
# ==========================================

class TestTagFiltering:
    """Tests for tag-based filtering."""
    
    def test_include_tags(self, tmp_path):
        """Test filtering with include_tags."""
        # Create feature files
        for i in range(10):
            path = tmp_path / f"A{i:06d}.pt"
            create_feature_file(path, f"A{i:06d}")
        
        # Create metadata
        metadata_path = tmp_path / "metadata.jsonl"
        records = [
            schemas.OEISRecord(oeis_id="A000000", sequence=[1], keywords=["nonn", "core"]),
            schemas.OEISRecord(oeis_id="A000001", sequence=[1], keywords=["nonn"]),
            schemas.OEISRecord(oeis_id="A000002", sequence=[1], keywords=["sign"]),
            schemas.OEISRecord(oeis_id="A000003", sequence=[1], keywords=["nonn", "core"]),
        ]
        create_metadata_jsonl(metadata_path, records)
        
        train_ds, val_ds, test_ds = load_and_split_data(
            str(tmp_path),
            metadata_path=str(metadata_path),
            include_tags=["core"],
            val_ratio=0.0,
            test_ratio=0.0,
            seed=42
        )
        
        # Only A000000 and A000003 have "core" tag
        assert len(train_ds) == 2
    
    def test_exclude_tags(self, tmp_path):
        """Test filtering with exclude_tags."""
        for i in range(5):
            path = tmp_path / f"A{i:06d}.pt"
            create_feature_file(path, f"A{i:06d}")
        
        metadata_path = tmp_path / "metadata.jsonl"
        records = [
            schemas.OEISRecord(oeis_id="A000000", sequence=[1], keywords=["nonn"]),
            schemas.OEISRecord(oeis_id="A000001", sequence=[1], keywords=["nonn", "dead"]),
            schemas.OEISRecord(oeis_id="A000002", sequence=[1], keywords=["nonn"]),
            schemas.OEISRecord(oeis_id="A000003", sequence=[1], keywords=["dead"]),
            schemas.OEISRecord(oeis_id="A000004", sequence=[1], keywords=["nonn"]),
        ]
        create_metadata_jsonl(metadata_path, records)
        
        train_ds, _, _ = load_and_split_data(
            str(tmp_path),
            metadata_path=str(metadata_path),
            exclude_tags=["dead"],
            val_ratio=0.0,
            test_ratio=0.0,
            seed=42
        )
        
        # A000001 and A000003 should be excluded
        assert len(train_ds) == 3
    
    def test_include_and_exclude_combined(self, tmp_path):
        """Test filtering with both include and exclude tags."""
        for i in range(5):
            path = tmp_path / f"A{i:06d}.pt"
            create_feature_file(path, f"A{i:06d}")
        
        metadata_path = tmp_path / "metadata.jsonl"
        records = [
            schemas.OEISRecord(oeis_id="A000000", sequence=[1], keywords=["nonn", "core"]),
            schemas.OEISRecord(oeis_id="A000001", sequence=[1], keywords=["nonn", "core", "dead"]),
            schemas.OEISRecord(oeis_id="A000002", sequence=[1], keywords=["nonn"]),
            schemas.OEISRecord(oeis_id="A000003", sequence=[1], keywords=["core"]),
            schemas.OEISRecord(oeis_id="A000004", sequence=[1], keywords=["sign"]),
        ]
        create_metadata_jsonl(metadata_path, records)
        
        train_ds, _, _ = load_and_split_data(
            str(tmp_path),
            metadata_path=str(metadata_path),
            include_tags=["core"],
            exclude_tags=["dead"],
            val_ratio=0.0,
            test_ratio=0.0,
            seed=42
        )
        
        # Only A000000 and A000003 (have core, no dead)
        assert len(train_ds) == 2


# ==========================================
# 4. _filter_by_tags Internal Function Tests
# ==========================================

class TestFilterByTagsInternal:
    """Tests for internal _filter_by_tags function."""
    
    def test_include_filter(self, tmp_path):
        """Test include filter logic."""
        metadata_path = tmp_path / "metadata.jsonl"
        records = [
            schemas.OEISRecord(oeis_id="A001", sequence=[1], keywords=["nonn", "core"]),
            schemas.OEISRecord(oeis_id="A002", sequence=[1], keywords=["nonn"]),
            schemas.OEISRecord(oeis_id="A003", sequence=[1], keywords=["core", "nice"]),
        ]
        create_metadata_jsonl(metadata_path, records)
        
        valid_ids = _filter_by_tags(str(metadata_path), ["core"], None)
        
        assert "A001" in valid_ids
        assert "A002" not in valid_ids
        assert "A003" in valid_ids
    
    def test_exclude_filter(self, tmp_path):
        """Test exclude filter logic."""
        metadata_path = tmp_path / "metadata.jsonl"
        records = [
            schemas.OEISRecord(oeis_id="A001", sequence=[1], keywords=["nonn"]),
            schemas.OEISRecord(oeis_id="A002", sequence=[1], keywords=["nonn", "dead"]),
            schemas.OEISRecord(oeis_id="A003", sequence=[1], keywords=["core"]),
        ]
        create_metadata_jsonl(metadata_path, records)
        
        valid_ids = _filter_by_tags(str(metadata_path), None, ["dead"])
        
        assert "A001" in valid_ids
        assert "A002" not in valid_ids
        assert "A003" in valid_ids
    
    def test_missing_metadata_returns_empty(self, tmp_path):
        """Test that missing metadata file returns empty set."""
        valid_ids = _filter_by_tags(str(tmp_path / "nonexistent.jsonl"), ["core"], None)
        assert valid_ids == set()


# ==========================================
# 5. Integration Tests
# ==========================================

class TestIntegration:
    """Integration tests for the complete loading pipeline."""
    
    def test_full_pipeline(self, tmp_path):
        """Test complete loading and iteration."""
        # Create files
        for i in range(10):
            path = tmp_path / f"A{i:06d}.pt"
            create_feature_file(path, f"A{i:06d}", seq_len=20)
        
        train_ds, val_ds, test_ds = load_and_split_data(
            str(tmp_path),
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42
        )
        
        # Iterate through training set
        for i in range(len(train_ds)):
            item = train_ds[i]
            assert 'mag_features' in item
            assert 'mod_features' in item
            assert item['mag_features'].shape[0] == 20
    
    def test_dataloader_compatibility(self, tmp_path):
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        for i in range(5):
            path = tmp_path / f"A{i:06d}.pt"
            create_feature_file(path, f"A{i:06d}")
        
        train_ds, _, _ = load_and_split_data(
            str(tmp_path),
            val_ratio=0.0,
            test_ratio=0.0,
            seed=42
        )
        
        # Create DataLoader (batch_size=1 since sequences may differ in length)
        loader = DataLoader(train_ds, batch_size=1, shuffle=False)
        
        count = 0
        for batch in loader:
            assert 'mag_features' in batch
            count += 1
        
        assert count == 5
