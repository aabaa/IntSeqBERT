"""
Tests for the data loader module with tag filtering.
"""

import pytest
import torch
import json
from pathlib import Path

from intseq_bert import loader


@pytest.fixture
def tmp_jsonl_with_tags(tmp_path):
    """
    Create a temporary JSONL file with keyword metadata.
    
    Creates 4 sequences with different tag combinations:
    - A000001: ["core", "nonn"]
    - A000002: ["easy", "nonn"]
    - A000003: ["core", "hard"]
    - A000004: ["nonn"]
    """
    data = [
        {
            "oeis_id": "A000001",
            "sequence": list(range(1, 21)),  # Length 20
            "keywords": ["core", "nonn"]
        },
        {
            "oeis_id": "A000002",
            "sequence": list(range(1, 16)),  # Length 15
            "keywords": ["easy", "nonn"]
        },
        {
            "oeis_id": "A000003",
            "sequence": list(range(1, 26)),  # Length 25
            "keywords": ["core", "hard"]
        },
        {
            "oeis_id": "A000004",
            "sequence": list(range(1, 11)),  # Length 10
            "keywords": ["nonn"]
        }
    ]
    
    jsonl_path = tmp_path / "metadata.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')
    
    return jsonl_path


@pytest.fixture
def tmp_features_pt(tmp_path):
    """
    Create a temporary .pt file with matching feature tensors.
    
    Creates dummy tensors with shape (SeqLen, 27) for each sequence.
    """
    features = {
        "A000001": torch.randn(20, 27, dtype=torch.float32),
        "A000002": torch.randn(15, 27, dtype=torch.float32),
        "A000003": torch.randn(25, 27, dtype=torch.float32),
        "A000004": torch.randn(10, 27, dtype=torch.float32)
    }
    
    pt_path = tmp_path / "features.pt"
    torch.save(features, pt_path)
    
    return pt_path


def test_tag_filtering_include(tmp_jsonl_with_tags, tmp_features_pt):
    """
    Test include_tags filtering - should only keep sequences with specified tags.
    """
    train_ds, val_ds, test_ds = loader.load_and_split_data(
        features_path=str(tmp_features_pt),
        metadata_path=str(tmp_jsonl_with_tags),
        include_tags=["core"],  # Only A000001 and A000003 have "core"
        val_ratio=0.0,
        test_ratio=0.0,
        seed=42
    )
    
    # Should have exactly 2 sequences (A000001, A000003)
    assert len(train_ds) == 2
    assert len(val_ds) == 0
    assert len(test_ds) == 0
    
    # Verify tensors have correct shape
    tensor1 = train_ds[0]
    assert tensor1.shape[1] == 27  # Feature dimension


def test_tag_filtering_exclude(tmp_jsonl_with_tags, tmp_features_pt):
    """
    Test exclude_tags filtering - should remove sequences with specified tags.
    """
    train_ds, val_ds, test_ds = loader.load_and_split_data(
        features_path=str(tmp_features_pt),
        metadata_path=str(tmp_jsonl_with_tags),
        exclude_tags=["easy"],  # Remove A000002
        val_ratio=0.0,
        test_ratio=0.0,
        seed=42
    )
    
    # Should have 3 sequences (all except A000002)
    assert len(train_ds) == 3
    assert len(val_ds) == 0
    assert len(test_ds) == 0


def test_combined_filtering(tmp_jsonl_with_tags, tmp_features_pt):
    """
    Test combination of include and exclude tags.
    Exclude takes precedence.
    """
    train_ds, val_ds, test_ds = loader.load_and_split_data(
        features_path=str(tmp_features_pt),
        metadata_path=str(tmp_jsonl_with_tags),
        include_tags=["core"],  # A000001, A000003
        exclude_tags=["hard"],  # Remove A000003
        val_ratio=0.0,
        test_ratio=0.0,
        seed=42
    )
    
    # Should have only A000001 (has "core" but not "hard")
    assert len(train_ds) == 1


def test_no_metadata_loading(tmp_features_pt):
    """
    Test loading without metadata - should load all sequences.
    """
    train_ds, val_ds, test_ds = loader.load_and_split_data(
        features_path=str(tmp_features_pt),
        metadata_path=None,  # No metadata filtering
        val_ratio=0.0,
        test_ratio=0.0,
        seed=42
    )
    
    # Should have all 4 sequences
    assert len(train_ds) == 4
    assert len(val_ds) == 0
    assert len(test_ds) == 0


def test_data_splitting(tmp_features_pt):
    """
    Test train/val/test split ratios are correct.
    """
    train_ds, val_ds, test_ds = loader.load_and_split_data(
        features_path=str(tmp_features_pt),
        metadata_path=None,
        val_ratio=0.25,   # 25% validation
        test_ratio=0.25,  # 25% test
        seed=42
    )
    
    total = len(train_ds) + len(val_ds) + len(test_ds)
    assert total == 4  # All sequences loaded
    
    # With 4 sequences: test=1, val=1, train=2
    assert len(test_ds) == 1
    assert len(val_ds) == 1
    assert len(train_ds) == 2


def test_min_len_filtering(tmp_path):
    """
    Test min_len parameter filters out short sequences.
    """
    # Create features with varying lengths
    features = {
        "A000001": torch.randn(5, 27, dtype=torch.float32),   # Too short
        "A000002": torch.randn(15, 27, dtype=torch.float32),  # OK
        "A000003": torch.randn(8, 27, dtype=torch.float32),   # Too short
        "A000004": torch.randn(20, 27, dtype=torch.float32)   # OK
    }
    
    pt_path = tmp_path / "features_varying.pt"
    torch.save(features, pt_path)
    
    train_ds, val_ds, test_ds = loader.load_and_split_data(
        features_path=str(pt_path),
        metadata_path=None,
        min_len=10,  # Require at least 10 elements
        val_ratio=0.0,
        test_ratio=0.0,
        seed=42
    )
    
    # Should have only 2 sequences (A000002, A000004)
    assert len(train_ds) == 2


def test_dataset_interface(tmp_features_pt):
    """
    Test IntSeqDataset interface works correctly.
    """
    train_ds, _, _ = loader.load_and_split_data(
        features_path=str(tmp_features_pt),
        metadata_path=None,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=42
    )
    
    # Test __len__
    assert len(train_ds) == 4
    
    # Test __getitem__ returns tensors
    tensor = train_ds[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[1] == 27  # Feature dimension
    
    # Test indexing works for all items
    for i in range(len(train_ds)):
        t = train_ds[i]
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32


def test_empty_result_handling(tmp_path):
    """
    Test handling when no sequences pass the filter.
    """
    # Create metadata with no matching tags
    data = [
        {"oeis_id": "A000001", "sequence": [1, 2, 3], "keywords": ["other"]}
    ]
    jsonl_path = tmp_path / "metadata_nomatch.jsonl"
    with open(jsonl_path, 'w') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')
    
    # Create matching features
    features = {"A000001": torch.randn(3, 27)}
    pt_path = tmp_path / "features_nomatch.pt"
    torch.save(features, pt_path)
    
    # Try to load with non-matching tags
    train_ds, val_ds, test_ds = loader.load_and_split_data(
        features_path=str(pt_path),
        metadata_path=str(jsonl_path),
        include_tags=["nonexistent"],
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )
    
    # Should return empty datasets
    assert len(train_ds) == 0
    assert len(val_ds) == 0
    assert len(test_ds) == 0


def test_reproducible_splitting(tmp_features_pt):
    """
    Test that same seed produces same split.
    """
    # Load with seed 42
    train1, val1, test1 = loader.load_and_split_data(
        features_path=str(tmp_features_pt),
        metadata_path=None,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=42
    )
    
    # Load again with same seed
    train2, val2, test2 = loader.load_and_split_data(
        features_path=str(tmp_features_pt),
        metadata_path=None,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=42
    )
    
    # Should have same sizes
    assert len(train1) == len(train2)
    assert len(val1) == len(val2)
    assert len(test1) == len(test2)
    
    # Tensors should be identical (same objects from same split)
    for i in range(len(train1)):
        assert torch.equal(train1[i], train2[i])
