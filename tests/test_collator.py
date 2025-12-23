import pytest
import torch
from intseq_bert.collator import IntSeqCollator

def test_collator_shapes():
    # Setup: 2 sequences with diff lengths
    # Seq1: len=5, dim=4
    # Seq2: len=3, dim=4
    dim = 4
    seq1 = torch.randn(5, dim)
    seq2 = torch.randn(3, dim)
    examples = [seq1, seq2]
    
    collator = IntSeqCollator(feature_dim=dim, mask_prob=0.5)
    batch = collator(examples)
    
    # Check keys
    assert "inputs" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert "mask_matrix" in batch
    
    # Check Shapes
    # Max len should be 5
    assert batch["inputs"].shape == (2, 5, dim)
    assert batch["attention_mask"].shape == (2, 5)
    assert batch["labels"].shape == (2, 5, dim)
    
def test_collator_padding_logic():
    dim = 4
    seq1 = torch.ones(5, dim) # Long
    seq2 = torch.ones(2, dim) # Short
    examples = [seq1, seq2]
    
    collator = IntSeqCollator(feature_dim=dim, mask_prob=0.0) # No masking for this test
    batch = collator(examples)
    
    # Check Seq2 padding (index 1)
    # indices [0, 1] should be real (1.0), [2, 3, 4] should be pad (0.0)
    assert torch.all(batch["inputs"][1, :2] == 1.0)
    assert torch.all(batch["inputs"][1, 2:] == 0.0)
    
    # Check Attention Mask
    expected_mask = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0]
    ])
    assert torch.equal(batch["attention_mask"], expected_mask)

def test_collator_masking_constraints():
    """Ensure padding tokens are NEVER masked."""
    dim = 4
    seq1 = torch.randn(5, dim)
    seq2 = torch.randn(2, dim)
    examples = [seq1, seq2]
    
    # Set mask_prob to 1.0 (Force masking everything possible)
    collator = IntSeqCollator(feature_dim=dim, mask_prob=1.0)
    batch = collator(examples)
    
    mask_matrix = batch["mask_matrix"]
    attention_mask = batch["attention_mask"]
    
    # 1. Real tokens should be masked (because prob=1.0)
    assert mask_matrix[0, 0] == True 
    assert mask_matrix[1, 0] == True
    
    # 2. Padding tokens should NOT be masked
    # Seq2 has padding at indices 2, 3, 4
    assert mask_matrix[1, 2] == False
    assert mask_matrix[1, 3] == False
    assert mask_matrix[1, 4] == False
    
    # Logical check: mask_matrix should be subset of attention_mask
    # (Cannot mask what does not exist)
    assert torch.all(mask_matrix <= attention_mask.bool())

def test_masking_value():
    """Ensure masked positions are actually zeroed out in inputs."""
    dim = 4
    seq1 = torch.ones(5, dim) # All 1s
    examples = [seq1]
    
    collator = IntSeqCollator(feature_dim=dim, mask_prob=1.0)
    batch = collator(examples)
    
    # Since prob=1.0, everything should be masked
    # Inputs should be 0.0, Labels should be 1.0
    assert torch.all(batch["inputs"] == 0.0)
    assert torch.all(batch["labels"] == 1.0)
    assert torch.all(batch["mask_matrix"] == True)