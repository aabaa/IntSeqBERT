"""
Tests for the Dual Stream Collator module.
Tests padding, masking, and target batching for the Dual Stream architecture.
"""

import pytest
import torch

from intseq_bert.collator import DualStreamCollator


# ==========================================
# Helper Functions
# ==========================================

def create_mock_batch(seq_lengths: list) -> list:
    """Create a mock batch of data items with variable sequence lengths."""
    batch = []
    for i, length in enumerate(seq_lengths):
        item = {
            'oeis_id': f'A{i:06d}',
            'mag_features': torch.randn(length, 5),
            'mod_features': torch.randn(length, 200),
            'targets': {
                'mag': torch.randn(length),  # Float targets
                'mod3': torch.randint(0, 3, (length,)),  # Long targets
                'mod5': torch.randint(0, 5, (length,)),
                'mod100': torch.randint(0, 100, (length,)),
            }
        }
        batch.append(item)
    return batch


# ==========================================
# 1. Basic Collator Tests
# ==========================================

class TestDualStreamCollatorBasic:
    """Basic tests for DualStreamCollator."""
    
    def test_initialization(self):
        """Test collator can be initialized."""
        collator = DualStreamCollator()
        assert collator.mask_prob == 0.15
        
        collator = DualStreamCollator(mask_prob=0.2)
        assert collator.mask_prob == 0.2
    
    def test_output_keys(self):
        """Test that collator returns correct keys."""
        collator = DualStreamCollator()
        batch = create_mock_batch([10, 10])
        
        result = collator(batch)
        
        expected_keys = [
            'mag_inputs', 'mod_inputs', 'attention_mask', 
            'mask_matrix', 'mag_labels', 'mod_labels', 'targets'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_output_shapes_same_length(self):
        """Test output shapes when all sequences have same length."""
        collator = DualStreamCollator()
        batch = create_mock_batch([10, 10, 10])
        
        result = collator(batch)
        
        assert result['mag_inputs'].shape == (3, 10, 5)
        assert result['mod_inputs'].shape == (3, 10, 200)
        assert result['attention_mask'].shape == (3, 10)
        assert result['mask_matrix'].shape == (3, 10)
        assert result['mag_labels'].shape == (3, 10, 5)
        assert result['mod_labels'].shape == (3, 10, 200)


# ==========================================
# 2. Padding Tests
# ==========================================

class TestPadding:
    """Tests for padding behavior."""
    
    def test_padding_with_variable_lengths(self):
        """Test padding with variable sequence lengths."""
        collator = DualStreamCollator()
        batch = create_mock_batch([5, 10, 8])
        
        result = collator(batch)
        
        # All should be padded to max_len=10
        assert result['mag_inputs'].shape == (3, 10, 5)
        assert result['mod_inputs'].shape == (3, 10, 200)
    
    def test_attention_mask_correct(self):
        """Test attention mask correctly marks real tokens."""
        collator = DualStreamCollator()
        batch = create_mock_batch([5, 10, 7])
        
        result = collator(batch)
        
        # Check attention mask sums match sequence lengths
        assert result['attention_mask'][0].sum().item() == 5
        assert result['attention_mask'][1].sum().item() == 10
        assert result['attention_mask'][2].sum().item() == 7
        
        # Check padding positions are 0
        assert result['attention_mask'][0, 5:].sum().item() == 0
        assert result['attention_mask'][2, 7:].sum().item() == 0


# ==========================================
# 3. Masking Tests
# ==========================================

class TestMasking:
    """Tests for dynamic masking behavior."""
    
    def test_mask_only_real_tokens(self):
        """Test that masking only affects real tokens, not padding."""
        collator = DualStreamCollator(mask_prob=0.5)  # High prob for testing
        batch = create_mock_batch([5, 10])
        
        result = collator(batch)
        
        # Mask should never be True where attention_mask is 0 (padding)
        padding_mask = result['attention_mask'] == 0
        invalid_masks = result['mask_matrix'] & padding_mask
        
        assert invalid_masks.sum().item() == 0
    
    def test_masked_positions_are_zeroed(self):
        """Test that masked positions have zero values in inputs."""
        collator = DualStreamCollator(mask_prob=1.0)  # Mask everything
        batch = create_mock_batch([10, 10])
        
        result = collator(batch)
        
        # All unpadded positions should be zeroed
        mask = result['mask_matrix']
        
        # Check mag_inputs are zeroed at masked positions
        masked_mag = result['mag_inputs'][mask]
        assert (masked_mag == 0.0).all()
        
        # Check mod_inputs are zeroed at masked positions
        masked_mod = result['mod_inputs'][mask]
        assert (masked_mod == 0.0).all()
    
    def test_labels_preserved_original_values(self):
        """Test that labels preserve original values (not zeroed)."""
        collator = DualStreamCollator(mask_prob=1.0)
        batch = create_mock_batch([10])
        
        # Store original values
        original_mag = batch[0]['mag_features'].clone()
        
        result = collator(batch)
        
        # Labels should match original (within padding range)
        assert torch.allclose(result['mag_labels'][0], original_mag)
    
    def test_zero_mask_prob(self):
        """Test that mask_prob=0 results in no masking."""
        collator = DualStreamCollator(mask_prob=0.0)
        batch = create_mock_batch([10, 10])
        
        result = collator(batch)
        
        # No positions should be masked
        assert result['mask_matrix'].sum().item() == 0


# ==========================================
# 4. Target Batching Tests
# ==========================================

class TestTargetBatching:
    """Tests for target tensor batching."""
    
    def test_targets_are_padded(self):
        """Test that targets are padded to same length."""
        collator = DualStreamCollator()
        batch = create_mock_batch([5, 10, 7])
        
        result = collator(batch)
        
        # All targets should have max_len=10
        assert result['targets']['mod3'].shape == (3, 10)
        assert result['targets']['mod5'].shape == (3, 10)
        assert result['targets']['mag'].shape == (3, 10)
    
    def test_classification_targets_padded_with_ignore_index(self):
        """Test that classification targets are padded with -100 (ignore index)."""
        collator = DualStreamCollator()
        batch = create_mock_batch([5, 10])
        
        result = collator(batch)
        
        # First sequence has length 5, so positions 5-9 should be -100
        padding_region = result['targets']['mod3'][0, 5:]
        assert (padding_region == -100).all()
    
    def test_regression_targets_padded_with_zero(self):
        """Test that regression targets are padded with 0."""
        collator = DualStreamCollator()
        batch = create_mock_batch([5, 10])
        
        result = collator(batch)
        
        # First sequence has length 5, so positions 5-9 should be 0
        padding_region = result['targets']['mag'][0, 5:]
        assert (padding_region == 0.0).all()
    
    def test_targets_preserve_original_values(self):
        """Test that target values in non-padded region match original."""
        collator = DualStreamCollator()
        batch = create_mock_batch([7])
        
        original_mod3 = batch[0]['targets']['mod3'].clone()
        
        result = collator(batch)
        
        # First 7 positions should match original
        assert torch.equal(result['targets']['mod3'][0, :7], original_mod3)


# ==========================================
# 5. Edge Cases
# ==========================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_item_batch(self):
        """Test with batch size of 1."""
        collator = DualStreamCollator()
        batch = create_mock_batch([15])
        
        result = collator(batch)
        
        assert result['mag_inputs'].shape == (1, 15, 5)
        assert result['mod_inputs'].shape == (1, 15, 200)
    
    def test_very_short_sequences(self):
        """Test with very short sequences."""
        collator = DualStreamCollator()
        batch = create_mock_batch([1, 2, 3])
        
        result = collator(batch)
        
        assert result['mag_inputs'].shape == (3, 3, 5)
        assert result['attention_mask'][0].sum().item() == 1
    
    def test_deterministic_with_seed(self):
        """Test that masking is different between calls (stochastic)."""
        collator = DualStreamCollator(mask_prob=0.5)
        batch = create_mock_batch([100, 100])
        
        torch.manual_seed(42)
        result1 = collator(batch)
        
        torch.manual_seed(123)
        result2 = collator(batch)
        
        # Different seeds should give different masks
        # (very unlikely to be identical with 100 positions and p=0.5)
        assert not torch.equal(result1['mask_matrix'], result2['mask_matrix'])


# ==========================================
# 6. DataLoader Integration Tests
# ==========================================

class TestDataLoaderIntegration:
    """Tests for integration with PyTorch DataLoader."""
    
    def test_with_dataloader(self):
        """Test collator works with DataLoader."""
        from torch.utils.data import DataLoader
        
        # Create a simple list dataset
        data = create_mock_batch([10, 8, 12, 6])
        
        collator = DualStreamCollator()
        loader = DataLoader(data, batch_size=2, collate_fn=collator)
        
        batch_count = 0
        for batch in loader:
            assert 'mag_inputs' in batch
            assert 'mod_inputs' in batch
            assert 'targets' in batch
            batch_count += 1
        
        assert batch_count == 2  # 4 items / 2 batch_size