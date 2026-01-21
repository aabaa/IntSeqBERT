"""
Tests for OEISCollator module.

Covers:
1. Input validation (required keys)
2. Padding behavior
3. Attention mask generation
4. Dynamic masking
5. Magnitude stream processing (dimension extension + mask flag)
6. Modulo stream processing (origin shift)
7. Label preparation
8. Output shape verification
9. Edge cases
"""

import pytest
import torch
from torch.utils.data import DataLoader

from intseq_bert.collator import OEISCollator
from intseq_bert import config


# ==========================================
# Helper Functions
# ==========================================

def create_mock_item(oeis_id: str, length: int) -> dict:
    """Create a single mock data item matching OEISDataset output contract."""
    return {
        config.KEY_OEIS_ID: oeis_id,
        config.KEY_MAG_FEATURES: torch.randn(length, config.MAG_RAW_DIM),
        config.KEY_MOD_FEATURES: torch.randn(length, config.MOD_FEATURE_DIM),
        config.KEY_MOD_INTEGERS: torch.randint(0, 50, (length, config.NUM_MODULI), dtype=torch.long),
    }


def create_mock_batch(seq_lengths: list) -> list:
    """Create a mock batch with variable sequence lengths."""
    return [create_mock_item(f"A{i:06d}", length) for i, length in enumerate(seq_lengths)]


# ==========================================
# 1. Input Validation Tests
# ==========================================

class TestInputValidation:
    """Tests for input data validation."""
    
    def test_empty_batch_raises_error(self):
        """Test that empty batch raises ValueError."""
        collator = OEISCollator()
        with pytest.raises(ValueError, match="Batch is empty"):
            collator([])
    
    def test_missing_mag_features_raises_error(self):
        """Test that missing mag_features raises KeyError."""
        collator = OEISCollator()
        batch = [{
            config.KEY_MOD_FEATURES: torch.randn(5, config.MOD_FEATURE_DIM),
            config.KEY_MOD_INTEGERS: torch.randint(0, 10, (5, config.NUM_MODULI)),
        }]
        with pytest.raises(KeyError, match=config.KEY_MAG_FEATURES):
            collator(batch)
    
    def test_missing_mod_features_raises_error(self):
        """Test that missing mod_features raises KeyError."""
        collator = OEISCollator()
        batch = [{
            config.KEY_MAG_FEATURES: torch.randn(5, config.MAG_RAW_DIM),
            config.KEY_MOD_INTEGERS: torch.randint(0, 10, (5, config.NUM_MODULI)),
        }]
        with pytest.raises(KeyError, match=config.KEY_MOD_FEATURES):
            collator(batch)
    
    def test_missing_mod_integers_raises_error(self):
        """Test that missing mod_integers raises KeyError."""
        collator = OEISCollator()
        batch = [{
            config.KEY_MAG_FEATURES: torch.randn(5, config.MAG_RAW_DIM),
            config.KEY_MOD_FEATURES: torch.randn(5, config.MOD_FEATURE_DIM),
        }]
        with pytest.raises(KeyError, match=config.KEY_MOD_INTEGERS):
            collator(batch)


# ==========================================
# 2. Output Shape Tests
# ==========================================

class TestOutputShapes:
    """Tests for output tensor shapes."""
    
    def test_output_keys(self):
        """Test that collator returns all expected keys."""
        collator = OEISCollator()
        batch = create_mock_batch([10, 10])
        result = collator(batch)
        
        expected_keys = [
            "mag_inputs", "mod_inputs", "mag_labels", "mod_labels",
            "token_ids", "token_labels",  # Vanilla Transformer support
            "attention_mask", "mask_matrix", "oeis_ids"
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_shapes_same_length_sequences(self):
        """Test output shapes when all sequences have same length."""
        collator = OEISCollator()
        batch = create_mock_batch([10, 10, 10])
        result = collator(batch)
        
        B, L = 3, 10
        assert result["mag_inputs"].shape == (B, L, config.MAG_EXTENDED_DIM)
        assert result["mod_inputs"].shape == (B, L, config.MOD_FEATURE_DIM)
        assert result["mag_labels"].shape == (B, L, config.MAG_RAW_DIM)
        assert result["mod_labels"].shape == (B, L, config.NUM_MODULI)
        assert result["attention_mask"].shape == (B, L)
        assert result["mask_matrix"].shape == (B, L)
        assert len(result["oeis_ids"]) == B
    
    def test_shapes_variable_length_sequences(self):
        """Test output shapes with variable length sequences (padding)."""
        collator = OEISCollator()
        batch = create_mock_batch([5, 10, 7])
        result = collator(batch)
        
        B, L = 3, 10  # L = max(5, 10, 7)
        assert result["mag_inputs"].shape == (B, L, config.MAG_EXTENDED_DIM)
        assert result["mod_inputs"].shape == (B, L, config.MOD_FEATURE_DIM)


# ==========================================
# 3. Attention Mask Tests
# ==========================================

class TestAttentionMask:
    """Tests for attention mask generation."""
    
    def test_attention_mask_values(self):
        """Test that attention mask correctly marks valid positions."""
        collator = OEISCollator()
        batch = create_mock_batch([5, 10, 7])
        result = collator(batch)
        
        # Check sums match sequence lengths
        assert result["attention_mask"][0].sum().item() == 5
        assert result["attention_mask"][1].sum().item() == 10
        assert result["attention_mask"][2].sum().item() == 7
    
    def test_attention_mask_padding_positions(self):
        """Test that padding positions have attention_mask = 0."""
        collator = OEISCollator()
        batch = create_mock_batch([5, 10])
        result = collator(batch)
        
        # Sequence 0: length 5, positions 5-9 should be 0
        assert result["attention_mask"][0, 5:].sum().item() == 0
        # Sequence 1: length 10, no padding
        assert result["attention_mask"][1, :].sum().item() == 10


# ==========================================
# 4. Dynamic Masking Tests
# ==========================================

class TestDynamicMasking:
    """Tests for dynamic masking behavior."""
    
    def test_padding_never_masked(self):
        """Test that padding positions are never masked."""
        collator = OEISCollator()
        collator.mask_prob = 0.5  # High prob for testing
        batch = create_mock_batch([5, 10])
        result = collator(batch)
        
        # Mask should never be True where attention_mask is 0
        padding_positions = result["attention_mask"] == 0
        invalid_masks = result["mask_matrix"] & padding_positions
        assert invalid_masks.sum().item() == 0
    
    def test_zero_mask_prob_no_masking(self):
        """Test that mask_prob=0 results in no masking."""
        collator = OEISCollator()
        collator.mask_prob = 0.0
        batch = create_mock_batch([10, 10])
        result = collator(batch)
        
        assert result["mask_matrix"].sum().item() == 0
    
    def test_full_mask_prob_all_masked(self):
        """Test that mask_prob=1.0 masks all valid tokens."""
        collator = OEISCollator()
        collator.mask_prob = 1.0
        batch = create_mock_batch([5, 10])
        result = collator(batch)
        
        # Total masked should equal total valid tokens (5 + 10 = 15)
        assert result["mask_matrix"].sum().item() == 15


# ==========================================
# 5. Magnitude Stream Processing Tests
# ==========================================

class TestMagnitudeStream:
    """Tests for magnitude stream dimension extension."""
    
    def test_dimension_extended(self):
        """Test that mag_inputs has extended dimension."""
        collator = OEISCollator()
        batch = create_mock_batch([10])
        result = collator(batch)
        
        # Should be MAG_EXTENDED_DIM = MAG_RAW_DIM + 1
        assert result["mag_inputs"].shape[2] == config.MAG_EXTENDED_DIM
        assert result["mag_inputs"].shape[2] == config.MAG_RAW_DIM + 1
    
    def test_unmasked_positions_flag_zero(self):
        """Test that unmasked positions have is_masked flag = 0."""
        collator = OEISCollator()
        collator.mask_prob = 0.0  # No masking
        batch = create_mock_batch([10])
        result = collator(batch)
        
        # Last channel (is_masked) should be 0 everywhere
        is_masked_channel = result["mag_inputs"][..., -1]
        assert (is_masked_channel == 0.0).all()
    
    def test_masked_positions_flag_one(self):
        """Test that masked positions have is_masked flag = 1."""
        collator = OEISCollator()
        collator.mask_prob = 1.0  # Mask all
        batch = create_mock_batch([10])
        result = collator(batch)
        
        # Last channel (is_masked) should be 1 for all valid positions
        is_masked_channel = result["mag_inputs"][0, :, -1]
        assert (is_masked_channel == 1.0).all()
    
    def test_masked_positions_content_zeroed(self):
        """Test that masked positions have content channels zeroed."""
        collator = OEISCollator()
        collator.mask_prob = 1.0
        batch = create_mock_batch([10])
        result = collator(batch)
        
        # Content channels (0 to MAG_RAW_DIM-1) should be 0 at masked positions
        content_channels = result["mag_inputs"][0, :, :config.MAG_RAW_DIM]
        assert (content_channels == 0.0).all()
    
    def test_unmasked_positions_preserve_content(self):
        """Test that unmasked positions preserve original content."""
        collator = OEISCollator()
        collator.mask_prob = 0.0
        batch = create_mock_batch([10])
        original_mag = batch[0][config.KEY_MAG_FEATURES].clone()
        result = collator(batch)
        
        # Content channels should match original
        content_channels = result["mag_inputs"][0, :10, :config.MAG_RAW_DIM]
        assert torch.allclose(content_channels, original_mag)


# ==========================================
# 6. Modulo Stream Processing Tests
# ==========================================

class TestModuloStream:
    """Tests for modulo stream origin shift."""
    
    def test_masked_positions_zeroed(self):
        """Test that masked positions in mod_inputs are zeroed (origin shift)."""
        collator = OEISCollator()
        collator.mask_prob = 1.0
        batch = create_mock_batch([10])
        result = collator(batch)
        
        # All valid positions should be zeroed
        mod_inputs = result["mod_inputs"][0, :10, :]
        assert (mod_inputs == 0.0).all()
    
    def test_unmasked_positions_preserve_values(self):
        """Test that unmasked positions preserve original values."""
        collator = OEISCollator()
        collator.mask_prob = 0.0
        batch = create_mock_batch([10])
        original_mod = batch[0][config.KEY_MOD_FEATURES].clone()
        result = collator(batch)
        
        # Should match original
        assert torch.allclose(result["mod_inputs"][0, :10, :], original_mod)


# ==========================================
# 7. Label Preparation Tests
# ==========================================

class TestLabelPreparation:
    """Tests for mag_labels and mod_labels."""
    
    def test_mag_labels_preserve_original(self):
        """Test that mag_labels preserve original feature values."""
        collator = OEISCollator()
        collator.mask_prob = 1.0
        batch = create_mock_batch([10])
        original_mag = batch[0][config.KEY_MAG_FEATURES].clone()
        result = collator(batch)
        
        # Labels should match original (even for masked positions)
        assert torch.allclose(result["mag_labels"][0, :10, :], original_mag)
    
    def test_mod_labels_unmasked_ignored(self):
        """Test that unmasked positions in mod_labels have IGNORE_INDEX."""
        collator = OEISCollator()
        collator.mask_prob = 0.0  # No masking
        batch = create_mock_batch([10])
        result = collator(batch)
        
        # All positions should be IGNORE_INDEX (since none are masked)
        assert (result["mod_labels"] == config.IGNORE_INDEX).all()
    
    def test_mod_labels_masked_preserve_values(self):
        """Test that masked positions in mod_labels preserve original integers."""
        collator = OEISCollator()
        collator.mask_prob = 1.0  # Mask all
        batch = create_mock_batch([10])
        original_mod_int = batch[0][config.KEY_MOD_INTEGERS].clone()
        result = collator(batch)
        
        # Masked positions should have original integer values
        assert torch.equal(result["mod_labels"][0, :10, :], original_mod_int)
    
    def test_mod_labels_padding_ignored(self):
        """Test that padding positions in mod_labels have IGNORE_INDEX."""
        collator = OEISCollator()
        collator.mask_prob = 1.0
        batch = create_mock_batch([5, 10])
        result = collator(batch)
        
        # Sequence 0: padding at positions 5-9
        padding_region = result["mod_labels"][0, 5:, :]
        assert (padding_region == config.IGNORE_INDEX).all()


# ==========================================
# 8. Token ID Processing Tests (Vanilla Transformer)
# ==========================================

class TestTokenIdProcessing:
    """Tests for token_ids and token_labels generation for Vanilla Transformer."""
    
    def test_token_ids_shape(self):
        """Test that token_ids has correct shape."""
        collator = OEISCollator()
        batch = create_mock_batch([10, 8, 12])
        result = collator(batch)
        
        B, L = 3, 12  # L = max length
        assert result["token_ids"].shape == (B, L)
        assert result["token_labels"].shape == (B, L)
    
    def test_token_ids_dtype(self):
        """Test that token_ids is LongTensor."""
        collator = OEISCollator()
        batch = create_mock_batch([10])
        result = collator(batch)
        
        assert result["token_ids"].dtype == torch.long
        assert result["token_labels"].dtype == torch.long
    
    def test_token_ids_valid_range(self):
        """Test that token_ids are within valid vocabulary range."""
        collator = OEISCollator()
        batch = create_mock_batch([10, 10])
        result = collator(batch)
        
        # All token IDs should be in [0, VANILLA_VOCAB_SIZE)
        assert (result["token_ids"] >= 0).all()
        assert (result["token_ids"] < config.VANILLA_VOCAB_SIZE).all()
    
    def test_masked_positions_have_mask_token(self):
        """Test that masked positions have MASK token (ID=1)."""
        collator = OEISCollator()
        collator.mask_prob = 1.0  # Mask all
        batch = create_mock_batch([10])
        result = collator(batch)
        
        # All valid positions should have MASK token
        valid_positions = result["attention_mask"][0, :10] == 1
        token_ids_valid = result["token_ids"][0, :10]
        assert (token_ids_valid == config.VANILLA_MASK_TOKEN_ID).all()
    
    def test_padding_positions_have_pad_token(self):
        """Test that padding positions have PAD token (ID=0)."""
        collator = OEISCollator()
        batch = create_mock_batch([5, 10])
        result = collator(batch)
        
        # Sequence 0: positions 5-9 are padding
        padding_tokens = result["token_ids"][0, 5:]
        assert (padding_tokens == config.VANILLA_PAD_TOKEN_ID).all()
    
    def test_unmasked_positions_have_valid_tokens(self):
        """Test that unmasked positions have valid token IDs (not MASK or PAD)."""
        collator = OEISCollator()
        collator.mask_prob = 0.0  # No masking
        batch = create_mock_batch([10])
        result = collator(batch)
        
        # Valid positions should have token IDs >= 2 (not PAD=0, not MASK=1)
        valid_tokens = result["token_ids"][0, :10]
        assert (valid_tokens >= config.VANILLA_UNK_TOKEN_ID).all()  # >= 2
    
    def test_token_labels_masked_only(self):
        """Test that token_labels has IGNORE_INDEX for non-masked positions."""
        collator = OEISCollator()
        collator.mask_prob = 0.0  # No masking
        batch = create_mock_batch([10])
        result = collator(batch)
        
        # No masked positions, so all labels should be IGNORE_INDEX
        assert (result["token_labels"] == config.IGNORE_INDEX).all()
    
    def test_token_labels_preserve_for_masked(self):
        """Test that masked positions have valid target token IDs in labels."""
        collator = OEISCollator()
        collator.mask_prob = 1.0  # Mask all
        batch = create_mock_batch([10])
        result = collator(batch)
        
        # Masked positions should have valid token IDs (not IGNORE_INDEX)
        masked_labels = result["token_labels"][0, :10]
        assert (masked_labels != config.IGNORE_INDEX).all()
        assert (masked_labels >= config.VANILLA_UNK_TOKEN_ID).all()  # >= 2
    
    def test_token_ids_from_raw_numbers(self):
        """Test that token_ids are correctly generated from raw numbers."""
        collator = OEISCollator()
        collator.mask_prob = 0.0  # No masking
        
        # Create batch with known numbers
        batch = [{
            config.KEY_OEIS_ID: "A000001",
            config.KEY_MAG_FEATURES: torch.randn(5, config.MAG_RAW_DIM),
            config.KEY_MOD_FEATURES: torch.randn(5, config.MOD_FEATURE_DIM),
            config.KEY_MOD_INTEGERS: torch.randint(0, 50, (5, config.NUM_MODULI), dtype=torch.long),
            "numbers": [0, 1, 5, 10, 100],
        }]
        result = collator(batch)
        
        # Token IDs should be numbers + 3 (offset for special tokens)
        expected = torch.tensor([3, 4, 8, 13, 103])  # 0+3, 1+3, 5+3, 10+3, 100+3
        assert torch.equal(result["token_ids"][0, :5], expected)
    
    def test_token_ids_negative_numbers_become_unk(self):
        """Test that negative numbers are mapped to UNK token."""
        collator = OEISCollator()
        collator.mask_prob = 0.0
        
        batch = [{
            config.KEY_OEIS_ID: "A000002",
            config.KEY_MAG_FEATURES: torch.randn(4, config.MAG_RAW_DIM),
            config.KEY_MOD_FEATURES: torch.randn(4, config.MOD_FEATURE_DIM),
            config.KEY_MOD_INTEGERS: torch.randint(0, 50, (4, config.NUM_MODULI), dtype=torch.long),
            "numbers": [-5, 0, -1, 10],
        }]
        result = collator(batch)
        
        # Negative numbers should become UNK (ID=2)
        assert result["token_ids"][0, 0].item() == config.VANILLA_UNK_TOKEN_ID  # -5 -> UNK
        assert result["token_ids"][0, 1].item() == 3  # 0 -> 3
        assert result["token_ids"][0, 2].item() == config.VANILLA_UNK_TOKEN_ID  # -1 -> UNK
        assert result["token_ids"][0, 3].item() == 13  # 10 -> 13
    
    def test_token_ids_out_of_vocab_become_unk(self):
        """Test that out-of-vocabulary integers become UNK."""
        collator = OEISCollator()
        collator.mask_prob = 0.0
        
        max_valid_int = config.VANILLA_VOCAB_SIZE - 3 - 1  # max_int
        
        batch = [{
            config.KEY_OEIS_ID: "A000003",
            config.KEY_MAG_FEATURES: torch.randn(3, config.MAG_RAW_DIM),
            config.KEY_MOD_FEATURES: torch.randn(3, config.MOD_FEATURE_DIM),
            config.KEY_MOD_INTEGERS: torch.randint(0, 50, (3, config.NUM_MODULI), dtype=torch.long),
            "numbers": [0, max_valid_int, max_valid_int + 1],
        }]
        result = collator(batch)
        
        # 0 -> 3, max_valid_int -> valid, max_valid_int + 1 -> UNK
        assert result["token_ids"][0, 0].item() == 3
        assert result["token_ids"][0, 1].item() == max_valid_int + 3
        assert result["token_ids"][0, 2].item() == config.VANILLA_UNK_TOKEN_ID
    
    def test_token_ids_huge_integers_become_unk(self):
        """Test that integers exceeding int64 range become UNK without overflow."""
        collator = OEISCollator()
        collator.mask_prob = 0.0
        
        # OEIS contains arbitrarily large integers (e.g., 10^100)
        huge_int = 10**100  # Far exceeds int64 range
        
        batch = [{
            config.KEY_OEIS_ID: "A000004",
            config.KEY_MAG_FEATURES: torch.randn(3, config.MAG_RAW_DIM),
            config.KEY_MOD_FEATURES: torch.randn(3, config.MOD_FEATURE_DIM),
            config.KEY_MOD_INTEGERS: torch.randint(0, 50, (3, config.NUM_MODULI), dtype=torch.long),
            "numbers": [0, huge_int, -huge_int],
        }]
        
        # Should not raise overflow error
        result = collator(batch)
        
        # First value (0) -> token 3
        assert result["token_ids"][0, 0].item() == 3
        # Large positive and negative integers -> UNK
        assert result["token_ids"][0, 1].item() == config.VANILLA_UNK_TOKEN_ID
        assert result["token_ids"][0, 2].item() == config.VANILLA_UNK_TOKEN_ID


# ==========================================
# 8. Edge Cases
# ==========================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_item_batch(self):
        """Test with batch size of 1."""
        collator = OEISCollator()
        batch = create_mock_batch([15])
        result = collator(batch)
        
        assert result["mag_inputs"].shape == (1, 15, config.MAG_EXTENDED_DIM)
    
    def test_very_short_sequences(self):
        """Test with very short sequences."""
        collator = OEISCollator()
        batch = create_mock_batch([1, 2, 3])
        result = collator(batch)
        
        assert result["mag_inputs"].shape == (3, 3, config.MAG_EXTENDED_DIM)
        assert result["attention_mask"][0].sum().item() == 1
    
    def test_oeis_ids_extracted(self):
        """Test that oeis_ids are correctly extracted."""
        collator = OEISCollator()
        batch = create_mock_batch([5, 10])
        result = collator(batch)
        
        assert result["oeis_ids"] == ["A000000", "A000001"]
    
    def test_missing_oeis_id_defaults_to_unknown(self):
        """Test that missing oeis_id defaults to 'unknown'."""
        collator = OEISCollator()
        batch = [{
            config.KEY_MAG_FEATURES: torch.randn(5, config.MAG_RAW_DIM),
            config.KEY_MOD_FEATURES: torch.randn(5, config.MOD_FEATURE_DIM),
            config.KEY_MOD_INTEGERS: torch.randint(0, 10, (5, config.NUM_MODULI)),
        }]
        result = collator(batch)
        
        assert result["oeis_ids"] == ["unknown"]


# ==========================================
# 9. DataLoader Integration
# ==========================================

class TestDataLoaderIntegration:
    """Tests for integration with PyTorch DataLoader."""
    
    def test_with_dataloader(self):
        """Test collator works correctly with DataLoader."""
        data = create_mock_batch([10, 8, 12, 6])
        collator = OEISCollator()
        loader = DataLoader(data, batch_size=2, collate_fn=collator)
        
        batch_count = 0
        for batch in loader:
            assert "mag_inputs" in batch
            assert "mod_inputs" in batch
            assert "mag_labels" in batch
            assert "mod_labels" in batch
            assert batch["mag_inputs"].shape[0] == 2  # batch_size
            batch_count += 1
        
        assert batch_count == 2  # 4 items / 2 = 2 batches