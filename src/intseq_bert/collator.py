"""
collator.py:
Handles dynamic masking and batch construction for the Dual Stream Architecture.
Implements the 'Mask Flag' strategy to distinguish between valid zeros and masked tokens.
Also provides token_ids for Vanilla Transformer compatibility.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import List, Dict, Any

# Centralized configuration
from . import config


def integer_to_token_id(value: int) -> int:
    """
    Convert an integer value to a token ID for Vanilla Transformer.
    
    Token ID mapping:
    - 0: PAD token (reserved)
    - 1: MASK token (reserved)
    - 2: UNK token (for out-of-vocabulary integers)
    - 3 onwards: Actual integer values (shifted by 3)
    
    For integers in [0, VOCAB_SIZE-4], we map to [3, VOCAB_SIZE-1].
    Integers outside this range become UNK.
    
    Args:
        value: Integer value to convert
        
    Returns:
        Token ID
    """
    # Special token offsets
    SPECIAL_TOKENS_OFFSET = 3  # PAD=0, MASK=1, UNK=2
    
    # Usable vocab range for integers
    max_int = config.VANILLA_VOCAB_SIZE - SPECIAL_TOKENS_OFFSET - 1
    
    # Map to token ID
    if 0 <= value <= max_int:
        return value + SPECIAL_TOKENS_OFFSET
    else:
        # Out of vocabulary -> UNK
        return config.VANILLA_UNK_TOKEN_ID


@dataclass
class OEISCollator:
    """
    Collator for IntSeqBERT and Vanilla Transformer.
    Performs dynamic masking on-the-fly based on config specifications.
    
    Strategies:
    1. Magnitude Stream: Appends a 5th channel 'is_masked'.
       - Unmasked: [val, s+, s-, s0, 0]
       - Masked:   [0,   0,  0,  0,  1]
    2. Modulo Stream: Zeros out Sin/Cos values.
       - Unmasked: [sin, cos, ...]
       - Masked:   [0,   0,   ...] (Origin is distinct from unit circle)
    3. Token IDs (for Vanilla): Integer values converted to token IDs.
       - Masked positions use MASK token (ID=1)
    """
    mask_prob: float = config.MASK_PROB

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Args:
            batch: List of dicts from OEISDataset. Expected keys are defined in config:
                   - KEY_MAG_FEATURES: (L, MAG_RAW_DIM)
                   - KEY_MOD_FEATURES: (L, MOD_FEATURE_DIM)
                   - KEY_MOD_INTEGERS: (L, NUM_MODULI)
                   - KEY_OEIS_ID: str
        """
        if not batch:
            raise ValueError("Batch is empty.")

        # 1. Extract streams
        # Verify required keys strictly
        required_keys = [config.KEY_MAG_FEATURES, config.KEY_MOD_FEATURES, config.KEY_MOD_INTEGERS]
        for key in required_keys:
            if key not in batch[0]:
                raise KeyError(f"Dataset must provide '{key}' for collator.")

        mag_list = [item[config.KEY_MAG_FEATURES] for item in batch]
        mod_list = [item[config.KEY_MOD_FEATURES] for item in batch]
        mod_int_list = [item[config.KEY_MOD_INTEGERS] for item in batch]

        # 2. Padding
        # Features are padded with sentinel value (config.PAD_VALUE_FEATURE)
        mag_padded = pad_sequence(mag_list, batch_first=True, padding_value=config.PAD_VALUE_FEATURE)
        mod_padded = pad_sequence(mod_list, batch_first=True, padding_value=config.PAD_VALUE_FEATURE)
        
        # Labels (integers) are padded with config.IGNORE_INDEX
        mod_int_padded = pad_sequence(mod_int_list, batch_first=True, padding_value=int(config.IGNORE_INDEX))

        batch_size, max_len, _ = mag_padded.size()

        # 3. Create Attention Mask (Valid positions = 1, Padding = 0)
        # Identify padding by checking against sentinel value
        # Note: We check the first channel [0] which is the log value
        valid_mask_bool = (mag_padded[..., 0] != config.PAD_VALUE_FEATURE)
        
        attention_mask = valid_mask_bool.long()

        # 4. Generate Mask Matrix (Bernoulli sampling)
        prob_matrix = torch.full((batch_size, max_len), self.mask_prob)
        # Do NOT mask padding tokens
        prob_matrix[~valid_mask_bool] = 0.0
        
        mask_matrix = torch.bernoulli(prob_matrix).bool() # (B, L)
        
        # 5. Prepare Inputs (Apply Masking)
        
        # --- Magnitude Stream Processing ---
        # Current shape: (B, L, MAG_RAW_DIM). Goal: (B, L, MAG_EXTENDED_DIM)
        # The last channel is the 'is_masked' flag.
        
        # Create the mask flag channel (B, L, 1)
        # Initialize with 0.0, set to 1.0 where masked
        is_masked_channel = torch.zeros((batch_size, max_len, 1), dtype=mag_padded.dtype)
        is_masked_channel[mask_matrix] = 1.0
        
        # Concatenate to form the extended vector: [Content, MaskFlag]
        mag_inputs = torch.cat([mag_padded, is_masked_channel], dim=2)
        
        # Zero out the content channels (indices 0 to MAG_RAW_DIM-1) at masked OR padding positions
        # This prevents sentinel values (-9999.0) from flowing into the embedding layer
        # Create broadcastable mask: 0.0 at masked/padding, 1.0 at valid unmasked
        valid_unmasked = valid_mask_bool & (~mask_matrix)  # True only for valid AND unmasked
        content_keep_mask = valid_unmasked.unsqueeze(-1).type_as(mag_padded) # (B, L, 1)
        
        # Apply mask to the content part only
        mag_inputs[..., :config.MAG_RAW_DIM] = mag_inputs[..., :config.MAG_RAW_DIM] * content_keep_mask
        
        # --- Modulo Stream Processing ---
        # Zero out Sin/Cos at masked positions (Origin shift)
        mod_inputs = mod_padded * content_keep_mask

        # --- Token ID Processing (for Vanilla Transformer) ---
        # Extract original integer values from the sequence
        # We need to get raw integers - they should be stored as the first modulo residue (mod 2)
        # Actually, we need the original integers which are in features.py KEY_INTEGERS
        # For now, we'll use the log magnitude to derive approximate integer values
        # BUT: The proper approach is to have raw integers in the dataset
        
        # Token IDs: We'll generate from the magnitude head's target
        # The mag_features[..., 0] contains log10(|value|+1)
        # We can reconstruct approximate |value| from this for tokenization
        # However, this loses sign and precision. Better approach: pass raw integers
        
        # For now, use a simple mapping based on modulo integers (first column is value mod 2)
        # This is a workaround - ideally we'd have raw integers in the dataset
        
        # Build token_ids tensor
        token_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        token_labels = torch.full((batch_size, max_len), config.IGNORE_INDEX, dtype=torch.long)
        
        # For each item in batch, we need raw integers
        # Check if raw integers are available
        if "integers" in batch[0]:
            # Raw integers available
            for b_idx, item in enumerate(batch):
                integers = item["integers"]
                seq_len = len(integers)
                for i, val in enumerate(integers):
                    if i < max_len:
                        token_id = integer_to_token_id(int(val))
                        token_ids[b_idx, i] = token_id
                        # Labels: save original token_id for loss computation
                        token_labels[b_idx, i] = token_id
        else:
            # No raw integers - use a placeholder approach
            # Set all valid positions to UNK (2) as fallback
            token_ids[valid_mask_bool] = config.VANILLA_UNK_TOKEN_ID
            token_labels[valid_mask_bool] = config.VANILLA_UNK_TOKEN_ID
        
        # Apply MASK token (ID=1) at masked positions for input
        MASK_TOKEN_ID = 1
        token_ids[mask_matrix] = MASK_TOKEN_ID
        
        # Set padding positions to PAD token (ID=0)
        token_ids[~valid_mask_bool] = config.VANILLA_PAD_TOKEN_ID
        
        # Labels: only predict masked positions
        token_labels[~mask_matrix] = config.IGNORE_INDEX

        # 6. Prepare Labels (Ground Truth)
        # For Magnitude Regression: We need original values. 
        # Loss computation will use 'mask_matrix' to filter relevant positions.
        mag_labels = mag_padded.clone()

        # For Modulo Classification: We need integer labels.
        mod_labels = mod_int_padded.clone()
        # Set label to IGNORE_INDEX where NOT masked (predict only masked tokens)
        mod_labels[~mask_matrix] = config.IGNORE_INDEX

        return {
            # IntSeqBERT inputs
            "mag_inputs": mag_inputs,           # (B, L, MAG_EXTENDED_DIM)
            "mod_inputs": mod_inputs,           # (B, L, MOD_FEATURE_DIM)
            "mag_labels": mag_labels,           # (B, L, MAG_RAW_DIM)
            "mod_labels": mod_labels,           # (B, L, NUM_MODULI)
            # Vanilla Transformer inputs
            "token_ids": token_ids,             # (B, L) LongTensor
            "token_labels": token_labels,       # (B, L) LongTensor (targets for LM)
            # Common
            "attention_mask": attention_mask,   # (B, L)
            "mask_matrix": mask_matrix,         # (B, L)
            "oeis_ids": [item.get(config.KEY_OEIS_ID, 'unknown') for item in batch]
        }