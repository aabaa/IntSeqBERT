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
        # Use raw integer sequence ("numbers") if available for accurate token IDs.
        # Falls back to magnitude-based approximation if "numbers" not present.
        
        # Special token offsets: PAD=0, MASK=1, UNK=2, integers start at 3
        SPECIAL_TOKENS_OFFSET = 3
        max_int = config.VANILLA_VOCAB_SIZE - SPECIAL_TOKENS_OFFSET - 1
        
        # Check if raw numbers are available
        has_numbers = "numbers" in batch[0]
        
        if has_numbers:
            # Use raw integer sequence for accurate token ID generation
            # This handles negative numbers and avoids rounding errors
            numbers_list = [torch.tensor(item["numbers"], dtype=torch.long) for item in batch]
            numbers_padded = pad_sequence(
                numbers_list, batch_first=True, padding_value=0
            )  # (B, L)
            
            # Map integers to token IDs:
            # - Non-negative integers in [0, max_int] -> [3, vocab_size-1]
            # - Negative integers or out-of-range -> UNK (ID=2)
            in_vocab_mask = (numbers_padded >= 0) & (numbers_padded <= max_int)
            token_ids = torch.where(
                in_vocab_mask,
                numbers_padded + SPECIAL_TOKENS_OFFSET,
                torch.full_like(numbers_padded, config.VANILLA_UNK_TOKEN_ID)
            )
        else:
            # Fallback: Reconstruct approximate integers from log magnitude
            # NOTE: This loses sign information and has rounding errors
            log_vals = mag_padded[..., 0]  # (B, L)
            approx_abs = torch.pow(10.0, log_vals) - 1  # Approximate |value|
            approx_abs = torch.clamp(approx_abs, min=0).long()  # Ensure non-negative
            
            # Map to token IDs
            in_vocab_mask = (approx_abs >= 0) & (approx_abs <= max_int)
            token_ids = torch.where(
                in_vocab_mask,
                approx_abs + SPECIAL_TOKENS_OFFSET,
                torch.full_like(approx_abs, config.VANILLA_UNK_TOKEN_ID)
            )
        
        # Initialize token_labels from token_ids (before masking input)
        token_labels = token_ids.clone()
        
        # Apply MASK token (ID=1) at masked positions for input
        token_ids = torch.where(
            mask_matrix,
            torch.full_like(token_ids, config.VANILLA_MASK_TOKEN_ID),
            token_ids
        )
        
        # Set padding positions to PAD token (ID=0)
        token_ids = torch.where(
            valid_mask_bool,
            token_ids,
            torch.full_like(token_ids, config.VANILLA_PAD_TOKEN_ID)
        )
        
        # Labels: only predict masked positions, set others to IGNORE_INDEX
        token_labels = torch.where(
            mask_matrix,
            token_labels,
            torch.full_like(token_labels, config.IGNORE_INDEX)
        )

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