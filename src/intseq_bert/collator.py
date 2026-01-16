"""
collator.py:
Handles dynamic masking and batch construction for the Dual Stream Architecture.
Implements the 'Mask Flag' strategy to distinguish between valid zeros and masked tokens.
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
    Collator for IntSeqBERT.
    Performs dynamic masking on-the-fly based on config specifications.
    
    Strategies:
    1. Magnitude Stream: Appends a 5th channel 'is_masked'.
       - Unmasked: [val, s+, s-, s0, 0]
       - Masked:   [0,   0,  0,  0,  1]
    2. Modulo Stream: Zeros out Sin/Cos values.
       - Unmasked: [sin, cos, ...]
       - Masked:   [0,   0,   ...] (Origin is distinct from unit circle)
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
        
        # Zero out the content channels (indices 0 to MAG_RAW_DIM-1) at masked positions
        # Create a broadcastable mask for content: 0.0 at masked, 1.0 at unmasked
        content_keep_mask = (~mask_matrix).unsqueeze(-1).type_as(mag_padded) # (B, L, 1)
        
        # Apply mask to the content part only
        mag_inputs[..., :config.MAG_RAW_DIM] = mag_inputs[..., :config.MAG_RAW_DIM] * content_keep_mask
        
        # --- Modulo Stream Processing ---
        # Zero out Sin/Cos at masked positions (Origin shift)
        mod_inputs = mod_padded * content_keep_mask

        # 6. Prepare Labels (Ground Truth)
        # For Magnitude Regression: We need original values. 
        # Loss computation will use 'mask_matrix' to filter relevant positions.
        mag_labels = mag_padded.clone()

        # For Modulo Classification: We need integer labels.
        mod_labels = mod_int_padded.clone()
        # Set label to IGNORE_INDEX where NOT masked (predict only masked tokens)
        mod_labels[~mask_matrix] = config.IGNORE_INDEX

        return {
            "mag_inputs": mag_inputs,           # (B, L, MAG_EXTENDED_DIM)
            "mod_inputs": mod_inputs,           # (B, L, MOD_FEATURE_DIM)
            "mag_labels": mag_labels,           # (B, L, MAG_RAW_DIM)
            "mod_labels": mod_labels,           # (B, L, NUM_MODULI)
            "attention_mask": attention_mask,   # (B, L)
            "mask_matrix": mask_matrix,         # (B, L)
            "oeis_ids": [item.get(config.KEY_OEIS_ID, 'unknown') for item in batch]
        }