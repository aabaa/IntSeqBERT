"""
features.py:
Core logic for converting raw integer sequences into model-ready tensors.
Handles Magnitude (Log10-Scale) and Modulo (Sin/Cos) transformations based on Rev.4 specification.
"""

import math
import torch
from typing import List, Dict
from . import config

def compute_magnitude_features(sequence: List[int]) -> torch.Tensor:
    """
    Converts a list of integers into Magnitude features.
    
    Format per number: [log_val, sign_plus, sign_minus, sign_zero]
    - log_val: 1.0 + log10(|x|) if x != 0, else 0.0
    - signs: One-hot-ish encoding for >0, <0, ==0
    
    Returns:
        Tensor of shape (L, config.MAG_RAW_DIM)
    """
    features = []
    
    for x in sequence:
        # 1. Log-Scale Absolute Value (Base-10)
        if x == 0:
            log_val = 0.0
            signs = [0.0, 0.0, 1.0] # Zero
        else:
            val_abs = abs(x)
            
            # Sign Encoding
            if x > 0:
                signs = [1.0, 0.0, 0.0] # Plus
            else:
                signs = [0.0, 1.0, 0.0] # Minus
            
            # Log calculation with overflow protection
            try:
                # Formula: 1.0 + log10(|x|)
                log_val = 1.0 + math.log10(val_abs)
            except OverflowError:
                # Fallback for extremely large integers that exceed float64 range
                # Approx: log10(|x|) ≈ len(str(|x|)) - 1
                # Formula: 1.0 + (len - 1) = len
                log_val = float(len(str(val_abs)))

        # Combine: [Value, S+, S-, S0]
        # Note: config.MAG_RAW_DIM is expected to be 4
        features.append([log_val] + signs)
        
    if not features:
        return torch.zeros((0, config.MAG_RAW_DIM), dtype=torch.float32)
        
    return torch.tensor(features, dtype=torch.float32)

def compute_modulo_features(sequence: List[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a list of integers into Modulo features and Integer labels.
    
    Args:
        sequence: List of integers.
        
    Returns:
        mod_features: (L, MOD_FEATURE_DIM) -> [sin(t1), cos(t1), ...] flattened
        mod_integers: (L, NUM_MODULI)      -> [r1, r2, ...] Raw remainders
    """
    mod_feats_list = []
    mod_ints_list = []
    
    # Constant factor for angle calculation: theta = (2 * pi * r) / m
    two_pi = 2.0 * math.pi
    
    for x in sequence:
        seq_feats = []
        seq_ints = []
        
        # Iterate over all defined moduli (e.g., 2 to 101)
        for m in config.MOD_RANGE:
            # 1. Compute Remainder
            # Python % returns positive remainder for positive divisor
            # Example: -5 % 3 = 1 (mathematically correct)
            r = x % m
            seq_ints.append(r)
            
            # 2. Compute Continuous Embedding (Sin/Cos)
            angle = (two_pi * r) / m
            seq_feats.append(math.sin(angle))
            seq_feats.append(math.cos(angle))
            
        mod_feats_list.append(seq_feats)
        mod_ints_list.append(seq_ints)
        
    if not mod_feats_list:
        return (
            torch.zeros((0, config.MOD_FEATURE_DIM), dtype=torch.float32),
            torch.zeros((0, config.NUM_MODULI), dtype=torch.long)
        )
        
    return (
        torch.tensor(mod_feats_list, dtype=torch.float32),
        torch.tensor(mod_ints_list, dtype=torch.long)
    )

def process_sequence(sequence: List[int]) -> Dict[str, torch.Tensor]:
    """
    Main entry point for processing a single sequence.
    Applies truncation (but NO padding) and converts to tensors.
    
    Args:
        sequence: Raw integer list from OEIS.
        
    Returns:
        Dict containing inputs and labels defined in config keys.
    """
    # 1. Truncation (Handle length limit)
    if len(sequence) > config.MAX_SEQUENCE_LENGTH:
        sequence = sequence[:config.MAX_SEQUENCE_LENGTH]

    # 2. Compute Features
    mag_features = compute_magnitude_features(sequence)
    mod_features, mod_integers = compute_modulo_features(sequence)
    
    # 3. Pack into Dictionary
    return {
        config.KEY_MAG_FEATURES: mag_features,   # (L, 4)
        config.KEY_MOD_FEATURES: mod_features,   # (L, 200)
        config.KEY_MOD_INTEGERS: mod_integers    # (L, 100)
    }
