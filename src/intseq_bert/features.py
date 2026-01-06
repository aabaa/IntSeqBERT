"""
Feature extraction logic for IntSeqBERT (Dual Model Architecture).
Separates features into 'Mod Spectrum' and 'Magnitude' streams.
Designed for easy unit testing of individual feature components.
"""

import math
import torch
import numpy as np
from typing import List, Dict, Tuple

# ==========================================
# Configuration
# ==========================================
# Mod Spectrum covers cycles from 2 to 101 (100 distinct moduli)
MOD_RANGE = range(2, 102)

# ==========================================
# 1. Magnitude Features (Atomic Functions)
# ==========================================

def compute_log_magnitude(seq: List[int]) -> List[float]:
    """
    Computes log10(|x|). Returns 0.0 for x=0.
    Used as the base for velocity/acceleration and regression targets.
    """
    # Using log10 for easier interpretation (number of digits)
    return [math.log10(abs(x)) if x != 0 else 0.0 for x in seq]

def compute_sign(seq: List[int]) -> List[float]:
    """Computes sign of x: 1.0, -1.0, or 0.0."""
    return [1.0 if x > 0 else (-1.0 if x < 0 else 0.0) for x in seq]

def compute_velocity(seq: List[int]) -> List[float]:
    """
    Computes 1st order difference of Log Magnitude (Growth Rate).
    Pads the first element with 0.0.
    """
    logs = compute_log_magnitude(seq)
    if not logs:
        return []
    
    velocity = [0.0] * len(seq)
    for i in range(1, len(seq)):
        velocity[i] = logs[i] - logs[i-1]
    return velocity

def compute_acceleration(seq: List[int]) -> List[float]:
    """
    Computes 2nd order difference of Log Magnitude (Curvature).
    Distinguishes Exponential (acc=0) vs Factorial (acc>0) vs Polynomial (acc<0).
    Pads the first two elements with 0.0.
    """
    vel = compute_velocity(seq)
    if not vel:
        return []
        
    acc = [0.0] * len(seq)
    for i in range(2, len(seq)):
        acc[i] = vel[i] - vel[i-1]
    return acc

def compute_normalized_index(seq: List[int]) -> List[float]:
    """
    Computes normalized position index [0.0, 1.0].
    Useful for position-dependent sequences (e.g., n^2).
    """
    length = len(seq)
    if length <= 1:
        return [0.0] * length
    return [i / (length - 1) for i in range(length)]

# ==========================================
# 2. Mod Spectrum Features (Atomic Functions)
# ==========================================

def compute_mod_residues(seq: List[int], m: int) -> List[int]:
    """Computes x % m. Used for generating training targets."""
    # Python's % operator handles negative numbers correctly for math mods
    # e.g., -1 % 3 -> 2
    return [x % m for x in seq]

def compute_mod_sin(seq: List[int], m: int) -> List[float]:
    """Computes sin(2*pi * (x % m) / m)."""
    scale = 2 * math.pi / m
    # Optimization: compute residues once if calling both sin/cos, 
    # but kept separate here for unit test independence.
    return [math.sin((x % m) * scale) for x in seq]

def compute_mod_cos(seq: List[int], m: int) -> List[float]:
    """Computes cos(2*pi * (x % m) / m)."""
    scale = 2 * math.pi / m
    return [math.cos((x % m) * scale) for x in seq]

# ==========================================
# 3. Main Extractor
# ==========================================

def extract_features(sequence: List[int]) -> Dict[str, torch.Tensor]:
    """
    Extracts all features for a given sequence using the atomic functions.
    
    Args:
        sequence: List of integers
        
    Returns:
        Dict containing:
        - 'mag_features': (SeqLen, 5) FloatTensor
        - 'mod_features': (SeqLen, 200) FloatTensor (sin/cos pairs for mod 2..101)
        - 'targets': Dict[str, LongTensor] containing true residues for training
    """
    seq_len = len(sequence)
    if seq_len == 0:
        raise ValueError("Sequence cannot be empty")

    # --- 1. Magnitude Features (5 dim) ---
    f_log = compute_log_magnitude(sequence)
    f_vel = compute_velocity(sequence)
    f_acc = compute_acceleration(sequence)
    f_sgn = compute_sign(sequence)
    f_idx = compute_normalized_index(sequence)
    
    # Stack: (SeqLen, 5)
    mag_data = [f_log, f_vel, f_acc, f_sgn, f_idx]
    # Transpose to (SeqLen, 5)
    mag_features = torch.tensor(mag_data, dtype=torch.float32).t()

    # --- 2. Mod Spectrum Features (200 dim) & Targets ---
    mod_data = []
    targets = {}
    
    for m in MOD_RANGE:
        # Features
        mod_data.append(compute_mod_sin(sequence, m))
        mod_data.append(compute_mod_cos(sequence, m))
        
        # Targets (for loss calculation)
        targets[f"mod{m}"] = torch.tensor(
            compute_mod_residues(sequence, m), 
            dtype=torch.long
        )
    
    # Stack features: (SeqLen, 200)
    mod_features = torch.tensor(mod_data, dtype=torch.float32).t()
    
    # Add magnitude regression target (same as f_log but separated for clarity)
    targets["mag"] = torch.tensor(f_log, dtype=torch.float32)

    return {
        "mag_features": mag_features,
        "mod_features": mod_features,
        "targets": targets
    }