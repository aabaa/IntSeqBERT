# src/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import math
import os
from typing import List, Optional

# Import functions from utils
from .utils import (
    is_prime, is_square, is_square_free, valuation, 
    popcount, digit_sum
)

class OEISDataset(Dataset):
    def __init__(self, data_path: Optional[str] = None, names_path: Optional[str] = None, seq_len: int = 128):
        self.seq_len = seq_len
        self.sequences = []
        
        # Production data loading logic (only if file exists)
        if data_path and os.path.exists(data_path):
            self._load_data(data_path, names_path)
        else:
            # Dummy data for testing/debugging
            self.sequences = self._generate_dummy_data()

    def _load_data(self, data_path, names_path):
        """Load and filter OEIS data"""
        ban_set = set()
        if names_path and os.path.exists(names_path):
            try:
                with open(names_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.startswith("#"): continue
                        lower = line.lower()
                        # Noise removal: constant expansions and error data
                        if "decimal expansion" in lower or "dead sequence" in lower or "erroneous" in lower:
                            ban_set.add(line[:7]) # A000000
            except Exception as e:
                print(f"Warning: Failed to load names file: {e}")

        with open(data_path, 'r') as f:
            for line in f:
                if len(line) < 10: continue
                parts = line.strip().split(',')
                if len(parts) < 2: continue
                
                oeis_id = parts[0]
                if oeis_id in ban_set: continue
                
                try:
                    # Large integers may cause overflow, so limit to a certain digit count
                    # Python's int automatically extends, so only consider calculation cost
                    seq = []
                    for x in parts[1:]:
                        val = int(x)
                        # Skip extremely large numbers to avoid errors in feature calculation
                        if abs(val) < 10**100: 
                            seq.append(val)
                    
                    if len(seq) >= 10: # Minimum length
                        self.sequences.append(seq)
                except ValueError:
                    continue

    def _generate_dummy_data(self):
        """Development dummy data generation"""
        seqs = []
        # Arithmetic sequence
        seqs.append([i for i in range(50)])
        # Fibonacci sequence
        fib = [1, 1]
        for _ in range(48): fib.append(fib[-1] + fib[-2])
        seqs.append(fib)
        # Prime numbers (simple)
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        seqs.append(primes * 5)
        return seqs

    def process_seq(self, seq: List[int]) -> torch.Tensor:
        """
        Convert integer sequence to 24-dimensional feature sequence
        Returns: Tensor [SeqLen, 24]
        """
        features = []
        prev_log = 0.0
        prev_diff = 0.0
        prev_val = 0
        
        for i, x in enumerate(seq):
            x_abs = abs(x)
            
            # --- 1. Analytic (5 dims) ---
            # Log Magnitude
            log_val = math.log1p(x_abs) if x_abs > 0 else 0.0
            
            # Diff1 (Velocity)
            diff1 = log_val - prev_log
            
            # Diff2 (Acceleration)
            diff2 = diff1 - prev_diff if i > 0 else 0.0
            
            # Sign
            sign = 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
            
            # Direction
            if i > 0:
                raw_diff = x - prev_val
                dir_sign = 1.0 if raw_diff > 0 else (-1.0 if raw_diff < 0 else 0.0)
            else:
                dir_sign = 0.0

            # Update history
            prev_log = log_val
            prev_diff = diff1
            prev_val = x
            
            # --- 2. Algebraic (10 dims) ---
            # Mod 2, 3, 5, 7, 8 Embedding (Sin/Cos)
            mods = []
            for p in [2, 3, 5, 7, 8]:
                # Circular embedding: 0 ~ 2pi
                phase = 2 * math.pi * (x % p) / p
                mods.append(math.sin(phase))
                mods.append(math.cos(phase))
            
            # --- 3. Number Theoretic (7 dims) ---
            # Valuation (2, 3, 5)
            vals = []
            for p in [2, 3, 5]:
                v = math.log1p(valuation(x, p))
                vals.append(v)
            
            is_z = 1.0 if x == 0 else 0.0
            is_sf = 1.0 if is_square_free(x) else 0.0
            is_p = 1.0 if is_prime(x_abs) else 0.0
            is_sq = 1.0 if is_square(x_abs) else 0.0
            
            # --- 4. Digital (2 dims) ---
            pop_log = math.log1p(popcount(x))
            d_sum_log = math.log1p(digit_sum(x))
            
            # Combine all features (Total 24 dims)
            feat = [
                log_val, diff1, diff2, sign, dir_sign, # 5
                *mods,                                 # 10
                *vals, is_z, is_sf, is_p, is_sq,       # 3 + 4 = 7
                pop_log, d_sum_log                     # 2
            ]
            
            features.append(feat)
            
        return torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Return raw integer sequence here.
        # In practice, you can call process_seq in collate_fn or here,
        # but it depends on the design. It's common to convert in the training loop for masking.
        # For testing, we return the raw sequence here.
        return self.sequences[idx]