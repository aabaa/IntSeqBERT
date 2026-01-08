"""
IntSeqSolver: Beam Search + CRT solver for integer sequence prediction.
Uses trained IntSeqBERT encoder to predict the next term in a sequence.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any
from sympy.ntheory.modular import crt
from .bert_model import IntSeqBERT


# Default prime set for CRT
DEFAULT_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101
]


def compute_magnitude_features(seq_list: List[int]) -> List[List[float]]:
    """
    Compute magnitude features for a sequence.
    Returns list of [log_magnitude, sign, diff_log, diff_sign, position].
    """
    features = []
    for i, val in enumerate(seq_list):
        sign = 1 if val > 0 else (-1 if val < 0 else 0)
        log_val = math.log10(abs(val) + 1)
        
        if i > 0:
            diff = val - seq_list[i - 1]
            diff_sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
            diff_log = math.log10(abs(diff) + 1)
        else:
            diff_sign = 0
            diff_log = 0
        
        pos = i / 100.0
        features.append([log_val, sign, diff_log, diff_sign, pos])
    
    return features


def compute_mod_features(seq_list: List[int]) -> List[List[float]]:
    """
    Compute modulo features for a sequence.
    Returns list of 200-dimensional features (mod residuals for mod 2-101).
    
    Note: The model expects 200-dim input where each pair of dimensions
    represents mod{m} residual information.
    """
    features = []
    for val in seq_list:
        # Simple: just use residuals directly as floats
        # Model was trained with this format
        mods = [float(val % m) for m in range(2, 102)]
        # Duplicate to get 200 dimensions (matching training data format)
        mods = mods + mods  # 100 + 100 = 200
        features.append(mods)
    return features


def beam_search_crt(
    mod_probs: Dict[int, np.ndarray],
    primes: List[int],
    beam_width: int = 20,
    top_per_mod: int = 5,
    prob_threshold: float = 1e-5
) -> List[Tuple[int, int, float]]:
    """
    Perform beam search using Chinese Remainder Theorem.
    
    Args:
        mod_probs: Dictionary mapping prime p to probability array of length p
        primes: List of primes to use for CRT
        beam_width: Maximum number of candidates to keep
        top_per_mod: Number of top remainders to consider per modulus
        prob_threshold: Minimum probability threshold
        
    Returns:
        List of (remainder, modulus, log_probability) tuples
    """
    candidates = [(0, 1, 0.0)]
    
    for p in primes:
        if p not in mod_probs:
            continue
            
        probs = mod_probs[p]
        new_candidates = []
        
        # Get top remainders by probability
        top_rems = np.argsort(probs)[-top_per_mod:][::-1]
        
        for rem, modulus, score in candidates:
            for r_new in top_rems:
                prob_new = probs[r_new]
                if prob_new < prob_threshold:
                    continue
                
                try:
                    res = crt([modulus, p], [rem, r_new])
                    if res is None:
                        continue
                    
                    new_rem, new_mod = res
                    new_score = score + np.log(prob_new + 1e-10)
                    new_candidates.append((int(new_rem), int(new_mod), new_score))
                except Exception:
                    continue
        
        # Keep top candidates
        new_candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = new_candidates[:beam_width]
    
    return candidates


def magnitude_matching(
    candidates: List[Tuple[int, int, float]],
    target_magnitude: float,
    pred_log_magnitude: float,
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """
    Match CRT candidates to predicted magnitude.
    
    Args:
        candidates: List of (remainder, modulus, score) from beam search
        target_magnitude: Predicted magnitude (linear scale)
        pred_log_magnitude: Predicted log magnitude
        top_k: Number of final candidates to return
        
    Returns:
        List of (value, magnitude_error) tuples
    """
    final_results = []
    
    for rem, modulus, score in candidates:
        k_approx = round((target_magnitude - rem) / modulus) if modulus > 0 else 0
        
        for k in [k_approx, k_approx - 1, k_approx + 1]:
            val = rem + k * modulus
            
            try:
                val_log = math.log10(abs(val) + 1)
                mag_error = abs(val_log - pred_log_magnitude)
            except Exception:
                mag_error = 100.0
            
            final_results.append((val, mag_error, score))
    
    # Sort by score (higher is better)
    final_results.sort(key=lambda x: x[2], reverse=True)
    
    # Deduplicate and return top_k
    output = []
    seen = set()
    for val, err, _ in final_results:
        if val not in seen:
            output.append((val, err))
            seen.add(val)
        if len(output) >= top_k:
            break
    
    return output


class IntSeqSolver:
    """Solver for integer sequence next-term prediction."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[IntSeqBERT] = None,
        device: Optional[str] = None,
        primes: Optional[List[int]] = None
    ):
        """
        Initialize solver with trained encoder.
        
        Args:
            model_path: Path to checkpoint file
            model: Pre-loaded model (alternative to model_path)
            device: Device to use ('cuda' or 'cpu')
            primes: List of primes for CRT (default: 26 primes up to 101)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.primes = primes or DEFAULT_PRIMES
        
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        elif model_path is not None:
            print(f"Loading model from {model_path} to {self.device}...")
            self.model, _ = IntSeqBERT.load_from_checkpoint(model_path, device=self.device)
            self.model.eval()
        else:
            raise ValueError("Either model_path or model must be provided")
    
    def preprocess_sequence(
        self,
        seq_list: List[int],
        max_len: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert integer list to model input tensors.
        
        Args:
            seq_list: Input sequence
            max_len: Maximum sequence length
            
        Returns:
            Tuple of (mag_tensor, mod_tensor, mask_tensor)
        """
        mag_features = compute_magnitude_features(seq_list)
        mod_features = compute_mod_features(seq_list)
        
        curr_len = len(seq_list)
        
        if curr_len < max_len:
            pad_len = max_len - curr_len
            mag_features += [[0.0] * 5] * pad_len
            mod_features += [[0] * 200] * pad_len
            mask = [1.0] * curr_len + [0.0] * pad_len
        else:
            mag_features = mag_features[-max_len:]
            mod_features = mod_features[-max_len:]
            mask = [1.0] * max_len
        
        mag_tensor = torch.tensor([mag_features], dtype=torch.float32).to(self.device)
        mod_tensor = torch.tensor([mod_features], dtype=torch.float32).to(self.device)
        mask_tensor = torch.tensor([mask], dtype=torch.float32).to(self.device)
        
        return mag_tensor, mod_tensor, mask_tensor
    
    def solve(
        self,
        input_seq: List[int],
        top_k: int = 5,
        beam_width: int = 20
    ) -> Dict[str, Any]:
        """
        Predict next term in sequence.
        
        Args:
            input_seq: Input integer sequence
            top_k: Number of candidates to return
            beam_width: Beam search width
            
        Returns:
            Dictionary with 'candidates' and 'predicted_magnitude'
        """
        # 1. Preprocess and run encoder
        mag_in, mod_in, mask = self.preprocess_sequence(input_seq)
        
        with torch.no_grad():
            outputs = self.model(mag_in, mod_in, mask)
        
        # 2. Extract predictions
        pred_log_mag = outputs['pred_mag'][0, -1, 0].item()
        pred_magnitude = 10 ** pred_log_mag - 1
        
        # 3. Get mod probabilities
        mod_probs = {}
        for p in self.primes:
            logits = outputs[f'mod{p}'][0, -1, :]
            probs = F.softmax(logits, dim=-1)
            mod_probs[p] = probs.cpu().numpy()
        
        # 4. Beam search with CRT
        candidates = beam_search_crt(mod_probs, self.primes, beam_width)
        
        # 5. Magnitude matching
        final_candidates = magnitude_matching(
            candidates, pred_magnitude, pred_log_mag, top_k
        )
        
        return {
            "candidates": final_candidates,
            "predicted_magnitude": pred_magnitude
        }
