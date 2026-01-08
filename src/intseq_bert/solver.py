"""
IntSeqSolver: Bayesian Beam Search Solver.

Algorithm:
1. Preprocess input using 'extract_features' to match Encoder training.
2. Predict 'pred_mag' (Log Magnitude) and 'mod_logits' (Mod probabilities).
3. Sort moduli (2-101) by confidence (max probability).
4. Apply Generalized CRT sequentially:
   - Fix "sure bits" first.
   - Resolve conflicts by prioritizing high-confidence moduli.
5. Rank candidates by Joint Log-Likelihood:
   - Score = log P(Mod sequence) + log P(Magnitude | Candidate)
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any

from .bert_model import IntSeqBERT
from .features import extract_features

# Use ALL moduli (2-101) to maximize information
ALL_MODULI = list(range(2, 102))

# Heuristic standard deviation for magnitude probability
# 0.2 in log10 scale allows for approx 1.6x error margin.
MAG_SIGMA = 0.2


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean Algorithm: ax + by = gcd(a, b)"""
    if a == 0:
        return b, 0, 1
    d, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return d, x, y


def solve_congruence(a1: int, m1: int, a2: int, m2: int) -> Tuple[Optional[int], int]:
    """
    Generalized CRT: Solves x = a1 (mod m1), x = a2 (mod m2).
    Returns (x, lcm) or (None, lcm) if inconsistent.
    """
    # Convert to Python int to avoid NumPy overflow
    a1, m1, a2, m2 = int(a1), int(m1), int(a2), int(m2)
    
    g, p, q = extended_gcd(m1, m2)
    
    # Consistency check
    if (a1 - a2) % g != 0:
        return None, (m1 * m2) // g
        
    lcm = (m1 * m2) // g
    k = (a2 - a1) // g
    x = (a1 + m1 * p * k) % lcm
    return x, lcm


def calculate_magnitude_log_prob(val: int, target_log_mag: float, sigma: float = MAG_SIGMA) -> float:
    """
    Calculates log probability of value x given target log-magnitude.
    Assumes Gaussian distribution in log-space.
    """
    if val == 0:
        # Assign a low probability for 0 if target is large
        log_val = -1.0 
    else:
        log_val = math.log10(abs(val))
    
    # Log-Likelihood of Gaussian (ignoring constants)
    return -0.5 * ((log_val - target_log_mag) / sigma) ** 2


def beam_search_bayesian(
    mod_probs: Dict[int, np.ndarray],
    pred_log_mag: float,
    beam_width: int = 20,
    top_per_mod: int = 3,
    prob_threshold: float = 0.01
) -> List[Tuple[int, float]]:
    """
    Bayesian Beam Search implementation.
    """
    
    # 1. Sort Moduli by Confidence
    mod_order = []
    for m, probs in mod_probs.items():
        max_p = np.max(probs)
        mod_order.append((m, max_p))
    
    # Sort descending: process strongest signals first
    mod_order.sort(key=lambda x: x[1], reverse=True)
    sorted_moduli = [x[0] for x in mod_order]

    # Beam State: (remainder, lcm, current_log_prob)
    beam = [(0, 1, 0.0)]
    
    # 2. Sequential CRT Application
    for m in sorted_moduli:
        probs = mod_probs[m]
        new_beam = []
        
        # Check top-k probable remainders for this mod
        top_rems = np.argsort(probs)[-top_per_mod:][::-1]
        
        for b_rem, b_lcm, b_score in beam:
            # Optimization: If LCM covers huge range, we essentially know the number.
            # But we continue to accumulate probability scores.
            
            for r_new in top_rems:
                p_new = probs[r_new]
                if p_new < prob_threshold:
                    continue
                
                # Attempt to merge
                new_rem, new_lcm = solve_congruence(b_rem, b_lcm, r_new, m)
                
                if new_rem is not None:
                    # Consistent!
                    score_update = np.log(p_new + 1e-12)
                    new_beam.append((new_rem, new_lcm, b_score + score_update))
        
        # If no consistent paths found for this mod, skip it (trust previous confident mods)
        if not new_beam:
            continue
            
        # Pruning
        new_beam.sort(key=lambda x: x[2], reverse=True)
        beam = new_beam[:beam_width]
    
    # 3. Final Candidate Generation & Scoring
    final_candidates = []
    target_magnitude = 10 ** pred_log_mag
    
    for rem, lcm, mod_score in beam:
        # We need to find integer k such that x = k*lcm + rem is close to target_magnitude
        
        # Check both positive and negative neighbors
        k_pos = round((target_magnitude - rem) / lcm)
        k_neg = round((-target_magnitude - rem) / lcm)
        
        search_range = set()
        for k in range(k_pos - 1, k_pos + 2): search_range.add(k)
        for k in range(k_neg - 1, k_neg + 2): search_range.add(k)
        
        for k in search_range:
            val = rem + k * lcm
            
            # Joint Score = Mod Score + Mag Score
            mag_score = calculate_magnitude_log_prob(val, pred_log_mag)
            total_score = mod_score + mag_score
            
            final_candidates.append((val, total_score))
            
    # Sort by Joint Score
    final_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Deduplicate
    unique_results = []
    seen = set()
    for val, score in final_candidates:
        if val not in seen:
            unique_results.append((val, score))
            seen.add(val)
    
    return unique_results


class IntSeqSolver:
    """Solver for integer sequence next-term prediction."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[IntSeqBERT] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
        elif model_path is not None:
            print(f"Loading model from {model_path} to {self.device}...")
            self.model, _ = IntSeqBERT.load_from_checkpoint(model_path, device=self.device)
            self.model.eval()
        else:
            raise ValueError("Either model_path or model must be provided")
    
    def solve(
        self,
        input_seq: List[int],
        top_k: int = 5,
        beam_width: int = 20
    ) -> Dict[str, Any]:
        """
        Predict next term in sequence using Bayesian Beam Search.
        """
        # 1. Preprocess (EXTREMELY IMPORTANT: Use same logic as training)
        feats = extract_features(input_seq)
        
        mag_f = feats['mag_features']
        mod_f = feats['mod_features']
        seq_len = mag_f.size(0)
        max_len = 128
        
        # Inference Padding
        if seq_len < max_len:
            pad = max_len - seq_len
            mag_in = torch.cat([mag_f, torch.zeros(pad, 5)], dim=0).unsqueeze(0)
            mod_in = torch.cat([mod_f, torch.zeros(pad, 200)], dim=0).unsqueeze(0)
            mask = torch.cat([torch.ones(seq_len), torch.zeros(pad)], dim=0).unsqueeze(0)
        else:
            mag_in = mag_f[-max_len:].unsqueeze(0)
            mod_in = mod_f[-max_len:].unsqueeze(0)
            mask = torch.ones(max_len).unsqueeze(0)
            
        mag_in, mod_in, mask = mag_in.to(self.device), mod_in.to(self.device), mask.to(self.device)
        
        # 2. Forward Pass
        with torch.no_grad():
            outputs = self.model(mag_in, mod_in, mask)
        
        # 3. Extract Predictions
        pred_log_mag = outputs['pred_mag'][0, -1, 0].item()
        pred_magnitude = 10 ** pred_log_mag
        
        mod_probs = {}
        for m in ALL_MODULI:
            key = f"mod{m}"
            if key in outputs:
                logits = outputs[key][0, -1, :]
                mod_probs[m] = F.softmax(logits, dim=-1).cpu().numpy()
        
        # 4. Bayesian Beam Search
        candidates_scored = beam_search_bayesian(
            mod_probs, 
            pred_log_mag, 
            beam_width=beam_width,
            top_per_mod=5 # Check top 5 probabilities for each mod
        )
        
        return {
            "candidates": candidates_scored[:top_k],
            "predicted_magnitude": pred_magnitude
        }
