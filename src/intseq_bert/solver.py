"""
IntSeqSolver: Bayesian Beam Search Solver with Sign Awareness.

Updates:
- Incorporates Sign prediction (Index 3) to filter candidates.
- Increased default beam_width for better precision.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any

from .bert_model import IntSeqBERT
from .features import extract_features

ALL_MODULI = list(range(2, 102))
MAG_SIGMA = 0.2
SIGN_SIGMA = 0.5  # Soft constraint for sign


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    d, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return d, x, y


def solve_congruence(a1: int, m1: int, a2: int, m2: int) -> Tuple[Optional[int], int]:
    a1, m1, a2, m2 = int(a1), int(m1), int(a2), int(m2)
    g, p, q = extended_gcd(m1, m2)
    if (a1 - a2) % g != 0:
        return None, (m1 * m2) // g
    lcm = (m1 * m2) // g
    k = (a2 - a1) // g
    x = (a1 + m1 * p * k) % lcm
    return x, lcm


def calculate_joint_log_prob(
    val: int, 
    target_log_mag: float, 
    target_sign: float, 
    mag_sigma: float = MAG_SIGMA
) -> float:
    """
    Calculates joint log probability of Magnitude AND Sign.
    """
    # 1. Magnitude Score
    if val == 0:
        log_val = -1.0 
        val_sign = 0.0
    else:
        log_val = math.log10(abs(val))
        val_sign = 1.0 if val > 0 else -1.0
    
    mag_score = -0.5 * ((log_val - target_log_mag) / mag_sigma) ** 2
    
    # 2. Sign Score (Regression check)
    # target_sign is approx 1.0, -1.0, or 0.0
    # We penalize distance from predicted sign
    sign_score = -0.5 * ((val_sign - target_sign) / SIGN_SIGMA) ** 2
    
    return mag_score + sign_score


def beam_search_bayesian(
    mod_probs: Dict[int, np.ndarray],
    pred_log_mag: float,
    pred_sign: float,  # NEW: Predicted sign value
    beam_width: int = 100, # Increased default
    top_per_mod: int = 3,
    prob_threshold: float = 0.01
) -> List[Tuple[int, float]]:
    
    # 1. Sort Moduli
    mod_order = []
    for m, probs in mod_probs.items():
        max_p = np.max(probs)
        mod_order.append((m, max_p))
    mod_order.sort(key=lambda x: x[1], reverse=True)
    sorted_moduli = [x[0] for x in mod_order]

    # Calculate LCM Limit
    target_mag = 10 ** pred_log_mag
    lcm_limit = max(1000, target_mag * 1000)

    # Beam State
    beam = [(0, 1, 0.0)]
    
    # 2. Sequential CRT
    for m in sorted_moduli:
        probs = mod_probs[m]
        new_beam = []
        
        top_rems = np.argsort(probs)[-top_per_mod:][::-1]
        
        for b_rem, b_lcm, b_score in beam:
            # LCM Cap
            if b_lcm > lcm_limit:
                 new_beam.append((b_rem, b_lcm, b_score))
                 continue

            for r_new in top_rems:
                p_new = probs[r_new]
                if p_new < prob_threshold:
                    continue
                
                new_rem, new_lcm = solve_congruence(b_rem, b_lcm, r_new, m)
                
                if new_rem is not None:
                    score_update = np.log(p_new + 1e-12)
                    new_beam.append((new_rem, new_lcm, b_score + score_update))
        
        if not new_beam:
            continue
            
        new_beam.sort(key=lambda x: x[2], reverse=True)
        
        # Deduplicate
        unique_beam = []
        seen = set()
        for item in new_beam:
            key = (item[0], item[1])
            if key not in seen:
                unique_beam.append(item)
                seen.add(key)
            if len(unique_beam) >= beam_width:
                break
        
        beam = unique_beam
    
    # 3. Final Candidate Generation
    final_candidates = []
    
    for rem, lcm, mod_score in beam:
        k_pos = round((target_mag - rem) / lcm)
        k_neg = round((-target_mag - rem) / lcm)
        
        search_range = set()
        for k in range(k_pos - 1, k_pos + 2): search_range.add(k)
        for k in range(k_neg - 1, k_neg + 2): search_range.add(k)
        
        for k in search_range:
            val = rem + k * lcm
            
            # Joint Score with Sign
            total_score = mod_score + calculate_joint_log_prob(val, pred_log_mag, pred_sign)
            
            final_candidates.append((val, total_score))
            
    final_candidates.sort(key=lambda x: x[1], reverse=True)
    
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
        beam_width: int = 100 # Default increased to 100
    ) -> Dict[str, Any]:
        
        feats = extract_features(input_seq)
        mag_f = feats['mag_features']
        mod_f = feats['mod_features']
        seq_len = mag_f.size(0)
        max_len = 128
        
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
        
        with torch.no_grad():
            outputs = self.model(mag_in, mod_in, mask)
        
        # Extract LogMag (Index 0) and Sign (Index 3)
        pred_log_mag = outputs['pred_mag'][0, -1, 0].item()
        pred_sign = outputs['pred_mag'][0, -1, 3].item() # NEW
        
        pred_magnitude = 10 ** pred_log_mag
        
        mod_probs = {}
        for m in ALL_MODULI:
            key = f"mod{m}"
            if key in outputs:
                logits = outputs[key][0, -1, :]
                mod_probs[m] = F.softmax(logits, dim=-1).cpu().numpy()
        
        candidates_scored = beam_search_bayesian(
            mod_probs, 
            pred_log_mag, 
            pred_sign,  # Pass sign
            beam_width=beam_width,
            top_per_mod=5
        )
        
        return {
            "candidates": candidates_scored[:top_k],
            "predicted_magnitude": pred_magnitude
        }