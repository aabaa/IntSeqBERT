"""
IntSeqSolver: Encoder-Decoder Hybrid.
Uses the trained IntSeqBERT Encoder, but applies the robust
Entropy-based Beam Search logic from the original IntSeqDecoder.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any

from .bert_model import IntSeqBERT
from .features import extract_features

# Use ALL moduli
ALL_MODULI = list(range(2, 102))


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


class IntSeqSolver:
    """
    Solver that wraps the Encoder to behave like the probabilistic Decoder.
    """
    
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

    def _decoder_beam_search(
        self,
        predictions: Dict[str, Any],
        beam_width: int = 50,
        max_candidates: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Ported directly from IntSeqDecoder.beam_search_solve.
        Uses Entropy to sort moduli.
        """
        # Extract scalar predictions
        mu = predictions["mag_mu"]
        sigma = math.exp(0.5 * predictions["mag_logvar"])
        
        # Search range (linear scale)
        min_val = 10 ** (mu - 3 * sigma)
        max_val = 10 ** (mu + 3 * sigma)
        
        # Sign Logic
        sign_idx = np.argmax(predictions["sign_logits"])
        sign = sign_idx - 1 # 0->-1, 1->0, 2->1
        if sign == 0:
            return [(0, 0.0)] # Score 0.0 for zero
            
        # 1. Collect and Sort Moduli by Entropy (Uncertainty)
        mod_info = []
        for m in ALL_MODULI:
            probs = predictions[f"mod{m}"] # numpy array
            
            # Entropy calculation
            # Add epsilon to avoid log(0)
            entropy = -np.sum(probs * np.log(probs + 1e-9))
            
            # Get top candidates
            # Dynamic k: if m is small, take fewer.
            k = min(m, 3)
            top_indices = np.argsort(probs)[-k:][::-1]
            
            candidates = []
            for idx in top_indices:
                p = probs[idx]
                if p > 0.01:
                    candidates.append((idx, np.log(p + 1e-12)))
            
            mod_info.append({
                'm': m,
                'entropy': entropy,
                'candidates': candidates
            })
            
        # Sort by entropy ASCENDING (Most confident/Lowest entropy first)
        mod_info.sort(key=lambda x: x['entropy'])
        
        # 2. Beam Search
        # Beam state: (current_remainder, current_lcm, log_prob_score)
        beam = [(0, 1, 0.0)]
        
        for info in mod_info:
            m = info['m']
            cands = info['candidates']
            
            new_beam = []
            
            for b_rem, b_lcm, b_score in beam:
                # LCM Cap based on Magnitude prediction
                # If LCM is already much larger than max_val, we don't need more constraints
                if b_lcm > max_val * 100:
                     new_beam.append((b_rem, b_lcm, b_score))
                     continue

                for c_rem, c_score in cands:
                    res, new_lcm = solve_congruence(b_rem, b_lcm, c_rem, m)
                    
                    if res is not None:
                        new_score = b_score + c_score
                        new_beam.append((res, new_lcm, new_score))
            
            # If all paths died, fallback strategy:
            # Skip this modulus (assume it's the outlier/error) and keep previous beam
            if not new_beam:
                continue
            
            # Pruning
            new_beam.sort(key=lambda x: x[2], reverse=True)
            
            # Dedup
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

        # 3. Final Selection
        final_results = []
        target_val = 10**mu
        
        for rem, lcm, score in beam:
            # Find k near target
            k_approx = (target_val - rem) / lcm
            k_near = round(k_approx)
            
            for k in range(k_near - 2, k_near + 3):
                if k < 0: continue
                val = k * lcm + rem
                if val == 0: continue
                
                # Magnitude score (Mahalanobis distance)
                log_val = math.log10(val)
                mag_score = -0.5 * ((log_val - mu) / sigma)**2
                
                total_score = score + mag_score
                final_results.append((val * sign, total_score))
                
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        # Unique output
        output = []
        seen = set()
        for val, sc in final_results:
            if val not in seen:
                output.append((val, sc))
                seen.add(val)
            if len(output) >= max_candidates:
                break
                
        return output

    def solve(
        self,
        input_seq: List[int],
        top_k: int = 5,
        beam_width: int = 50
    ) -> Dict[str, Any]:
        """
        Wrapper to convert Encoder output to Decoder-like format.
        """
        # 1. Preprocess
        feats = extract_features(input_seq)
        mag_f, mod_f = feats['mag_features'], feats['mod_features']
        
        # Padding logic (omitted for brevity, same as before)
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

        # 2. Forward
        with torch.no_grad():
            outputs = self.model(mag_in, mod_in, mask)
            
        # 3. Adapt Encoder Output to Decoder Format
        # Encoder: pred_mag (B, L, 5) -> [Log, Vel, Acc, Sign, Idx]
        pred_log_mag = outputs['pred_mag'][0, -1, 0].item()
        pred_sign_val = outputs['pred_mag'][0, -1, 3].item()
        
        # Simulate Sign Logits (3 classes: -1, 0, 1)
        # 0->-1, 1->0, 2->1
        sign_logits = np.array([-10.0, -10.0, -10.0])
        if pred_sign_val > 0.2:
            sign_logits[2] = 10.0 # Positive
        elif pred_sign_val < -0.2:
            sign_logits[0] = 10.0 # Negative
        else:
            sign_logits[1] = 10.0 # Zero
            
        # Prepare pseudo-decoder dictionary
        predictions = {
            "mag_mu": pred_log_mag,
            "mag_logvar": math.log(0.2**2), # Fixed variance 0.2
            "sign_logits": sign_logits
        }
        
        for m in ALL_MODULI:
            key = f"mod{m}"
            if key in outputs:
                logits = outputs[key][0, -1, :]
                predictions[f"mod{m}"] = F.softmax(logits, dim=-1).cpu().numpy()
        
        # 4. Run Decoder Logic
        candidates_scored = self._decoder_beam_search(
            predictions, 
            beam_width=beam_width,
            max_candidates=top_k
        )
        
        return {
            "candidates": candidates_scored,
            "predicted_magnitude": 10**pred_log_mag
        }
