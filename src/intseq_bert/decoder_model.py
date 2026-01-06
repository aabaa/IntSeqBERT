"""
Number-theoretic decoder/solver for reconstructing integers from latent representations.
Implements Heteroscedastic Regression for Magnitude and Beam Search CRT for Modular Reconstruction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import heapq

# Configuration
MOD_RANGE = range(2, 102)  # Moduli 2..101

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean Algorithm.
    Returns (g, x, y) such that ax + by = g = gcd(a, b).
    """
    if a == 0:
        return b, 0, 1
    d, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return d, x, y

def solve_congruence(a1: int, m1: int, a2: int, m2: int) -> Tuple[Optional[int], int]:
    """
    Solves the system:
      x = a1 (mod m1)
      x = a2 (mod m2)
    Returns (x, lcm(m1, m2)). Returns (None, lcm) if inconsistent.
    """
    g, p, q = extended_gcd(m1, m2)
    
    # Check consistency: (a1 - a2) must be divisible by g
    if (a1 - a2) % g != 0:
        return None, (m1 * m2) // g
        
    lcm = (m1 * m2) // g
    
    # Solution using Bezout's identity
    # m1*p + m2*q = g
    # x = a1 + m1 * p * (a2 - a1) / g
    # We use integer division for (a2 - a1) // g
    k = (a2 - a1) // g
    x = (a1 + m1 * p * k) % lcm
    
    return x, lcm


class IntSeqDecoder(nn.Module):
    """
    Decoder module that predicts the next integer from a latent vector.
    
    Architecture:
      1. Magnitude Head: Predicts log10(|x|) mean and variance (Heteroscedastic Regression).
      2. Mod Heads: 100 separate heads for predicting x mod m for m in 2..101.
      3. Solver: Beam Search CRT to reconstruct integer from predictions.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 1. Magnitude Head (Regression)
        # Outputs: [mu (mean), s (log_variance), sign_logits (3 classes: -, 0, +)]
        self.mag_head = nn.Linear(hidden_dim, 2) 
        self.sign_head = nn.Linear(hidden_dim, 3) # -1, 0, 1
        
        # 2. Mod Spectrum Heads (Classification)
        # One head for each modulus from 2 to 101
        self.mod_heads = nn.ModuleDict({
            f"mod{m}": nn.Linear(hidden_dim, m) for m in MOD_RANGE
        })
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Latent vector (Batch, d_model) - typically the [MASK] token state or pooled state.
            
        Returns:
            Dict containing predictions:
            - 'mag_mu': (B, 1)
            - 'mag_logvar': (B, 1)
            - 'sign_logits': (B, 3)
            - 'mod{m}_logits': (B, m) for each m
        """
        h = self.trunk(x)
        
        # Magnitude
        mag_out = self.mag_head(h) # (B, 2)
        mu = mag_out[:, 0:1]
        logvar = mag_out[:, 1:2]
        
        # Sign
        sign_logits = self.sign_head(h)
        
        # Mods
        mod_logits = {}
        for m_str, head in self.mod_heads.items():
            mod_logits[m_str] = head(h)
            
        return {
            "mag_mu": mu,
            "mag_logvar": logvar,
            "sign_logits": sign_logits,
            **mod_logits
        }
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes combined loss (Gaussian NLL for mag, CrossEntropy for mods).
        
        Args:
            predictions: Output from forward()
            targets: Dict with 'mag', 'mod2', 'mod3'... keys.
                     Note: 'mag' target should be log10(|x|).
        """
        device = predictions["mag_mu"].device
        total_loss = 0.0
        
        # 1. Magnitude Loss (Heteroscedastic Regression)
        # Loss = 0.5 * exp(-s) * (y - mu)^2 + 0.5 * s
        mu = predictions["mag_mu"].squeeze(-1)
        s = predictions["mag_logvar"].squeeze(-1)
        y = targets["mag"].to(device)
        
        # Ensure targets are valid (mask out padding if necessary, but here assuming valid batch)
        # Simple NLL
        loss_mag = 0.5 * torch.exp(-s) * (y - mu)**2 + 0.5 * s
        total_loss += loss_mag.mean()
        
        # 2. Sign Loss
        # Create sign targets from raw values? 
        # Usually target dict should have 'sign' or we infer it.
        # For now, let's assume 'sign' is implicitly handled or we add it to data loader.
        # Skipping sign loss for brevity unless added to data pipeline.
        
        # 3. Mod Losses (Cross Entropy)
        # Summing 100 losses might be large, so we scale it down or average.
        mod_loss_sum = 0.0
        count = 0
        
        for m in MOD_RANGE:
            key = f"mod{m}"
            if key in targets:
                logits = predictions[key] # (B, m)
                labels = targets[key].to(device) # (B,)
                
                # Check for ignore_index (-100) used in collator
                loss_m = F.cross_entropy(logits, labels, ignore_index=-100)
                mod_loss_sum += loss_m
                count += 1
        
        if count > 0:
            # Average mod loss to keep scale consistent with mag loss
            total_loss += mod_loss_sum / count
            
        return total_loss

    def beam_search_solve(
        self,
        predictions: Dict[str, torch.Tensor],
        beam_width: int = 10,
        max_candidates: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Solves for the integer x using Beam Search CRT.
        Designed for inference on a SINGLE item (Batch size 1).
        
        Returns:
            List of (predicted_integer, score) tuples.
        """
        # Extract scalar predictions
        mu = predictions["mag_mu"].item()
        sigma = math.exp(0.5 * predictions["mag_logvar"].item())
        
        # Search range in log10 scale: [mu - 3sigma, mu + 3sigma]
        # Converted to linear scale
        min_val = 10 ** (mu - 3 * sigma)
        max_val = 10 ** (mu + 3 * sigma)
        
        # Sign
        sign_idx = torch.argmax(predictions["sign_logits"]).item()
        sign = sign_idx - 1 # 0->-1, 1->0, 2->1
        if sign == 0:
            return [(0, 1.0)]
        
        # 1. Collect and Sort Moduli by Confidence (Entropy)
        mod_info = []
        for m in MOD_RANGE:
            logits = predictions[f"mod{m}"].flatten()
            probs = F.softmax(logits, dim=0)
            
            # Entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            
            # Get top k candidates for this mod
            # For small m, k=1 or 2. For large m, maybe more.
            # Let's use dynamic k based on probability mass?
            # For simplicity: Top-3 or prob > 0.1
            top_vals, top_indices = torch.topk(probs, k=min(m, 3))
            
            candidates = []
            for p, idx in zip(top_vals, top_indices):
                if p.item() > 0.01: # Filter very low prob
                    candidates.append((idx.item(), math.log(p.item())))
            
            mod_info.append({
                'm': m,
                'entropy': entropy,
                'candidates': candidates
            })
            
        # Sort by entropy ascending (most confident first)
        mod_info.sort(key=lambda x: x['entropy'])
        
        # 2. Beam Search
        # Beam state: (current_remainder, current_lcm, log_prob_score)
        beam = [(0, 1, 0.0)] # Initial state: x = 0 (mod 1)
        
        for info in mod_info:
            m = info['m']
            cands = info['candidates']
            
            new_beam = []
            
            for b_rem, b_lcm, b_score in beam:
                # Early exit if LCM is already large enough to cover the range significantly
                # But we need redundancy to correct errors, so we continue.
                # Only prune if beam is too large.
                
                for c_rem, c_score in cands:
                    # Solve CRT: x = b_rem (mod b_lcm), x = c_rem (mod m)
                    res, new_lcm = solve_congruence(b_rem, b_lcm, c_rem, m)
                    
                    if res is not None:
                        # Consistent
                        new_score = b_score + c_score
                        new_beam.append((res, new_lcm, new_score))
            
            if not new_beam:
                # No consistent extensions found? 
                # This implies the current most-confident beam path is dead.
                # In strict beam search, this branch dies.
                continue
                
            # Prune beam
            # Sort by score descending
            new_beam.sort(key=lambda x: x[2], reverse=True)
            beam = new_beam[:beam_width]
            
            # Optimization: If LCM is huge, we might stop
            if beam[0][1] > max_val * 10: # Sufficient precision
                break
        
        # 3. Final Selection
        # Convert modular results to actual integers within Magnitude range
        final_results = []
        
        for rem, lcm, score in beam:
            # We have x = rem (mod lcm)
            # We want x in [min_val, max_val] approximately
            # k * lcm + rem approx 10^mu
            
            target = 10**mu
            
            # Find k such that k*lcm + rem is close to target
            k_float = (target - rem) / lcm
            k_near = round(k_float)
            
            # Check a few neighbors
            for k in range(k_near - 1, k_near + 2):
                if k < 0: continue # magnitudes are usually positive logic here
                val = k * lcm + rem
                if val == 0: continue
                
                # Calculate distance score (Mahalanobis distance style)
                # How well does it fit the magnitude prediction?
                log_val = math.log10(val)
                mag_score = -0.5 * ((log_val - mu) / sigma)**2
                
                # Combine mod score and mag score
                # Tuning factor might be needed to balance them
                total_score = score + mag_score
                
                final_results.append((val * sign, total_score))
        
        # Sort by total score
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top unique integers
        unique_results = []
        seen = set()
        for val, sc in final_results:
            if val not in seen:
                unique_results.append((val, sc))
                seen.add(val)
                if len(unique_results) >= max_candidates:
                    break
                    
        return unique_results
