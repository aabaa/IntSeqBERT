"""
Number-theoretic decoder for reconstructing integers from feature vectors.
Implements Dynamic Confidence-Ordered CRT and ResNet Architecture for Identity Mapping.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

# Configuration constants
NUM_MAGNITUDE_BINS = 4096
MAX_LOG_VALUE = 100.0  # Covers up to 10^100

# =========================================================
# Dynamic CRT Lookup Table (Precomputed Logic)
# Bases: 3, 7, 8, 11, 13, 25 (Coprime set covering all features)
# =========================================================
PRIMES = [3, 7, 8, 11, 13, 25]
NUM_BASES = 6
LUT_SIZE = 1 << NUM_BASES  # 64

def extended_gcd(a, b):
    """Extended Euclidean Algorithm."""
    if a == 0: return b, 0, 1
    d, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    return d, x, x1

def precompute_crt_lut():
    """
    Precomputes CRT Basis and LCM for all 64 subsets of moduli.
    Returns: (basis_lut, lcm_lut) on CPU
    """
    basis_lut = torch.zeros((LUT_SIZE, NUM_BASES), dtype=torch.long)
    lcm_lut = torch.zeros((LUT_SIZE,), dtype=torch.long)

    for mask in range(1, LUT_SIZE):
        selected = [i for i in range(NUM_BASES) if (mask >> i) & 1]
        if not selected:
            lcm_lut[mask] = 1
            continue
        
        mods = [PRIMES[i] for i in selected]
        current_lcm = 1
        for m in mods:
            current_lcm = math.lcm(current_lcm, m)
        lcm_lut[mask] = current_lcm
        
        for i in selected:
            m_i = PRIMES[i]
            M_i = current_lcm // m_i
            _, inv, _ = extended_gcd(M_i, m_i)
            # Basis weight: w_i = (inv * M_i)
            weight = (inv % m_i) * M_i
            basis_lut[mask, i] = weight
            
    return basis_lut, lcm_lut


class NumberTheoreticDecoder(nn.Module):
    """
    Decoder that reconstructs integers from 35-dimensional feature vectors.
    Uses ResNet Architecture to solve Identity Mapping problem.
    """
    
    def __init__(
        self,
        input_dim: int = 35,
        hidden_dim: int = 512,  # Increased capacity
        dropout: float = 0.05   # Reduced dropout for numerical stability
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection (35 -> 512)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # ResBlock 1 (Pre-Norm)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act1 = nn.GELU()
        
        # ResBlock 2 (Pre-Norm)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.GELU()
        
        # Multi-task heads
        self.sign_head = nn.Linear(hidden_dim, 3)           # 0:-, 1:0, 2:+
        self.mag_head = nn.Linear(hidden_dim, NUM_MAGNITUDE_BINS) # Classification
        
        # Modulo heads
        self.mod3_head = nn.Linear(hidden_dim, 3)
        self.mod5_head = nn.Linear(hidden_dim, 5)
        self.mod7_head = nn.Linear(hidden_dim, 7)
        self.mod8_head = nn.Linear(hidden_dim, 8)
        self.mod10_head = nn.Linear(hidden_dim, 10)
        self.mod11_head = nn.Linear(hidden_dim, 11)
        self.mod13_head = nn.Linear(hidden_dim, 13)
        self.mod100_head = nn.Linear(hidden_dim, 100)

        # Register CRT Buffers
        basis_lut, lcm_lut = precompute_crt_lut()
        self.register_buffer('crt_basis_lut', basis_lut)
        self.register_buffer('crt_lcm_lut', lcm_lut)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Input Projection
        out = self.input_proj(x)
        out = self.input_bn(out)
        out = self.dropout(out)
        
        # ResBlock 1 (Skip Connection)
        residual = out
        out = self.ln1(out)
        out = self.fc1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = out + residual  # Add input to output
        
        # ResBlock 2 (Skip Connection)
        residual = out
        out = self.ln2(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = out + residual  # Add input to output
        
        h = out
        
        return {
            "sign": self.sign_head(h),
            "mag": self.mag_head(h),
            "mod3": self.mod3_head(h),
            "mod5": self.mod5_head(h),
            "mod7": self.mod7_head(h),
            "mod8": self.mod8_head(h),
            "mod10": self.mod10_head(h),
            "mod11": self.mod11_head(h),
            "mod13": self.mod13_head(h),
            "mod100": self.mod100_head(h)
        }
    
    @staticmethod
    def bin_to_log_range_vec(bin_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert bin indices to (log_min, log_center) tensors."""
        bin_width = MAX_LOG_VALUE / NUM_MAGNITUDE_BINS
        log_min = bin_indices.float() * bin_width
        log_center = (bin_indices.float() + 0.5) * bin_width
        return log_min, log_center

    def batch_reconstruct(
        self,
        feature_vectors: torch.Tensor,
        top_k_bins: int = 5,
        neighbors: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct integers using Dynamic Confidence-Ordered CRT.
        """
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            preds = self.forward(feature_vectors)
            batch_size = feature_vectors.size(0)
            device = feature_vectors.device

            # ------------------------------------------------
            # 0. Basic Predictions (Sign & Magnitude)
            # ------------------------------------------------
            sign_probs = F.softmax(preds['sign'], dim=1)
            sign_idx = torch.argmax(sign_probs, dim=1)
            sign_val = sign_idx - 1  # {-1, 0, 1}

            # Magnitude (Center of Top-1 Bin)
            mag_log_probs_all = F.log_softmax(preds["mag"], dim=1)
            top_bin = torch.argmax(mag_log_probs_all, dim=1)
            _, log_center = self.bin_to_log_range_vec(top_bin)
            est_mag = torch.pow(10, log_center.to(device)) # (B,)

            # ------------------------------------------------
            # 1. Rank Moduli by Information Gain (Log-Odds Lift)
            # ------------------------------------------------
            bases_config = [
                ('mod3', 3), ('mod7', 7), ('mod8', 8),
                ('mod11', 11), ('mod13', 13), ('mod100', 25)
            ]
            
            scores_list = []
            residues_list = []
            head_log_probs = []
            
            for name, m in bases_config:
                if m == 25:
                    lp100 = F.log_softmax(preds['mod100'], dim=1)
                    p25 = torch.zeros(batch_size, 25, device=device)
                    for i in range(4):
                        p25 += torch.exp(lp100[:, i*25:(i+1)*25])
                    log_p = torch.log(p25 + 1e-10)
                else:
                    log_p = F.log_softmax(preds[name], dim=1)
                
                head_log_probs.append(log_p)
                max_lp, max_idx = torch.max(log_p, dim=1)
                score = max_lp + math.log(m)
                
                scores_list.append(score)
                residues_list.append(max_idx)
            
            all_scores = torch.stack(scores_list, dim=1)
            all_residues = torch.stack(residues_list, dim=1)
            _, sorted_indices = torch.sort(all_scores, descending=True, dim=1)

            # ------------------------------------------------
            # 2. Parallel CRT Hypotheses Generation (Levels 2 to 6)
            # ------------------------------------------------
            candidates_list = []
            
            for k in range(2, 7):
                top_k_idx = sorted_indices[:, :k]
                mask = torch.sum(1 << top_k_idx, dim=1)
                
                curr_basis = self.crt_basis_lut[mask]
                curr_lcm = self.crt_lcm_lut[mask]
                
                x_base = torch.sum(all_residues * curr_basis, dim=1) % curr_lcm
                
                diff = est_mag - x_base.float()
                step_k = torch.round(diff / curr_lcm.float())
                cand_mag = x_base + step_k.long() * curr_lcm
                
                cand_final = cand_mag * sign_val
                candidates_list.append(cand_final)

            # ------------------------------------------------
            # 3. Unified Scoring & Selection
            # ------------------------------------------------
            all_cands = torch.stack(candidates_list, dim=1) # (B, 5)
            
            # A. Bin Probability Score
            cand_abs = all_cands.float().abs()
            cand_log10 = torch.log10(cand_abs + 1e-10)
            bin_width = MAX_LOG_VALUE / NUM_MAGNITUDE_BINS
            cand_bins = (cand_log10 / bin_width).long().clamp(0, NUM_MAGNITUDE_BINS - 1)
            
            score_mag = torch.gather(mag_log_probs_all, 1, cand_bins)
            
            # B. Modulo Probability Score
            score_mods = torch.zeros_like(score_mag)
            
            for i, (name, m) in enumerate(bases_config):
                lp = head_log_probs[i]
                cand_res = all_cands.abs() % m
                s = torch.gather(lp, 1, cand_res)
                score_mods += s
            
            total_score = score_mag + score_mods
            
            best_idx = torch.argmax(total_score, dim=1)
            best_integers = torch.gather(all_cands, 1, best_idx.unsqueeze(1)).squeeze(1)
            best_scores = torch.gather(total_score, 1, best_idx.unsqueeze(1)).squeeze(1)

        if was_training:
            self.train()
        
        return best_integers, best_scores

    def reconstruct_value(self, features: torch.Tensor, **kwargs) -> Tuple[int, float]:
        """Wrapper for backward compatibility."""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        best_ints, best_scores = self.batch_reconstruct(features)
        return best_ints[0].item(), best_scores[0].item()