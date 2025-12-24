"""
Number-theoretic decoder for reconstructing integers from feature vectors.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from .features import log_magnitude


# Configuration constants for magnitude binning
NUM_MAGNITUDE_BINS = 4096
MAX_LOG_VALUE = 100.0  # Covers up to 10^100

class NumberTheoreticDecoder(nn.Module):
    """
    Decoder that reconstructs integers from 35-dimensional feature vectors.
    
    Uses multi-task learning to predict:
    - Sign (classification: -, 0, +)
    - Magnitude (classification: 4096 bins covering 0 to 10^100)
    - Modulo residues (classification: mod 3, 5, 7, 8, 10, 11, 13, 100)
    
    Reconstruction uses probabilistic Chinese Remainder Theorem search.
    
    Args:
        input_dim: Input feature dimension (default: 35)
        hidden_dim: Hidden layer dimension (default: 256)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        input_dim: int = 35,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Shared encoder (expand compressed features)
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Multi-task heads
        self.sign_head = nn.Linear(hidden_dim, 3)           # 0:-, 1:0, 2:+
        self.mag_head = nn.Linear(hidden_dim, NUM_MAGNITUDE_BINS)  # Classification (4096 bins)
        self.mod3_head = nn.Linear(hidden_dim, 3)           # mod 3
        self.mod5_head = nn.Linear(hidden_dim, 5)           # mod 5
        self.mod7_head = nn.Linear(hidden_dim, 7)           # mod 7
        self.mod8_head = nn.Linear(hidden_dim, 8)           # mod 8
        self.mod10_head = nn.Linear(hidden_dim, 10)         # mod 10
        self.mod11_head = nn.Linear(hidden_dim, 11)         # mod 11
        self.mod13_head = nn.Linear(hidden_dim, 13)         # mod 13
        self.mod100_head = nn.Linear(hidden_dim, 100)       # mod 100
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through decoder.
        
        Args:
            x: Input features (batch_size, input_dim) or (input_dim,)
        
        Returns:
            Dictionary with predictions from all heads:
                - sign: (batch, 3) logits
                - mag: (batch, 1) regression
                - mod3: (batch, 3) logits
                - mod5: (batch, 5) logits
                - mod8: (batch, 8) logits
                - mod10: (batch, 10) logits
        """
        # Handle single vector input
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Shared encoding
        h = self.shared_encoder(x)
        
        # Multi-task predictions
        return {
            "sign": self.sign_head(h),
            "mag": self.mag_head(h),      # Now (batch, 4096) logits
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
    def log_value_to_bin(log_val: float) -> int:
        """Convert log10 magnitude to bin index."""
        if log_val <= 0:
            return 0
        bin_idx = int((log_val / MAX_LOG_VALUE) * NUM_MAGNITUDE_BINS)
        return max(0, min(bin_idx, NUM_MAGNITUDE_BINS - 1))
    
    @staticmethod
    def bin_to_log_range(bin_idx: int) -> Tuple[float, float]:
        """Convert bin index to (log_min, log_max) range."""
        bin_width = MAX_LOG_VALUE / NUM_MAGNITUDE_BINS
        log_min = bin_idx * bin_width
        log_max = (bin_idx + 1) * bin_width
        return (log_min, log_max)
    
    def batch_reconstruct(
        self,
        feature_vectors: torch.Tensor,
        top_k_bins: int = 5,
        neighbors: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fully vectorized bin-based reconstruction with Top-K magnitude proposals.
        
        Args:
            feature_vectors: (B, input_dim) feature vectors
            top_k_bins: Number of top magnitude bins to explore (default: 5)
            neighbors: Neighbor radius around bin center ±neighbors (default: 3)
        
        Returns:
            Tuple of:
                - best_integers: (B,) reconstructed integers
                - best_scores: (B,) confidence scores
        """
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            batch_size = feature_vectors.shape[0]
            device = feature_vectors.device
            
            # 1. Get predictions for entire batch
            preds = self.forward(feature_vectors)
            
            # 2. Get sign
            sign_idx = torch.argmax(preds["sign"], dim=1)  # (B,)
            sign_value = sign_idx - 1  # Map to [-1, 0, 1]
            
            # 3. Top-K magnitude bins
            mag_probs = F.softmax(preds["mag"], dim=1)  # (B, 4096)
            topk_probs, topk_bins = torch.topk(mag_probs, top_k_bins, dim=1)  # (B, K)
            
            # 4. VECTORIZED GRID GENERATION
            # Convert bins to magnitude centers
            bin_width = MAX_LOG_VALUE / NUM_MAGNITUDE_BINS
            log_centers = (topk_bins.float() + 0.5) * bin_width  # (B, K) log10 values
            mag_centers = torch.pow(10, log_centers).long()  # (B, K) integer magnitudes
            
            # Create offset grid: (-neighbors, ..., +neighbors)
            num_neighbors = 2 * neighbors + 1
            offsets = torch.arange(-neighbors, neighbors + 1, device=device)  # (Neighbors,)
            
            # Broadcast: (B, K, 1) + (1, 1, Neighbors) -> (B, K, Neighbors)
            candidates_grid = mag_centers.unsqueeze(2) + offsets.view(1, 1, -1)
            
            # Apply sign: (B, 1, 1) * (B, K, Neighbors) -> (B, K, Neighbors)
            candidates_grid = sign_value.view(-1, 1, 1) * candidates_grid
            
            # Flatten to (B, Total_Candidates) where Total = K * Neighbors
            total_candidates = top_k_bins * num_neighbors
            candidates = candidates_grid.view(batch_size, total_candidates)  # (B, T)
            
            # 5. VECTORIZED SCORING
            # Initialize scores
            scores = torch.zeros(batch_size, total_candidates, device=device, dtype=torch.float32)
            
            # Add magnitude bin probabilities
            # Map candidates back to bins and gather their log probabilities
            mag_log_probs = torch.log(mag_probs + 1e-10)  # (B, 4096)
            candidate_abs = torch.abs(candidates.float())
            candidate_log = torch.log10(candidate_abs + 1e-10)  # Use log10
            candidate_bins = (candidate_log / bin_width).long().clamp(0, NUM_MAGNITUDE_BINS - 1)
            mag_scores = torch.gather(mag_log_probs, 1, candidate_bins)  # (B, T)
            scores += mag_scores
            
            # Modulo scoring for all 8 heads
            mod_heads = ['mod3', 'mod5', 'mod7', 'mod8', 'mod10', 'mod11', 'mod13', 'mod100']
            mod_bases = [3, 5, 7, 8, 10, 11, 13, 100]
            
            for head, base in zip(mod_heads, mod_bases):
                mod_log_probs = F.log_softmax(preds[head], dim=1)  # (B, base)
                residues = candidates % base  # (B, T)
                mod_scores = torch.gather(mod_log_probs, 1, residues)  # (B, T)
                scores += mod_scores
            
            # 6. VECTORIZED SELECTION
            best_indices = torch.argmax(scores, dim=1)  # (B,)
            best_integers = torch.gather(candidates, 1, best_indices.unsqueeze(1)).squeeze(1)
            best_scores = torch.gather(scores, 1, best_indices.unsqueeze(1)).squeeze(1)
        
        if was_training:
            self.train()
        
        return best_integers, best_scores

