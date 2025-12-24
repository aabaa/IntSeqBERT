"""
Number-theoretic decoder for reconstructing integers from feature vectors.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from .features import log_magnitude


def inverse_magnitude(y: float) -> float:
    """
    Inverse transformation of log_magnitude.
    
    Args:
        y: Log-magnitude value (output from log_magnitude)
    
    Returns:
        Reconstructed magnitude (absolute value)
    """
    if y < 0.5:
        return 0.0
    return math.exp(y - 1.0)


class NumberTheoreticDecoder(nn.Module):
    """
    Decoder that reconstructs integers from 27-dimensional feature vectors.
    
    Uses multi-task learning to predict:
    - Sign (classification: -, 0, +)
    - Magnitude (regression: log-scale)
    - Modulo residues (classification: mod 3, 5, 8, 10)
    
    Reconstruction uses probabilistic Chinese Remainder Theorem search.
    
    Args:
        input_dim: Input feature dimension (default: 27)
        hidden_dim: Hidden layer dimension (default: 256)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        input_dim: int = 27,
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
        self.sign_head = nn.Linear(hidden_dim, 3)      # 0:-, 1:0, 2:+
        self.mag_head = nn.Linear(hidden_dim, 1)       # Regression
        self.mod3_head = nn.Linear(hidden_dim, 3)      # mod 3
        self.mod5_head = nn.Linear(hidden_dim, 5)      # mod 5
        self.mod8_head = nn.Linear(hidden_dim, 8)      # mod 8
        self.mod10_head = nn.Linear(hidden_dim, 10)    # mod 10
    
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
            "mag": self.mag_head(h),
            "mod3": self.mod3_head(h),
            "mod5": self.mod5_head(h),
            "mod8": self.mod8_head(h),
            "mod10": self.mod10_head(h)
        }
    
    def reconstruct_value(
        self,
        features: torch.Tensor,
        search_window: int = 150,
        lambda_mag: float = 0.5
    ) -> Tuple[int, float]:
        """
        Reconstruct integer from feature vector using probabilistic CRT search.
        
        Args:
            features: Feature vector (27,) or (1, 27)
            search_window: Search radius around base estimate (default: 150)
            lambda_mag: Weight for magnitude penalty (default: 0.5)
        
        Returns:
            Tuple of (reconstructed_value, confidence)
                - reconstructed_value: Best integer candidate
                - confidence: Score difference between top 2 candidates
        """
        # Preserve original training mode
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            # Get predictions
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            preds = self.forward(features)
            
            # 1. Compute base estimate
            sign_logits = preds["sign"][0]  # (3,)
            sign_idx = torch.argmax(sign_logits).item()
            
            # Map: 0→-1, 1→0, 2→+1
            sign_value = sign_idx - 1
            
            # Magnitude prediction
            mag_pred = preds["mag"][0, 0].item()
            mag_value = inverse_magnitude(mag_pred)
            
            # Base estimate
            x_base = int(sign_value * mag_value)
            
            # 2. Create search window
            candidates = list(range(x_base - search_window, x_base + search_window + 1))
            
            # 3. Score each candidate
            scores = []
            
            # Get log probabilities for modulo predictions
            mod3_logprobs = F.log_softmax(preds["mod3"][0], dim=0)
            mod5_logprobs = F.log_softmax(preds["mod5"][0], dim=0)
            mod8_logprobs = F.log_softmax(preds["mod8"][0], dim=0)
            mod10_logprobs = F.log_softmax(preds["mod10"][0], dim=0)
            
            for c in candidates:
                # Compute score
                score = 0.0
                
                # Modulo term: sum of log probabilities
                score += mod3_logprobs[c % 3].item()
                score += mod5_logprobs[c % 5].item()
                score += mod8_logprobs[c % 8].item()
                score += mod10_logprobs[c % 10].item()
                
                # Magnitude term: penalize deviation from predicted magnitude
                # IMPORTANT: Use log_magnitude for consistency
                true_mag = log_magnitude([c])[0]
                mag_error = (true_mag - mag_pred) ** 2
                score -= lambda_mag * mag_error
                
                scores.append(score)
            
            # 4. Find best candidate
            scores_tensor = torch.tensor(scores)
            top_indices = torch.topk(scores_tensor, k=min(2, len(scores)))
            
            best_idx = top_indices.indices[0].item()
            best_candidate = candidates[best_idx]
            
            # Confidence: score difference
            if len(scores) >= 2:
                confidence = (top_indices.values[0] - top_indices.values[1]).item()
            else:
                confidence = top_indices.values[0].item()
        
        # Restore original training mode
        if was_training:
            self.train()
        
        return best_candidate, confidence
