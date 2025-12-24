"""
IntSeqBERT: BERT-style Transformer model for integer sequence representation learning.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    
    Args:
        d_model: Dimension of the model
        max_len: Maximum sequence length
        dropout: Dropout rate
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)


class IntSeqBERT(nn.Module):
    """
    BERT-style model for integer sequence representation learning.
    
    Takes 27-dimensional feature vectors and uses Transformer encoder
    with masked reconstruction objective.
    
    Args:
        input_dim: Input feature dimension (default: 27)
        d_model: Transformer hidden dimension (default: 128)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of encoder layers (default: 6)
        dim_feedforward: FFN hidden dimension (default: 512)
        max_len: Maximum sequence length (default: 5000)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        input_dim: int = 27,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 1. Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Important: (batch, seq, feature) format
            norm_first=True    # Pre-LN for better training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Prediction head (regression to reconstruct features)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, input_dim)
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask_matrix: Optional[torch.Tensor] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass of IntSeqBERT.
        
        Args:
            inputs: Masked input tensor of shape (batch_size, seq_len, input_dim)
            attention_mask: Padding mask of shape (batch_size, seq_len)
                           1 = valid token, 0 = padding
            labels: Original (unmasked) tensor of shape (batch_size, seq_len, input_dim)
                   Required for loss computation
            mask_matrix: Boolean mask of shape (batch_size, seq_len)
                        True = masked position (compute loss here)
                        False = unmasked position
        
        Returns:
            Dictionary containing:
                - prediction: Reconstructed tensor (batch_size, seq_len, input_dim)
                - loss: MSE loss on masked positions (scalar) or None
        """
        # Step 1: Project input to d_model dimension
        # inputs: (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
        x = self.input_proj(inputs)
        
        # Step 2: Add positional encoding
        x = self.pos_encoder(x)
        
        # Step 3: Convert attention_mask to src_key_padding_mask
        # attention_mask: 1=valid, 0=pad
        # src_key_padding_mask: True=pad, False=valid (inverse)
        src_key_padding_mask = (attention_mask == 0)
        
        # Step 4: Pass through Transformer encoder
        # x: (batch, seq_len, d_model)
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Step 5: Prediction head
        # encoded: (batch, seq_len, d_model) -> (batch, seq_len, input_dim)
        prediction = self.prediction_head(encoded)
        
        # Step 6: Compute loss if labels and mask_matrix provided
        loss = None
        if labels is not None and mask_matrix is not None:
            loss = self._compute_masked_loss(prediction, labels, mask_matrix)
        
        return {
            "prediction": prediction,
            "loss": loss
        }
    
    def _compute_masked_loss(
        self,
        prediction: torch.Tensor,
        labels: torch.Tensor,
        mask_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE loss only on masked positions.
        
        Args:
            prediction: Model predictions (batch, seq_len, input_dim)
            labels: Ground truth labels (batch, seq_len, input_dim)
            mask_matrix: Boolean mask (batch, seq_len), True = compute loss
        
        Returns:
            Scalar loss tensor
        """
        # Expand mask to match feature dimension
        # mask_matrix: (batch, seq_len) -> (batch, seq_len, 1)
        mask_expanded = mask_matrix.unsqueeze(-1)
        
        # Compute squared error
        squared_error = (prediction - labels) ** 2  # (batch, seq_len, input_dim)
        
        # Apply mask and sum
        masked_error = squared_error * mask_expanded  # Zero out non-masked positions
        
        # Count number of masked positions
        num_masked = mask_matrix.sum()
        
        # Edge case: if no positions are masked, return 0 loss
        if num_masked == 0:
            return torch.tensor(0.0, device=prediction.device)
        
        # Average over masked positions and features
        loss = masked_error.sum() / (num_masked * self.input_dim)
        
        return loss
