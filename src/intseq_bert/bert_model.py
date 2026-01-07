"""
IntSeqBERT: BERT-style Transformer model for integer sequence representation learning.
Updated for Dual Stream Architecture (Magnitude + Mod Spectrum) + Multitask Classification.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any

# Define MOD_RANGE locally to avoid circular imports with decoder_model
MOD_RANGE = list(range(2, 102))

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            seq_len = self.pe.size(0)
            x = x[:, :seq_len, :]
            
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)


class IntSeqBERT(nn.Module):
    """
    Dual Stream BERT Model with Multitask Learning.
    
    1. Reconstruction Tasks (MSE):
       - Reconstruct Magnitude features (dim=5)
       - Reconstruct Mod Spectrum features (dim=200)
    
    2. Classification Tasks (CrossEntropy) [NEW]:
       - Predict exact residuals for Mod 2 to Mod 101
    
    Args:
        mag_dim: Magnitude feature dimension (default: 5)
        mod_dim: Mod spectrum feature dimension (default: 200)
        d_model: Transformer hidden dimension (default: 128)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of encoder layers (default: 6)
        dim_feedforward: FFN hidden dimension (default: 512)
        max_len: Maximum sequence length (default: 5000)
        dropout: Dropout rate (default: 0.1)
        multitask: Whether to include Mod classification heads (default: True)
    """
    
    def __init__(
        self,
        mag_dim: int = 5,
        mod_dim: int = 200,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        max_len: int = 5000,
        dropout: float = 0.1,
        multitask: bool = True
    ):
        super().__init__()
        
        self.mag_dim = mag_dim
        self.mod_dim = mod_dim
        self.d_model = d_model
        self.multitask = multitask
        
        # 1. Dual Input Projections
        self.mag_proj = nn.Linear(mag_dim, d_model)
        self.mod_proj = nn.Linear(mod_dim, d_model)
        
        # Fusion Norm
        self.fusion_norm = nn.LayerNorm(d_model)
        
        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Dual Prediction Heads (Reconstruction)
        self.mag_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mag_dim)
        )
        
        self.mod_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mod_dim)
        )

        # 5. Multitask Classification Heads [NEW]
        if self.multitask:
            self.mod_cls_heads = nn.ModuleDict({
                f"mod{m}": nn.Linear(d_model, m) for m in MOD_RANGE
            })
    
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[str] = None
    ) -> Tuple['IntSeqBERT', Dict]:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', {})
        
        model_args = {
            'mag_dim': config.get('mag_dim', 5),
            'mod_dim': config.get('mod_dim', 200),
            'd_model': config.get('d_model', 128),
            'nhead': config.get('nhead', 4),
            'num_layers': config.get('num_layers', 6),
            'dim_feedforward': config.get('dim_feedforward', 512),
            'max_len': config.get('max_len', 5000),
            'dropout': config.get('dropout', 0.1),
            'multitask': config.get('multitask', True) # Default to True for compatibility
        }
        
        model = cls(**model_args)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
        # strict=False allows loading old weights into new multitask architecture 
        # (missing keys for mod_cls_heads will be ignored initialized randomly)
        
        model = model.to(device)
        
        return model, checkpoint
    
    def forward(
        self,
        mag_inputs: torch.Tensor,
        mod_inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        mag_labels: Optional[torch.Tensor] = None,
        mod_labels: Optional[torch.Tensor] = None,
        mask_matrix: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Returns dictionary containing:
          - encoded_state: (B, L, D)
          - pred_mag: (B, L, 5)
          - pred_mod: (B, L, 200)
          - loss: Scalar (Reconstruction Loss only)
          - mod{N}: (B, L, N) Logits for classification (if multitask)
        """
        
        # Step 1: Embed and Fuse
        x_mag = self.mag_proj(mag_inputs)
        x_mod = self.mod_proj(mod_inputs)
        x = x_mag + x_mod
        x = self.fusion_norm(x)
        
        # Step 2: Positional Encoding
        x = self.pos_encoder(x)
        
        # Step 3: Transformer Encoder
        src_key_padding_mask = (attention_mask == 0)
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Step 4: Reconstruction Heads
        pred_mag = self.mag_head(encoded)
        pred_mod = self.mod_head(encoded)
        
        # Step 5: Compute Reconstruction Loss (Internal convenience)
        loss = None
        if mag_labels is not None and mod_labels is not None and mask_matrix is not None:
            loss_mag = self._compute_masked_loss(pred_mag, mag_labels, mask_matrix, self.mag_dim)
            loss_mod = self._compute_masked_loss(pred_mod, mod_labels, mask_matrix, self.mod_dim)
            loss = loss_mag + loss_mod
        
        results = {
            "encoded_state": encoded,
            "pred_mag": pred_mag,
            "pred_mod": pred_mod,
            "loss": loss
        }

        # Step 6: Multitask Heads (Logits)
        # Note: CE Loss calculation is delegated to the training script
        if self.multitask:
            for m in MOD_RANGE:
                # Reuse the contextualized embeddings to predict residuals
                results[f"mod{m}"] = self.mod_cls_heads[f"mod{m}"](encoded)
        
        return results
    
    def _compute_masked_loss(
        self,
        prediction: torch.Tensor,
        labels: torch.Tensor,
        mask_matrix: torch.Tensor,
        dim: int
    ) -> torch.Tensor:
        """
        Compute MSE loss only on masked positions.
        """
        mask_expanded = mask_matrix.unsqueeze(-1) # (B, L, 1)
        
        squared_error = (prediction - labels) ** 2
        masked_error = squared_error * mask_expanded
        
        num_masked = mask_matrix.sum()
        
        if num_masked == 0:
            return torch.tensor(0.0, device=prediction.device)
        
        # Normalize by number of masked tokens and dimensions
        loss = masked_error.sum() / (num_masked * dim)
        return loss