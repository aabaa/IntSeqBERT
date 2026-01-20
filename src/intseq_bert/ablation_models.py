"""
ablation_models.py:
Ablation model implementation using Magnitude-only input (no Modulo stream, no FiLM).
Used to demonstrate the importance of the Modulo stream in IntSeqBERT.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

from . import config
from .base_models import (
    BasePreTrainedModel,
    BaseForPreTraining,
    ModLogitsMixin,
    PositionalEncoding,
)


# ============================================================
# Ablation Embeddings (Magnitude Only)
# ============================================================

class AblationEmbeddings(nn.Module):
    """
    Magnitude feature only embedding.
    No Modulo stream, No FiLM fusion.
    
    This provides a fair comparison to IntSeqBERT by removing the
    periodic/modular information while keeping everything else the same.
    """
    
    def __init__(
        self,
        d_model: int = config.D_MODEL,
        dropout: float = config.DROPOUT,
        max_len: int = config.MAX_SEQUENCE_LENGTH
    ):
        super().__init__()
        self.d_model = d_model
        
        # Magnitude input (5 dims) -> d_model
        # Same MLP structure as IntSeqBERT for fair comparison
        self.mag_proj = nn.Sequential(
            nn.Linear(config.MAG_EXTENDED_DIM, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Common components (same as IntSeqBERT)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mag_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mag_features: (B, L, MAG_EXTENDED_DIM)
        
        Returns:
            embeddings: (B, L, d_model)
        """
        # 1. Projection (force FP32 for numerical stability)
        with torch.amp.autocast('cuda', enabled=False):
            x = self.mag_proj(mag_features.float())
            
        # 2. Positional Encoding
        x = self.pos_encoding(x)
        
        # 3. LayerNorm & Dropout
        return self.dropout(self.layer_norm(x))


# ============================================================
# Ablation Model (Backbone)
# ============================================================

class AblationModel(BasePreTrainedModel):
    """
    Ablation model backbone: Transformer Encoder with Magnitude-only embeddings.
    
    Architecture is identical to IntSeqBERT except for the embedding layer,
    which only uses Magnitude features (no Modulo stream, no FiLM fusion).
    """
    
    def __init__(
        self,
        d_model: int = config.D_MODEL,
        nhead: int = config.NHEAD,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Ablation Embedding (Magnitude only)
        self.embeddings = AblationEmbeddings(d_model, dropout, config.MAX_SEQUENCE_LENGTH)
        
        # Standard Transformer Encoder (same as IntSeqBERT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * config.FEEDFORWARD_MULTIPLIER,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        mag_features: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            mag_features: (B, L, MAG_EXTENDED_DIM)
            src_key_padding_mask: (B, L) BoolTensor, True where padding
        
        Returns:
            last_hidden_state: (B, L, d_model)
        """
        emb = self.embeddings(mag_features)
        return self.encoder(emb, src_key_padding_mask=src_key_padding_mask)


# ============================================================
# Ablation for Pre-Training
# ============================================================

class AblationForPreTraining(BaseForPreTraining):
    """
    Ablation model for pre-training.
    
    Uses Magnitude-only input but predicts all tasks (magnitude, sign, modulo).
    This allows direct comparison with IntSeqBERT to measure the impact of
    removing the Modulo stream.
    
    Key difference from IntSeqBERT:
    - mod_features argument is accepted for interface compatibility but IGNORED
    - Only mag_features is used for prediction
    """
    
    def __init__(
        self,
        d_model: int = config.D_MODEL,
        nhead: int = config.NHEAD,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT
    ):
        super().__init__(d_model)
        self.backbone = AblationModel(d_model, nhead, num_layers, dropout)
        
        # Store config for checkpoint saving
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        self.apply(self._init_weights)

    def forward(
        self,
        mag_features: torch.Tensor,
        mod_features: torch.Tensor,  # Received but IGNORED (for interface compatibility)
        src_key_padding_mask: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Forward pass with optional loss computation.
        
        Args:
            mag_features: (B, L, MAG_EXTENDED_DIM) - Required
            mod_features: (B, L, MOD_FEATURE_DIM) - **IGNORED** (interface compatibility only)
            src_key_padding_mask: (B, L) BoolTensor
            labels: Optional dict with mask_map, mag_targets, sign_targets, mod_targets
        
        Returns:
            Dict with predictions and optional loss
        """
        # Backbone uses ONLY mag_features (mod_features is ignored)
        hidden_state = self.backbone(mag_features, src_key_padding_mask)
        
        # Predictions (same heads as IntSeqBERT)
        # Force FP32 for magnitude head stability
        with torch.amp.autocast(device_type='cuda', enabled=False):
            mag_preds = self.mag_head(hidden_state.float())
            mag_mu = mag_preds[..., 0]
            mag_log_var = mag_preds[..., 1]
        
        sign_logits = self.sign_head(hidden_state)
        unified_mod_logits = self.mod_head(hidden_state)
        
        outputs = {
            "predictions": {
                "mag_mu": mag_mu,
                "mag_log_var": mag_log_var,
                "sign_logits": sign_logits,
                "mod_logits": unified_mod_logits
            }
        }
        
        # Loss Calculation (identical to IntSeqBERT)
        if labels is not None:
            mask_map = labels["mask_map"]
            
            # Filter by mask
            target_mag = labels["mag_targets"][mask_map].float()
            pred_mu = mag_mu[mask_map]
            pred_log_var = mag_log_var[mask_map]
            
            target_sign = labels["sign_targets"][mask_map]
            pred_sign = sign_logits[mask_map]
            
            target_mods = labels["mod_targets"][mask_map]
            pred_mods = unified_mod_logits[mask_map]
            
            # Compute losses using base class methods
            loss_mag = self._compute_mag_loss(pred_mu, pred_log_var, target_mag)
            loss_sign = nn.functional.cross_entropy(pred_sign, target_sign)
            loss_mod = self._compute_mod_loss(pred_mods, target_mods)
            
            # Weighted sum
            w_mag, w_sign, w_mod = self.loss_weights
            weighted_loss = w_mag * loss_mag + w_sign * loss_sign + w_mod * loss_mod
            
            outputs["loss"] = weighted_loss
            outputs["loss_breakdown"] = {
                "raw_mag": loss_mag.detach(),
                "raw_sign": loss_sign.detach(),
                "raw_mod": loss_mod.detach(),
                "w_mag": w_mag,
                "w_sign": w_sign,
                "w_mod": w_mod
            }
        
        return outputs
