"""
intseq_models.py:
IntSeqBERT model implementation using Dual Stream Embedding with FiLM fusion.
Inherits from base classes defined in base_models.py.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from . import config
from .base_models import (
    BaseEmbeddings,
    BaseTransformerModel,
    BaseForPreTraining,
    ModLogitsMixin,
    generate_sinusoidal_encoding,
)


# ============================================================
# IntSeqBERT Embeddings (Dual Stream + FiLM)
# ============================================================


class IntSeqEmbeddings(BaseEmbeddings):
    """
    Dual Stream Input Layer.
    Fuses Magnitude (Continuous) and Modulo (Discrete/Periodic) streams using FiLM.
    """
    
    def __init__(
        self,
        d_model: int = config.D_MODEL,
        dropout: float = config.DROPOUT,
        max_len: int = config.MAX_SEQUENCE_LENGTH
    ):
        super().__init__(d_model, dropout, max_len)
        
        # Projections (config-driven: v3 adds MLP option)
        if config.INPUT_PROJ_TYPE == 'mlp':
            self.mag_proj = nn.Sequential(
                nn.Linear(config.MAG_EXTENDED_DIM, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )
        else:
            # 'linear' (v2 style)
            self.mag_proj = nn.Linear(config.MAG_EXTENDED_DIM, d_model)
        
        self.mod_proj = nn.Linear(config.MOD_FEATURE_DIM, d_model)
        
        # FiLM Generators (Conditioning on Modulo stream)
        self.film_scale = nn.Linear(d_model, d_model)  # Gamma
        self.film_shift = nn.Linear(d_model, d_model)  # Beta
        
        # Positional Encoding (Fixed)
        self.register_buffer("pos_encoding", generate_sinusoidal_encoding(max_len, d_model))
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize FiLM weights to be close to identity."""
        nn.init.zeros_(self.film_scale.weight)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)
        
        # Standard init for projections (handle both Linear and Sequential)
        if isinstance(self.mag_proj, nn.Sequential):
            for module in self.mag_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        else:
            nn.init.xavier_uniform_(self.mag_proj.weight)
        nn.init.xavier_uniform_(self.mod_proj.weight)
    
    def forward(
        self,
        mag_features: torch.Tensor,
        mod_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            mag_features: (B, L, MAG_EXTENDED_DIM)
            mod_features: (B, L, MOD_FEATURE_DIM)
        
        Returns:
            embeddings: (B, L, d_model)
        """
        # IMPORTANT: Force FP32 for magnitude stream to prevent FP16 overflow
        mag_features = mag_features.float()
        
        # 1. Projection - disable autocast to keep computation in FP32
        with torch.amp.autocast(device_type='cuda', enabled=False):
            h_mag = self.mag_proj(mag_features)  # (B, L, D)
        
        # Modulo stream passes through ReLU
        h_mod = torch.relu(self.mod_proj(mod_features))  # (B, L, D)
        
        # 1.5. Pre-FiLM Dropout (v3: regularization before fusion)
        if config.USE_PRE_FILM_DROPOUT:
            h_mag = self.dropout(h_mag)
            h_mod = self.dropout(h_mod)
        
        # 2. FiLM Generation
        gamma = self.film_scale(h_mod)  # (B, L, D)
        beta = self.film_shift(h_mod)   # (B, L, D)
        
        # 3. Modulation (Feature-wise Affine Transformation)
        gamma = gamma.float()
        beta = beta.float()
        h_fused = (1.0 + gamma) * h_mag + beta
        
        # 4. Add Position Encoding
        seq_len = h_fused.size(1)
        h_out = h_fused + self.pos_encoding[:, :seq_len, :].float()
        
        # 5. Norm & Dropout
        h_out = self.layer_norm(h_out)
        h_out = self.dropout(h_out)
        
        return h_out


# ============================================================
# IntSeqBERT Model (Backbone)
# ============================================================


class IntSeqModel(BaseTransformerModel):
    """
    IntSeqBERT Transformer Encoder Backbone.
    Wraps IntSeqEmbeddings and PyTorch's TransformerEncoder.
    """
    
    def __init__(
        self,
        d_model: int = config.D_MODEL,
        nhead: int = config.NHEAD,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT
    ):
        super().__init__(d_model, nhead, num_layers, dropout)
        
        self.embeddings = IntSeqEmbeddings(d_model, dropout)
    
    def forward(
        self,
        mag_features: torch.Tensor,
        mod_features: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            mag_features: (B, L, MAG_EXTENDED_DIM)
            mod_features: (B, L, MOD_FEATURE_DIM)
            src_key_padding_mask: (B, L) BoolTensor, True where padding
        
        Returns:
            last_hidden_state: (B, L, d_model)
        """
        # Embed
        x = self.embeddings(mag_features, mod_features)
        
        # Encode
        last_hidden_state = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        return last_hidden_state


# ============================================================
# IntSeqBERT for Pre-Training
# ============================================================


class IntSeqForPreTraining(BaseForPreTraining):
    """
    IntSeqBERT Training Wrapper with Multi-Task Heads.
    Implements Automatic Weighted Loss for magnitude, sign, and modulo prediction.
    """
    
    def __init__(
        self,
        d_model: int = config.D_MODEL,
        nhead: int = config.NHEAD,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT
    ):
        super().__init__(d_model)
        
        self.bert = IntSeqModel(d_model, nhead, num_layers, dropout)
        
        # Store config for checkpoint
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
    
    def forward(
        self,
        mag_features: torch.Tensor,
        mod_features: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Forward pass with optional loss computation.
        
        Args:
            mag_features: (B, L, MAG_EXTENDED_DIM)
            mod_features: (B, L, MOD_FEATURE_DIM)
            src_key_padding_mask: (B, L) BoolTensor
            labels: Optional dict with mask_map, mag_targets, sign_targets, mod_targets
        
        Returns:
            Dict with predictions and optional loss
        """
        # 1. Backbone Forward
        hidden_state = self.bert(mag_features, mod_features, src_key_padding_mask)
        
        # 2. Predictions
        # IMPORTANT: Run magnitude head in FP32 to prevent FP16 overflow
        with torch.amp.autocast(device_type='cuda', enabled=False):
            mag_preds = self.mag_head(hidden_state.float())  # (B, L, 2)
            mag_mu = mag_preds[..., 0]
            mag_log_var = mag_preds[..., 1]
        
        sign_logits = self.sign_head(hidden_state)           # (B, L, 3)
        unified_mod_logits = self.mod_head(hidden_state)     # (B, L, SumMods)
        
        outputs = {
            "predictions": {
                "mag_mu": mag_mu,
                "mag_log_var": mag_log_var,
                "sign_logits": sign_logits,
                "mod_logits": unified_mod_logits
            }
        }
        
        # 3. Loss Calculation (Training Only)
        if labels is not None:
            mask_map = labels["mask_map"]  # (B, L) Boolean mask
            
            # Filter by mask
            target_mag = labels["mag_targets"][mask_map].float()
            pred_mu = mag_mu[mask_map]
            pred_log_var = mag_log_var[mask_map]
            
            target_sign = labels["sign_targets"][mask_map]
            pred_sign = sign_logits[mask_map]
            
            target_mods = labels["mod_targets"][mask_map]  # (N_masked, 100)
            pred_mods = unified_mod_logits[mask_map]       # (N_masked, SumMods)
            
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
