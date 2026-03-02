"""
vanilla_models.py:
Vanilla Transformer implementation for baseline comparison with IntSeqBERT.
Uses standard token embeddings instead of Dual Stream + FiLM.
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
    PositionalEncoding,
)


# ============================================================
# Vanilla Embeddings (Standard Token Embedding)
# ============================================================


class VanillaEmbeddings(BaseEmbeddings):
    """
    Standard token embedding with positional encoding.
    Maps integer token IDs to dense vectors.
    """
    
    def __init__(
        self,
        d_model: int = config.D_MODEL,
        dropout: float = config.DROPOUT,
        max_len: int = config.MAX_SEQUENCE_LENGTH,
        vocab_size: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ):
        super().__init__(d_model, dropout, max_len)
        
        # Get vocab config with defaults
        self.vocab_size = vocab_size or getattr(config, "VANILLA_VOCAB_SIZE", 20003)
        self.pad_token_id = pad_token_id or getattr(config, "VANILLA_PAD_TOKEN_ID", 0)
        
        # Token embedding
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=d_model,
            padding_idx=self.pad_token_id
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        
        # Scaling factor
        self.scale = math.sqrt(d_model)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize embedding weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        # Zero out padding embedding
        if self.pad_token_id is not None:
            self.token_embedding.weight.data[self.pad_token_id].zero_()
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) LongTensor of token IDs
        
        Returns:
            embeddings: (B, L, d_model)
        """
        # Token embedding with scaling
        x = self.token_embedding(input_ids) * self.scale
        
        # Add positional encoding (includes dropout)
        x = self.pos_encoding(x)
        
        # Layer norm
        x = self.layer_norm(x)
        
        return x


# ============================================================
# Vanilla Transformer Model (Backbone)
# ============================================================


class VanillaModel(BaseTransformerModel):
    """
    Vanilla Transformer Encoder Backbone.
    Standard architecture without Dual Stream or FiLM.
    """
    
    def __init__(
        self,
        d_model: int = config.D_MODEL,
        nhead: int = config.NHEAD,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT,
        vocab_size: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ):
        super().__init__(d_model, nhead, num_layers, dropout)
        
        self.embeddings = VanillaEmbeddings(
            d_model=d_model,
            dropout=dropout,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) LongTensor of token IDs
            src_key_padding_mask: (B, L) BoolTensor, True where padding
        
        Returns:
            last_hidden_state: (B, L, d_model)
        """
        # Embed
        x = self.embeddings(input_ids)
        
        # Encode
        last_hidden_state = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        return last_hidden_state


# ============================================================
# Vanilla Transformer for Pre-Training
# ============================================================


class VanillaTransformerForPreTraining(BaseForPreTraining):
    """
    Vanilla Transformer with multi-task heads for pre-training.
    
    Main task: Token ID prediction (lm_head)
    Diagnostic tasks: Magnitude, Sign, Modulo prediction (for comparison with IntSeqBERT)
    """
    
    def __init__(
        self,
        d_model: int = config.D_MODEL,
        nhead: int = config.NHEAD,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT,
        vocab_size: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ):
        super().__init__(d_model)
        
        # Get vocab config
        self.vocab_size = vocab_size or getattr(config, "VANILLA_VOCAB_SIZE", 20003)
        self.pad_token_id = pad_token_id or getattr(config, "VANILLA_PAD_TOKEN_ID", 0)
        
        # Backbone
        self.backbone = VanillaModel(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id
        )
        
        # Main task head: Token prediction
        self.lm_head = nn.Linear(d_model, self.vocab_size)
        
        # Store config
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Forward pass with optional loss computation.
        
        Args:
            input_ids: (B, L) LongTensor of token IDs
            src_key_padding_mask: (B, L) BoolTensor, True where padding
            labels: Optional dict with mask_map, token_targets, and diagnostic targets
        
        Returns:
            Dict with predictions and optional loss
        """
        # 1. Backbone Forward
        hidden_state = self.backbone(input_ids, src_key_padding_mask)
        
        # 2. Main Task Prediction
        logits = self.lm_head(hidden_state)  # (B, L, vocab_size)
        
        # 3. Diagnostic Predictions (for comparison with IntSeqBERT)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            mag_preds = self.mag_head(hidden_state.float())  # (B, L, 2)
            mag_mu = mag_preds[..., 0]
            mag_log_var = mag_preds[..., 1]
        
        sign_logits = self.sign_head(hidden_state)          # (B, L, 3)
        mod_logits = self.mod_head(hidden_state)            # (B, L, sum(MOD_RANGE))
        
        outputs = {
            "predictions": {
                "logits": logits,              # Main: token prediction
                "mag_mu": mag_mu,              # Diagnostic
                "mag_log_var": mag_log_var,    # Diagnostic
                "mod_logits": mod_logits,      # Diagnostic
                "sign_logits": sign_logits     # Diagnostic
            }
        }
        
        # 4. Loss Calculation (Training Only)
        if labels is not None:
            mask_map = labels["mask_map"]  # (B, L) Boolean mask
            
            # Main loss: Token prediction
            if "token_targets" in labels:
                target_tokens = labels["token_targets"][mask_map]
                pred_logits = logits[mask_map]
                loss_lm = nn.functional.cross_entropy(
                    pred_logits,
                    target_tokens,
                    ignore_index=self.pad_token_id
                )
            else:
                loss_lm = torch.tensor(0.0, device=input_ids.device)
            
            # Diagnostic losses
            if "mag_targets" in labels:
                target_mag = labels["mag_targets"][mask_map].float()
                pred_mu = mag_mu[mask_map]
                pred_log_var = mag_log_var[mask_map]
                loss_mag = self._compute_mag_loss(pred_mu, pred_log_var, target_mag)
            else:
                loss_mag = torch.tensor(0.0, device=input_ids.device)
            
            if "sign_targets" in labels:
                target_sign = labels["sign_targets"][mask_map]
                pred_sign = sign_logits[mask_map]
                loss_sign = nn.functional.cross_entropy(pred_sign, target_sign)
            else:
                loss_sign = torch.tensor(0.0, device=input_ids.device)
            
            if "mod_targets" in labels:
                target_mods = labels["mod_targets"][mask_map]
                pred_mods = mod_logits[mask_map]
                loss_mod = self._compute_mod_loss(pred_mods, target_mods)
            else:
                loss_mod = torch.tensor(0.0, device=input_ids.device)
            
            # Total loss: LM is main, others are diagnostic (weighted lower)
            w_mag, w_sign, w_mod = self.loss_weights
            diagnostic_weight = 0.1  # Scale down diagnostic losses
            
            total_loss = loss_lm + diagnostic_weight * (
                w_mag * loss_mag + w_sign * loss_sign + w_mod * loss_mod
            )
            
            outputs["loss"] = total_loss
            outputs["loss_breakdown"] = {
                "raw_lm": loss_lm.detach(),
                "raw_mag": loss_mag.detach(),
                "raw_sign": loss_sign.detach(),
                "raw_mod": loss_mod.detach()
            }
        
        return outputs
