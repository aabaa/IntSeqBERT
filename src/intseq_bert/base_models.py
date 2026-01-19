"""
base_models.py:
Shared components and base classes for IntSeqBERT and Vanilla Transformer.
Provides common infrastructure to ensure fair comparison experiments.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from . import config


# ============================================================
# Mixins
# ============================================================


class ModLogitsMixin:
    """Mixin providing mod logits splitting functionality."""
    
    def _split_mod_logits(self, logits: torch.Tensor) -> List[torch.Tensor]:
        """
        Splits unified mod logits into a list of tensors for each modulus.
        
        Args:
            logits: (B, L, sum(MOD_RANGE)) or (N, sum(MOD_RANGE))
        
        Returns:
            List of tensors, one per modulus
        """
        return torch.split(logits, config.MOD_RANGE, dim=-1)


# ============================================================
# Shared Components
# ============================================================


def generate_sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    Generates standard sinusoidal positional encoding table.
    
    Args:
        max_len: Maximum sequence length
        d_model: Model dimension
    
    Returns:
        Positional encoding tensor of shape (1, max_len, d_model)
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(config.POSITIONAL_ENCODING_BASE) / d_model)
    )
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # Add batch dimension for broadcasting: (1, L, D)
    return pe.unsqueeze(0)


class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding.
    Can be used by any model requiring positional embeddings.
    """
    
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = generate_sinusoidal_encoding(max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) tensor
        
        Returns:
            (B, L, D) tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
# Abstract Base Classes
# ============================================================


class BasePreTrainedModel(nn.Module):
    """
    Base class for all pre-trained models.
    Provides weight initialization and checkpoint loading.
    """
    
    def __init__(self):
        super().__init__()
    
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights for common module types.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: str = "cpu",
        **kwargs
    ) -> "BasePreTrainedModel":
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file
            device: Device to load model on
            **kwargs: Additional arguments for model initialization
        
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Extract config from checkpoint if available
        ckpt_config = checkpoint.get("config", {})
        
        # Merge with kwargs (kwargs take precedence)
        init_kwargs = {**ckpt_config, **kwargs}
        
        # Create model instance
        model = cls(**init_kwargs)
        
        # Load state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        return model.to(device).eval()


class BaseEmbeddings(nn.Module):
    """
    Base class for embedding layers.
    Subclasses should implement the forward method.
    """
    
    def __init__(
        self,
        d_model: int,
        dropout: float,
        max_len: int
    ):
        super().__init__()
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


class BaseTransformerModel(BasePreTrainedModel):
    """
    Base Transformer Encoder backbone.
    Subclasses should define self.embeddings before calling super().__init__().
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
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * config.FEEDFORWARD_MULTIPLIER,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


class BaseForPreTraining(BasePreTrainedModel, ModLogitsMixin):
    """
    Base class for pre-training models with multi-task heads.
    Provides common prediction heads for fair comparison.
    """
    
    def __init__(
        self,
        d_model: int = config.D_MODEL
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # --- Shared Prediction Heads (for diagnostic comparison) ---
        
        # Magnitude Head (Heteroscedastic Regression)
        self.mag_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)  # [mu, log_var]
        )
        
        # Sign Head (Classification: +, -, 0)
        self.sign_head = nn.Linear(d_model, config.NUM_SIGN_CLASSES)
        
        # Modulo Head (Unified Classification)
        total_mod_classes = sum(config.MOD_RANGE)
        self.mod_head = nn.Linear(d_model, total_mod_classes)
        
        # Fixed Loss Weights
        self.register_buffer("loss_weights", torch.tensor([
            config.LOSS_WEIGHT_MAG,
            config.LOSS_WEIGHT_SIGN,
            config.LOSS_WEIGHT_MOD
        ]))
    
    def _compute_mag_loss(
        self,
        pred_mu: torch.Tensor,
        pred_log_var: torch.Tensor,
        target_mag: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute magnitude loss with heteroscedastic option.
        
        Args:
            pred_mu: Predicted mean (N,)
            pred_log_var: Predicted log variance (N,)
            target_mag: Target magnitude (N,)
        
        Returns:
            Scalar loss tensor
        """
        # Force FP32 for stability
        pred_mu = pred_mu.float()
        pred_log_var = pred_log_var.float()
        target_mag = target_mag.float()
        
        # Compute reconstruction loss
        if config.MAG_LOSS_TYPE == 'huber':
            recon_loss = nn.functional.smooth_l1_loss(pred_mu, target_mag, reduction='none', beta=1.0)
        elif config.MAG_LOSS_TYPE == 'mse':
            recon_loss = nn.functional.mse_loss(pred_mu, target_mag, reduction='none')
        elif config.MAG_LOSS_TYPE == 'l1':
            recon_loss = nn.functional.l1_loss(pred_mu, target_mag, reduction='none')
        else:
            raise ValueError(f"Unknown MAG_LOSS_TYPE: {config.MAG_LOSS_TYPE}")
        
        # Apply heteroscedastic weighting if enabled
        if config.USE_HETEROSCEDASTIC_LOSS:
            pred_log_var = torch.clamp(pred_log_var, config.LOG_VAR_CLIP_MIN, config.LOG_VAR_CLIP_MAX)
            precision = torch.exp(-pred_log_var)
            loss_per_sample = 0.5 * pred_log_var + recon_loss * precision
            loss_per_sample = torch.clamp(loss_per_sample, max=100.0)
            return loss_per_sample.mean()
        else:
            return recon_loss.mean()
    
    def _compute_mod_loss(
        self,
        pred_logits: torch.Tensor,
        target_mods: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normalized modulo loss across all moduli.
        
        Args:
            pred_logits: (N, sum(MOD_RANGE))
            target_mods: (N, num_moduli)
        
        Returns:
            Scalar loss tensor
        """
        pred_split = self._split_mod_logits(pred_logits)
        
        total_loss = 0.0
        for i, logits in enumerate(pred_split):
            m = config.MOD_RANGE[i]
            loss_m = nn.functional.cross_entropy(logits, target_mods[:, i])
            # Normalize by log(m)
            total_loss += loss_m / math.log(m)
        
        return total_loss / len(config.MOD_RANGE)
