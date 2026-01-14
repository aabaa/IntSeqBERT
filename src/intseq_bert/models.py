"""
models.py:
Core Neural Network definitions for IntSeqBERT.
Implements Dual Stream Embedding with FiLM fusion and Automatic Weighted Loss.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from . import config

# --- Helper Functions ---

def _generate_sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Generates standard sinusoidal positional encoding table."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(config.POSITIONAL_ENCODING_BASE) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # Add batch dimension for broadcasting: (1, L, D)
    return pe.unsqueeze(0)


# --- Core Classes ---

class IntSeqEmbeddings(nn.Module):
    """
    Dual Stream Input Layer.
    Fuses Magnitude (Continuous) and Modulo (Discrete/Periodic) streams using FiLM.
    """
    def __init__(self, d_model: int = config.D_MODEL, dropout: float = config.DROPOUT, max_len: int = config.MAX_SEQUENCE_LENGTH):
        super().__init__()
        
        # Projections
        self.mag_proj = nn.Linear(config.MAG_EXTENDED_DIM, d_model)
        self.mod_proj = nn.Linear(config.MOD_FEATURE_DIM, d_model)
        
        # FiLM Generators (Conditioning on Modulo stream)
        self.film_scale = nn.Linear(d_model, d_model) # Gamma
        self.film_shift = nn.Linear(d_model, d_model) # Beta
        
        # Post-processing
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Positional Encoding (Fixed)
        self.register_buffer("pos_encoding", _generate_sinusoidal_encoding(max_len, d_model))
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize FiLM weights to be close to identity."""
        nn.init.zeros_(self.film_scale.weight)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)
        
        # Standard init for projections
        nn.init.xavier_uniform_(self.mag_proj.weight)
        nn.init.xavier_uniform_(self.mod_proj.weight)

    def forward(self, mag_features: torch.Tensor, mod_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mag_features: (B, L, 5)
            mod_features: (B, L, 200)
        Returns:
            embeddings: (B, L, d_model)
        """
        # 1. Projection
        h_mag = self.mag_proj(mag_features) # (B, L, D)
        
        # Modulo stream passes through ReLU to increase expressivity before generating FiLM params
        h_mod = torch.relu(self.mod_proj(mod_features)) # (B, L, D)
        
        # 2. FiLM Generation
        gamma = self.film_scale(h_mod) # (B, L, D)
        beta = self.film_shift(h_mod)  # (B, L, D)
        
        # 3. Modulation (Feature-wise Affine Transformation)
        # h_fused = (1 + gamma) * h_mag + beta
        h_fused = (1.0 + gamma) * h_mag + beta
        
        # 4. Add Position Encoding
        seq_len = h_fused.size(1)
        h_out = h_fused + self.pos_encoding[:, :seq_len, :]
        
        # 5. Norm & Dropout
        h_out = self.layer_norm(h_out)
        h_out = self.dropout(h_out)
        
        return h_out


class IntSeqModel(nn.Module):
    """
    Base Transformer Encoder Backbone.
    Wraps IntSeqEmbeddings and PyTorch's TransformerEncoder.
    """
    def __init__(self, 
                 d_model: int = config.D_MODEL,
                 nhead: int = config.NHEAD,
                 num_layers: int = config.NUM_LAYERS,
                 dropout: float = config.DROPOUT):
        super().__init__()
        
        self.embeddings = IntSeqEmbeddings(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * config.FEEDFORWARD_MULTIPLIER,
            dropout=dropout,
            batch_first=True,
            norm_first=True # Pre-LN
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, 
                mag_features: torch.Tensor, 
                mod_features: torch.Tensor, 
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src_key_padding_mask: (B, L) BoolTensor, True where padding.
        """
        # Embed
        x = self.embeddings(mag_features, mod_features) # (B, L, D)
        
        # Encode
        # Note: PyTorch TransformerEncoder takes src_key_padding_mask
        last_hidden_state = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        return last_hidden_state


class IntSeqForPreTraining(nn.Module):
    """
    Training Wrapper with Multi-Task Heads and Automatic Weighted Loss.
    """
    def __init__(self, 
                 d_model: int = config.D_MODEL,
                 nhead: int = config.NHEAD,
                 num_layers: int = config.NUM_LAYERS,
                 dropout: float = config.DROPOUT):
        super().__init__()
        
        self.bert = IntSeqModel(d_model, nhead, num_layers, dropout)
        
        # --- Prediction Heads ---
        
        # 1. Magnitude Head (Heteroscedastic Regression)
        # Predicts Mean (mu) and Log-Variance (log_var)
        self.mag_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2) # [mu, log_var]
        )
        
        # 2. Sign Head (Classification: +, -, 0)
        self.sign_head = nn.Linear(d_model, config.NUM_SIGN_CLASSES)
        
        # 3. Modulo Head (Unified Classification)
        # Output dim is sum of all moduli (2+3+...+101) approx 5150
        total_mod_classes = sum(config.MOD_RANGE)
        self.mod_head = nn.Linear(d_model, total_mod_classes)
        
        # --- Fixed Loss Weights ---
        # Using fixed weights to prevent task collapse
        # Values from config: LOSS_WEIGHT_MAG, LOSS_WEIGHT_SIGN, LOSS_WEIGHT_MOD
        self.register_buffer("loss_weights", torch.tensor([
            config.LOSS_WEIGHT_MAG,
            config.LOSS_WEIGHT_SIGN,
            config.LOSS_WEIGHT_MOD
        ]))

    def _split_mod_logits(self, logits: torch.Tensor) -> List[torch.Tensor]:
        """Splits the unified mod logits into a list of tensors for each modulus."""
        # config.MOD_RANGE must be a list of integers [2, 3, ..., 101]
        return torch.split(logits, config.MOD_RANGE, dim=-1)

    def forward(self, 
                mag_features: torch.Tensor, 
                mod_features: torch.Tensor, 
                src_key_padding_mask: torch.Tensor,
                labels: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Union[torch.Tensor, Dict]]:
        
        # 1. Backbone Forward
        hidden_state = self.bert(mag_features, mod_features, src_key_padding_mask)
        
        # 2. Predictions
        mag_preds = self.mag_head(hidden_state)      # (B, L, 2)
        mag_mu = mag_preds[..., 0]
        mag_log_var = mag_preds[..., 1]
        
        sign_logits = self.sign_head(hidden_state)   # (B, L, 3)
        
        unified_mod_logits = self.mod_head(hidden_state) # (B, L, SumMods)
        
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
            mask_map = labels["mask_map"] # (B, L) Boolean mask (True where masked/predicting)
            
            # Filter targets and predictions by mask to compute loss only on masked tokens
            # This flattens the tensors: (N_masked, ...)
            
            # --- A. Magnitude Loss (Gaussian NLL) ---
            target_mag = labels["mag_targets"][mask_map]
            pred_mu = mag_mu[mask_map]
            pred_log_var = mag_log_var[mask_map]
            
            # NLL = 0.5 * log(sigma^2) + (y - mu)^2 / (2 * sigma^2)
            #     = 0.5 * log_var + (y - mu)^2 * 0.5 * exp(-log_var)
            precision = torch.exp(-pred_log_var)
            loss_mag = 0.5 * pred_log_var + 0.5 * (target_mag - pred_mu)**2 * precision
            loss_mag = loss_mag.mean()
            
            # --- B. Sign Loss (CrossEntropy) ---
            target_sign = labels["sign_targets"][mask_map]
            pred_sign = sign_logits[mask_map]
            loss_sign = nn.functional.cross_entropy(pred_sign, target_sign)
            
            # --- C. Modulo Loss (Normalized Mean CrossEntropy across all moduli) ---
            # Each modulus's loss is normalized by log(m) so random prediction loss = 1.0
            target_mods = labels["mod_targets"][mask_map] # (N_masked, 100)
            pred_mods_flat = unified_mod_logits[mask_map] # (N_masked, SumMods)
            
            # Split logits per modulus
            pred_mods_split = self._split_mod_logits(pred_mods_flat)
            
            total_mod_loss = 0.0
            for i, mod_logits in enumerate(pred_mods_split):
                # mod_logits: (N_masked, m)
                # target_mods[:, i]: (N_masked,)
                m = config.MOD_RANGE[i]
                loss_m = nn.functional.cross_entropy(mod_logits, target_mods[:, i])
                # Normalize by log(m) - random prediction CE is log(m)
                norm_loss_m = loss_m / math.log(m)
                total_mod_loss += norm_loss_m
                
            loss_mod = total_mod_loss / len(config.MOD_RANGE)
            
            # --- D. Fixed Weighted Sum ---
            # Weights: Mag=1.0, Sign=1.0, Mod=2.0
            w_mag, w_sign, w_mod = self.loss_weights
            
            weighted_loss = w_mag * loss_mag + w_sign * loss_sign + w_mod * loss_mod
            
            outputs["loss"] = weighted_loss
            
            # Optional: Return individual losses for monitoring
            outputs["loss_breakdown"] = {
                "raw_mag": loss_mag.detach(),
                "raw_sign": loss_sign.detach(),
                "raw_mod": loss_mod.detach(),
                "w_mag": w_mag,
                "w_sign": w_sign,
                "w_mod": w_mod
            }
            
        return outputs
