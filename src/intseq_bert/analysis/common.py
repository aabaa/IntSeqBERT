"""
common.py:
Common utilities shared across analysis modules.

Contains model wrappers, split functions, and other shared code.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List
from abc import ABC, abstractmethod

from intseq_bert import config


# ==========================================
# Constants
# ==========================================

# Clip range for log variance to prevent numerical issues
LOG_VAR_CLIP_MIN = -10
LOG_VAR_CLIP_MAX = 10


# ==========================================
# Model Wrapper (Abstract Base)
# ==========================================

class ModelWrapper(ABC):
    """Abstract base class for model wrappers."""
    
    @abstractmethod
    def predict(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Run inference and return predictions.
        
        Returns:
            {
                "mag_mu": (B, L),
                "mag_log_var": (B, L),
                "sign_logits": (B, L, 3),
                "mod_logits": (B, L, ~5150)
            }
        """
        pass
    
    def predict_with_details(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Returns predictions including optional attention weights.
        Default implementation calls predict().
        """
        return self.predict(batch)
    
    def supports_attention(self) -> bool:
        """Whether this model supports attention weight extraction."""
        return False
    
    @abstractmethod
    def get_mod_log_probs(self, mod_logits: torch.Tensor) -> List[torch.Tensor]:
        """Return log-probabilities for each modulus."""
        pass


class IntSeqWrapper(ModelWrapper):
    """Wrapper for IntSeqForPreTraining model."""
    
    def __init__(self, checkpoint_path: str, device: str):
        from intseq_bert.models import IntSeqForPreTraining
        self.model = IntSeqForPreTraining.from_checkpoint(checkpoint_path)
        self.model.to(device).eval()
        self.device = device
    
    def predict(self, batch: Dict) -> Dict:
        with torch.no_grad():
            outputs = self.model(
                mag_features=batch["mag_inputs"].to(self.device),
                mod_features=batch["mod_inputs"].to(self.device),
                src_key_padding_mask=(batch["attention_mask"] == 0).to(self.device)
            )
        return outputs["predictions"]
    
    def get_mod_log_probs(self, mod_logits: torch.Tensor) -> List[torch.Tensor]:
        split_logits = split_mod_logits(mod_logits)
        return [F.log_softmax(logits, dim=-1) for logits in split_logits]


def create_model_wrapper(
    model_type: str,
    checkpoint_path: str,
    device: str
) -> ModelWrapper:
    """Factory function to create appropriate model wrapper."""
    if model_type == "intseq":
        return IntSeqWrapper(checkpoint_path, device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ==========================================
# Utility Functions
# ==========================================

def split_mod_logits(mod_logits: torch.Tensor) -> List[torch.Tensor]:
    """
    Split concatenated mod logits into per-modulus tensors.
    
    Args:
        mod_logits: (N, L, sum(MOD_RANGE)) or (L, sum(MOD_RANGE))
    
    Returns:
        List of tensors, one per modulus (length = len(MOD_RANGE))
    """
    splits = []
    offset = 0
    for m in config.MOD_RANGE:
        if mod_logits.dim() == 2:
            splits.append(mod_logits[:, offset:offset+m])
        else:
            splits.append(mod_logits[:, :, offset:offset+m])
        offset += m
    return splits


def get_mod_index(modulus: int) -> int:
    """
    Get the index of a modulus in MOD_RANGE.
    
    Args:
        modulus: The modulus value (2-101)
    
    Returns:
        Index in MOD_RANGE
    
    Raises:
        ValueError: If modulus not in MOD_RANGE
    """
    try:
        return config.MOD_RANGE.index(modulus)
    except ValueError:
        raise ValueError(
            f"Modulus {modulus} not in MOD_RANGE. "
            f"Valid range: {config.MOD_RANGE[0]}-{config.MOD_RANGE[-1]}"
        )
