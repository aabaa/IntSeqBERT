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
LOG_VAR_CLIP_MIN = config.LOG_VAR_CLIP_MIN
LOG_VAR_CLIP_MAX = config.LOG_VAR_CLIP_MAX


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
        import torch
        from intseq_bert.intseq_models import IntSeqForPreTraining
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model config from checkpoint or use defaults
        if "config" in checkpoint:
            model_config = checkpoint["config"]
            self.model = IntSeqForPreTraining(
                d_model=model_config.get("d_model", config.D_MODEL),
                nhead=model_config.get("nhead", config.NHEAD),
                num_layers=model_config.get("num_layers", config.NUM_LAYERS),
                dropout=model_config.get("dropout", config.DROPOUT)
            )
        else:
            # Use default config
            self.model = IntSeqForPreTraining()
        
        # Load state dict
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            # Assume checkpoint is the state dict itself
            self.model.load_state_dict(checkpoint)
        
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


class VanillaWrapper(ModelWrapper):
    """Wrapper for VanillaTransformerForPreTraining model."""
    
    def __init__(self, checkpoint_path: str, device: str):
        import torch
        from intseq_bert.vanilla_models import VanillaTransformerForPreTraining
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model config from checkpoint or use defaults
        if "config" in checkpoint:
            model_config = checkpoint["config"]
            self.model = VanillaTransformerForPreTraining(
                d_model=model_config.get("d_model", config.D_MODEL),
                nhead=model_config.get("nhead", config.NHEAD),
                num_layers=model_config.get("num_layers", config.NUM_LAYERS),
                dropout=model_config.get("dropout", config.DROPOUT),
                vocab_size=model_config.get("vocab_size", config.VANILLA_VOCAB_SIZE),
                pad_token_id=model_config.get("pad_token_id", config.VANILLA_PAD_TOKEN_ID)
            )
        else:
            # Use default config
            self.model = VanillaTransformerForPreTraining()
        
        # Load state dict
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            # Assume checkpoint is the state dict itself
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device).eval()
        self.device = device
    
    def predict(self, batch: Dict) -> Dict:
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["token_ids"].to(self.device),
                src_key_padding_mask=(batch["attention_mask"] == 0).to(self.device)
            )
        return outputs["predictions"]
    
    def get_mod_log_probs(self, mod_logits: torch.Tensor) -> List[torch.Tensor]:
        split_logits = split_mod_logits(mod_logits)
        return [F.log_softmax(logits, dim=-1) for logits in split_logits]


class AblationWrapper(ModelWrapper):
    """Wrapper for AblationForPreTraining model.
    
    Uses only magnitude features for prediction (modulo features are ignored).
    This is used to demonstrate the importance of the Modulo stream.
    """
    
    def __init__(self, checkpoint_path: str, device: str):
        import torch
        from intseq_bert.ablation_models import AblationForPreTraining
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model config from checkpoint or use defaults
        if "config" in checkpoint:
            model_config = checkpoint["config"]
            self.model = AblationForPreTraining(
                d_model=model_config.get("d_model", config.D_MODEL),
                nhead=model_config.get("nhead", config.NHEAD),
                num_layers=model_config.get("num_layers", config.NUM_LAYERS),
                dropout=model_config.get("dropout", config.DROPOUT)
            )
        else:
            # Use default config
            self.model = AblationForPreTraining()
        
        # Load state dict
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            # Assume checkpoint is the state dict itself
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device).eval()
        self.device = device
    
    def predict(self, batch: Dict) -> Dict:
        with torch.no_grad():
            outputs = self.model(
                mag_features=batch["mag_inputs"].to(self.device),
                mod_features=batch["mod_inputs"].to(self.device),  # Ignored by model
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
    """
    Factory function to create appropriate model wrapper.
    
    Args:
        model_type: 'intseq', 'vanilla', or 'ablation'
        checkpoint_path: Path to checkpoint file
        device: Device to load model on ('cuda', 'cpu', etc.)
    
    Returns:
        ModelWrapper instance
    """
    if model_type == "intseq":
        return IntSeqWrapper(checkpoint_path, device)
    elif model_type == "vanilla":
        return VanillaWrapper(checkpoint_path, device)
    elif model_type == "ablation":
        return AblationWrapper(checkpoint_path, device)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Supported types: 'intseq', 'vanilla', 'ablation'"
        )


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
