"""
models.py:
Re-export module for backward compatibility and unified model access.

This module provides a single import point for all model classes:
- IntSeqBERT models (IntSeqForPreTraining, IntSeqModel, IntSeqEmbeddings)
- Vanilla Transformer models (VanillaTransformerForPreTraining, VanillaModel, VanillaEmbeddings)
- Ablation models (AblationForPreTraining, AblationModel, AblationEmbeddings)
- Base classes (BaseForPreTraining, BaseTransformerModel, BaseEmbeddings, etc.)

Usage:
    from intseq_bert.models import IntSeqForPreTraining, VanillaTransformerForPreTraining
    from intseq_bert import models
    model = models.IntSeqForPreTraining(...)
"""

# Base models (shared components)
from .base_models import (
    ModLogitsMixin,
    generate_sinusoidal_encoding,
    PositionalEncoding,
    BasePreTrainedModel,
    BaseEmbeddings,
    BaseTransformerModel,
    BaseForPreTraining,
)

# IntSeqBERT models
from .intseq_models import (
    IntSeqEmbeddings,
    IntSeqModel,
    IntSeqForPreTraining,
)

# Vanilla Transformer models
from .vanilla_models import (
    VanillaEmbeddings,
    VanillaModel,
    VanillaTransformerForPreTraining,
)

# Ablation models (Magnitude only, no Modulo stream)
from .ablation_models import (
    AblationEmbeddings,
    AblationModel,
    AblationForPreTraining,
)

__all__ = [
    # Base models
    "ModLogitsMixin",
    "generate_sinusoidal_encoding",
    "PositionalEncoding",
    "BasePreTrainedModel",
    "BaseEmbeddings",
    "BaseTransformerModel",
    "BaseForPreTraining",
    # IntSeq models
    "IntSeqEmbeddings",
    "IntSeqModel",
    "IntSeqForPreTraining",
    # Vanilla models
    "VanillaEmbeddings",
    "VanillaModel",
    "VanillaTransformerForPreTraining",
    # Ablation models
    "AblationEmbeddings",
    "AblationModel",
    "AblationForPreTraining",
]
