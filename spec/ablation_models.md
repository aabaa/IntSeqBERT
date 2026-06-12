# `src/intseq_bert/ablation_models.py` Implementation Specification

## 1. Overview

This module implements the Ablation model, which removes the Modulo stream and FiLM fusion from IntSeqBERT v3.

Purpose:

- Measure how much performance drops when only Magnitude information is provided.
- Quantify the contribution of the Modulo stream, especially for number-theoretic reasoning.

File:

- `src/intseq_bert/ablation_models.py`

Dependencies:

- `base_models.py` for shared base classes and heads.
- `config.py` for dimensions and hyperparameters.

---

## 2. Class Design

The Ablation model keeps the Transformer backbone and pre-training heads comparable to IntSeqBERT. Only the embedding layer is replaced.

### 2.1 `AblationEmbeddings`

Receives only Magnitude features and projects them to `d_model`. No Modulo projection and no FiLM fusion are used.

```python
class AblationEmbeddings(nn.Module):
    """
    Magnitude-only embedding.
    No Modulo stream and no FiLM fusion.
    """
```

Constructor:

| Argument | Type | Default |
|----------|------|---------|
| `d_model` | int | `config.D_MODEL` |
| `dropout` | float | `config.DROPOUT` |
| `max_len` | int | `config.MAX_SEQUENCE_LENGTH` |

Architecture:

```python
mag_proj = nn.Sequential(
    nn.Linear(config.MAG_EXTENDED_DIM, d_model),
    nn.GELU(),
    nn.Linear(d_model, d_model),
)
pos_encoding = PositionalEncoding(d_model, dropout, max_len)
layer_norm = nn.LayerNorm(d_model)
dropout = nn.Dropout(dropout)
```

Forward:

| Input | Shape |
|-------|-------|
| `mag_features` | `(B, L, MAG_EXTENDED_DIM)` |

Output:

| Output | Shape |
|--------|-------|
| `embeddings` | `(B, L, d_model)` |

Magnitude projection is run with autocast disabled for FP32 stability.

### 2.2 `AblationModel`

Transformer Encoder backbone that uses `AblationEmbeddings`.

```python
class AblationModel(BasePreTrainedModel):
    """Transformer Encoder with Magnitude-only embeddings."""
```

Architecture:

```python
embeddings = AblationEmbeddings(d_model, dropout, config.MAX_SEQUENCE_LENGTH)
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=d_model * config.FEEDFORWARD_MULTIPLIER,
    dropout=dropout,
    batch_first=True,
    norm_first=True,
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
```

Forward input:

- `mag_features`: `(B, L, MAG_EXTENDED_DIM)`
- `src_key_padding_mask`: `(B, L)` BoolTensor, `True` where padding.

Forward output:

- `last_hidden_state`: `(B, L, d_model)`

### 2.3 `AblationForPreTraining`

Pre-training wrapper. For fair comparison, it uses the same prediction heads as `IntSeqForPreTraining`, including the diagnostic Modulo head.

```python
class AblationForPreTraining(BaseForPreTraining):
    """
    Magnitude-only model that still predicts mag, sign, and mod tasks.
    """
```

Forward arguments:

| Argument | Shape | Behavior |
|----------|-------|----------|
| `mag_features` | `(B, L, MAG_EXTENDED_DIM)` | Used |
| `mod_features` | `(B, L, MOD_FEATURE_DIM)` | Accepted for interface compatibility, ignored |
| `src_key_padding_mask` | `(B, L)` | Used |
| `labels` | dict or `None` | Optional training labels |

Forward output:

```python
{
    "predictions": {
        "mag_mu": ...,
        "mag_log_var": ...,
        "sign_logits": ...,
        "mod_logits": ...,
    },
    "loss": ...,
    "loss_breakdown": ...,
}
```

Loss calculation reuses the same base-class helpers as IntSeqBERT:

- `_compute_mag_loss`
- `cross_entropy` for sign
- `_compute_mod_loss`

---

## 3. Runtime Integration

### 3.1 `train.py`

Add the `--model_type ablation` branch:

```python
def create_model(model_type: str, device: str):
    if model_type == "intseq":
        from intseq_bert.intseq_models import IntSeqForPreTraining
        return IntSeqForPreTraining().to(device)
    if model_type == "vanilla":
        from intseq_bert.vanilla_models import VanillaTransformerForPreTraining
        return VanillaTransformerForPreTraining().to(device)
    if model_type == "ablation":
        from intseq_bert.ablation_models import AblationForPreTraining
        return AblationForPreTraining().to(device)
    raise ValueError(f"Unknown model_type: {model_type}")
```

### 3.2 Analysis Scripts

Add `AblationWrapper` to `analysis/common.py` so analysis scripts can instantiate the model in the same way as IntSeq and Vanilla.

```python
class AblationWrapper(ModelWrapper):
    """Wrapper for Ablation model inference."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        from intseq_bert.ablation_models import AblationForPreTraining
        self.model = AblationForPreTraining.from_checkpoint(checkpoint_path, device)
        self.model.eval()
        self.device = device

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.model(
                mag_features=batch["mag_inputs"].to(self.device),
                mod_features=batch["mod_inputs"].to(self.device),  # ignored
                src_key_padding_mask=(batch["attention_mask"] == 0).to(self.device),
            )
        return outputs["predictions"]
```

---

## 4. Test Requirements

`tests/test_ablation_models.py` should cover:

| Test | Purpose |
|------|---------|
| `test_ablation_embeddings_forward` | `AblationEmbeddings` returns the expected shape |
| `test_ablation_model_forward` | `AblationModel` returns `(B, L, d_model)` |
| `test_ablation_for_pretraining_forward` | Predictions include `mag_mu`, `sign_logits`, and `mod_logits` |
| `test_ablation_loss_computation` | `loss` is returned when labels are provided |
| `test_mod_features_ignored` | Changing `mod_features` does not change outputs |
| `test_from_checkpoint` | Loading from a checkpoint succeeds |

---

## 5. Expected Experimental Behavior

| Metric | Expected behavior | Rationale |
|--------|-------------------|-----------|
| **Mod Accuracy** | Severe drop toward random | No Modulo information is provided |
| **Sign Accuracy** | Drop, especially for parity-related structure | `mod 2` information is removed |
| **Magnitude MAE** | Similar to IntSeqBERT | Magnitude is directly provided |
| **Solver Accuracy** | Drop | Periodic constraints are harder to recover |

The model separates "magnitude prediction" from "number-theoretic structure understanding" and provides evidence for the role of the Modulo stream.

---

## 6. Output Layout

```text
checkpoints/ablation_std/
├── best_model.pt
├── last_checkpoint.pt
└── history.csv

results/analysis/ablation/
├── overall_metrics.csv
└── figures/
    └── comparison.png
```
