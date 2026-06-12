# `src/intseq_bert/intseq_models.py` Implementation Specification

## 1. Overview

This module defines the neural architecture for IntSeqBERT. Following the design style of HuggingFace Transformers, the implementation is organized into three layers:

1. `IntSeqEmbeddings`
2. `IntSeqModel`
3. `IntSeqForPreTraining`

Shared infrastructure such as base classes, heads, checkpoint loading, and loss helpers is documented in [base_models.md](./base_models.md).

Key features:

- **Dual-stream input**: separate magnitude and modulo streams.
- **FiLM fusion**: modulo features modulate magnitude representations.
- **FP32 critical paths**: magnitude-related computations are forced to FP32 for numerical stability.
- **Fixed multi-task loss weights**: `w_mag=1.0`, `w_sign=1.0`, `w_mod=2.0`.

---

## 2. Dependencies and Config

Libraries:

- `math`
- `torch`
- `torch.nn`

Config constants:

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAG_EXTENDED_DIM` | 5 | Magnitude input dimension, including `is_masked` |
| `MOD_FEATURE_DIM` | 200 | Modulo sin/cos input dimension |
| `MOD_RANGE` | `list(range(2, 102))` | Moduli 2 through 101 |
| `NUM_MODULI` | 100 | Number of moduli |
| `INPUT_PROJ_TYPE` | `"mlp"` | Magnitude projection type: `"linear"` or `"mlp"` |
| `USE_PRE_FILM_DROPOUT` | `True` | Apply dropout before FiLM |
| `DROPOUT` | `0.2` | Default dropout |
| `MAG_LOSS_TYPE` | `"huber"` | Magnitude loss type |
| `USE_HETEROSCEDASTIC_LOSS` | `False` | Optional uncertainty loss |

Sign-class indices:

```python
SIGN_POSITIVE = 0
SIGN_NEGATIVE = 1
SIGN_ZERO = 2
```

---

## 3. Class Design

### 3.1 `IntSeqEmbeddings`

Input layer that fuses magnitude and periodicity. The modulo stream generates FiLM scale and shift parameters that modulate the magnitude stream.

Constructor arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `d_model` | int | `config.D_MODEL` | Hidden dimension |
| `dropout` | float | `config.DROPOUT` | Dropout rate |
| `max_len` | int | `config.MAX_SEQUENCE_LENGTH` | Maximum sequence length |

Architecture:

```python
if INPUT_PROJ_TYPE == "mlp":
    mag_proj = Sequential(Linear(5, d_model), GELU(), Linear(d_model, d_model))
else:
    mag_proj = Linear(MAG_EXTENDED_DIM, d_model)

mod_proj = Linear(MOD_FEATURE_DIM, d_model)
film_scale = Linear(d_model, d_model)
film_shift = Linear(d_model, d_model)
pos_encoding = PositionalEncoding(d_model, dropout, max_len)
layer_norm = LayerNorm(d_model)
dropout = Dropout(dropout)
```

Initialization:

```python
nn.init.zeros_(self.film_scale.weight)
nn.init.zeros_(self.film_scale.bias)
```

Zero-initializing the FiLM scale prevents early training from destroying the magnitude representation.

Forward input:

- `mag_features`: `(B, L, 5)`
- `mod_features`: `(B, L, 200)`

Forward output:

- `embeddings`: `(B, L, d_model)`

Processing flow:

```text
1. Project magnitude features in FP32.
2. Project modulo features and apply ReLU.
3. Optionally apply pre-FiLM dropout to both streams.
4. Generate gamma and beta from the modulo stream.
5. Fuse: h_fused = (1 + gamma) * h_mag + beta.
6. Add positional encoding, LayerNorm, and dropout.
```

OEIS contains values around `10^210`, producing log values around 210. Magnitude-related projections and heads therefore run in FP32 to avoid unstable intermediate values under mixed precision.

### 3.2 `IntSeqModel`

Transformer Encoder backbone.

Constructor arguments:

| Argument | Type | Description |
|----------|------|-------------|
| `d_model` | int | Hidden dimension |
| `nhead` | int | Number of attention heads |
| `num_layers` | int | Number of encoder layers |
| `dropout` | float | Dropout rate |

Architecture:

```python
embeddings = IntSeqEmbeddings(d_model, dropout)
encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=d_model * 4,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    ),
    num_layers=num_layers,
)
```

Forward input:

- `mag_features`: `(B, L, 5)`
- `mod_features`: `(B, L, 200)`
- `src_key_padding_mask`: `(B, L)`, where `True` means padding.

Forward output:

- `last_hidden_state`: `(B, L, d_model)`

### 3.3 `IntSeqForPreTraining`

Pre-training wrapper with prediction heads and loss computation.

Prediction heads:

| Head | Structure | Output | Precision |
|------|-----------|--------|-----------|
| `mag_head` | `Linear(d_model, d_model) -> ReLU -> Linear(d_model, 2)` | `[mu, log(sigma^2)]` | FP32 |
| `sign_head` | `Linear(d_model, 3)` | `[Positive, Negative, Zero]` logits | FP16/FP32 |
| `mod_head` | `Linear(d_model, sum(MOD_RANGE))` | Concatenated modulo logits | FP16/FP32 |

The sign class order follows `SIGN_POSITIVE=0`, `SIGN_NEGATIVE=1`, and `SIGN_ZERO=2`.

Fixed loss weights:

```python
loss_weights = register_buffer(torch.tensor([1.0, 1.0, 2.0]))
```

The modulo task receives double weight to encourage periodic-structure learning. Earlier experiments used automatic weighted loss, but fixed weights were adopted after task collapse in the modulo objective.

Forward arguments:

| Argument | Shape | Required | Description |
|----------|-------|----------|-------------|
| `mag_features` | `(B, L, 5)` | yes | magnitude stream input |
| `mod_features` | `(B, L, 200)` | yes | modulo stream input |
| `src_key_padding_mask` | `(B, L)` | yes | Padding mask |
| `labels` | dict | no | Training labels |

`labels` dictionary:

| Key | Shape | Description |
|-----|-------|-------------|
| `mag_targets` | `(B, L)` Float | Original magnitude target |
| `sign_targets` | `(B, L)` Long | Class index: Pos, Neg, Zero |
| `mod_targets` | `(B, L, 100)` Long | Residues for each modulus |
| `mask_map` | `(B, L)` Bool | `True` for positions included in loss |

Output:

```python
{
  "loss": Tensor,  # only when labels are provided
  "predictions": {
    "mag_mu": (B, L),
    "mag_log_var": (B, L),
    "sign_logits": (B, L, NUM_SIGN_CLASSES),
    "mod_logits": (B, L, sum(MOD_RANGE)),
  },
  "loss_breakdown": {
    "raw_mag": Tensor,
    "raw_sign": Tensor,
    "raw_mod": Tensor,
    "w_mag": Tensor,
    "w_sign": Tensor,
    "w_mod": Tensor,
  },
}
```

---

## 4. Loss Computation

Losses are computed only on masked positions (`mask_map == True`).

### 4.1 Magnitude Loss

The loss type and uncertainty behavior are controlled through config.

| `MAG_LOSS_TYPE` | Loss | Notes |
|-----------------|------|-------|
| `"huber"` | SmoothL1Loss | Default, robust to outliers |
| `"mse"` | MSELoss | Squared error |
| `"l1"` | L1Loss | Absolute error |

Magnitude loss is forced to FP32:

```python
target_mag = labels["mag_targets"][mask_map].float()
pred_mu = mag_mu[mask_map].float()
pred_log_var = mag_log_var[mask_map].float()
```

When `USE_HETEROSCEDASTIC_LOSS` is enabled, the model uses a Gaussian NLL-style objective with clipped log variance. By default, uncertainty estimation is disabled for stability and the deterministic reconstruction loss is used.

### 4.2 Sign Loss

```python
L_sign = CrossEntropyLoss(sign_logits, sign_targets)
```

### 4.3 Modulo Loss

Each modulus-specific loss is normalized by the maximum entropy `log(m)`, so random prediction has comparable normalized loss across moduli.

```python
L_mod_list = []
for m in MOD_RANGE:
    loss_m = CrossEntropy(logits_m, targets[:, m])
    L_mod_list.append(loss_m / log(m))

L_mod = mean(L_mod_list)
```

`mod_head` output is split by `_split_mod_logits()`.

### 4.4 Total Loss

```text
L_total = w_mag * L_mag + w_sign * L_sign + w_mod * L_mod
```

The fixed defaults are `w_mag = 1.0`, `w_sign = 1.0`, and `w_mod = 2.0`.

---

## 5. Helper Methods

### `_split_mod_logits(logits: Tensor) -> List[Tensor]`

Splits the concatenated `mod_head` output into per-modulus logits.

```python
return torch.split(logits, config.MOD_RANGE, dim=-1)
```

### `_generate_sinusoidal_encoding(max_len: int, d_model: int) -> Tensor`

Generates fixed sinusoidal positional encodings.

---

## 6. Future Extensions

| Item | Description | Priority |
|------|-------------|----------|
| Relative Positional Encoding | Strengthen adjacent-term relation learning, e.g. RoPE | Medium |
| Gradient Checkpointing | Improve memory efficiency | Low |

---

## 7. Related Baselines

The Vanilla Transformer and Ablation model share the same encoder depth, hidden dimension, attention heads, and pre-training heads where possible. Their dedicated specifications are:

- [vanilla_models.md](./vanilla_models.md)
- [ablation_models.md](./ablation_models.md)
