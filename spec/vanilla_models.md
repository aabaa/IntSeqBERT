# `src/intseq_bert/vanilla_models.py` Implementation Specification

## 1. Overview

**Target file:** `src/intseq_bert/vanilla_models.py`

This module implements the Vanilla Transformer baseline used for comparison with IntSeqBERT. It follows the FACT-style approach of representing integers as discrete token IDs.

Shared base classes, loss helpers, and `_split_mod_logits` are documented in [base_models.md](./base_models.md).

| Class | Role |
|-------|------|
| `VanillaEmbeddings` | Integer-token embedding plus positional encoding |
| `VanillaModel` | Transformer Encoder backbone |
| `VanillaTransformerForPreTraining` | Pre-training wrapper with LM and diagnostic heads |

---

## 2. Config Constants

| Constant | Default | Description |
|----------|---------|-------------|
| `VANILLA_VOCAB_SIZE` | 20003 | Token vocabulary size: integers 0..19,999 plus three special tokens |
| `VANILLA_PAD_TOKEN_ID` | 0 | Padding token ID |
| `VANILLA_MASK_TOKEN_ID` | 1 | Mask token ID |
| `VANILLA_UNK_TOKEN_ID` | 2 | Unknown token ID |

Existing constants used: `D_MODEL`, `NHEAD`, `NUM_LAYERS`, `DROPOUT`, `MOD_RANGE`, and `MAX_SEQUENCE_LENGTH`.

---

## 3. Class Design

### 3.1 `VanillaEmbeddings`

```python
class VanillaEmbeddings(BaseEmbeddings):
    """Token embedding with positional encoding for Vanilla Transformer."""
```

| Argument | Type | Default |
|----------|------|---------|
| `d_model` | int | `config.D_MODEL` |
| `dropout` | float | `config.DROPOUT` |
| `max_len` | int | `config.MAX_SEQUENCE_LENGTH` |
| `vocab_size` | Optional[int] | `None` |
| `pad_token_id` | Optional[int] | `None` |

Components:

| Component | Definition |
|-----------|------------|
| `token_embedding` | `nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_token_id)` |
| `pos_encoding` | Sinusoidal encoding added to token embeddings |
| `layer_norm` | `nn.LayerNorm(d_model)` |
| `dropout` | `nn.Dropout(dropout)` |

### 3.2 `VanillaModel`

```python
class VanillaModel(BaseTransformerModel):
    """Transformer Encoder backbone for the Vanilla model."""
```

| Argument | Type | Default |
|----------|------|---------|
| `d_model` | int | `config.D_MODEL` |
| `nhead` | int | `config.NHEAD` |
| `num_layers` | int | `config.NUM_LAYERS` |
| `dropout` | float | `config.DROPOUT` |
| `vocab_size` | Optional[int] | `None` |
| `pad_token_id` | Optional[int] | `None` |

Components:

| Component | Definition |
|-----------|------------|
| `embeddings` | `VanillaEmbeddings(...)` |
| `encoder_layer` | `nn.TransformerEncoderLayer(batch_first=True, norm_first=True)` |
| `encoder` | `nn.TransformerEncoder(encoder_layer, num_layers)` |

### 3.3 `VanillaTransformerForPreTraining`

```python
class VanillaTransformerForPreTraining(BaseForPreTraining):
    """Vanilla Transformer with multi-task heads for pre-training."""
```

Components:

| Component | Definition | Description |
|-----------|------------|-------------|
| `backbone` | `VanillaModel(...)` | Encoder backbone |
| `lm_head` | `nn.Linear(d_model, self.vocab_size)` | Main token-prediction head |
| `mag_head` | `nn.Linear(d_model, 2)` | Diagnostic magnitude head |
| `sign_head` | `nn.Linear(d_model, 3)` | Diagnostic sign head |
| `mod_head` | `nn.Linear(d_model, sum(MOD_RANGE))` | Diagnostic modulo head |

---

## 4. Input and Output Shapes

Forward arguments:

| Class | Arguments | Shapes |
|-------|-----------|--------|
| `VanillaEmbeddings` | `input_ids` | `(B, L)` LongTensor |
| `VanillaModel` | `input_ids`, `src_key_padding_mask` | `(B, L)`, `(B, L)` Bool |
| `VanillaTransformerForPreTraining` | `input_ids`, `src_key_padding_mask`, `labels` | same plus optional label dict |

Forward outputs:

```python
# VanillaEmbeddings
embeddings: (B, L, d_model)

# VanillaModel
last_hidden_state: (B, L, d_model)

# VanillaTransformerForPreTraining
{
    "predictions": {
        "logits": (B, L, vocab_size),
        "mag_mu": (B, L),
        "mag_log_var": (B, L),
        "mod_logits": (B, L, 5150),
        "sign_logits": (B, L, 3),
    },
    "loss": Tensor or None,
    "loss_breakdown": {...} or None,
}
```

---

## 5. Loss Computation

| Head | Loss | Weight | Notes |
|------|------|--------|-------|
| `lm_head` | `CrossEntropyLoss(ignore_index=self.pad_token_id)` | 1.0 | Main objective |
| `mag_head` | `BaseForPreTraining._compute_mag_loss` | 0.1 | Diagnostic |
| `sign_head` | `CrossEntropyLoss` | 0.1 | Diagnostic |
| `mod_head` | Per-modulus `CrossEntropyLoss` | 0.2 effective | Diagnostic; `0.1 * w_mod` with `w_mod=2.0` |

Setting the diagnostic-head weights to `0.0` turns the model into a pure language-model baseline.

Masking follows the same masked language modeling strategy as IntSeqBERT:

1. Replace 15% of inputs with `VANILLA_MASK_TOKEN_ID`.
2. Predict the original token at masked positions.
3. Exclude unmasked positions from the LM loss.

---

## 6. Helper Methods

### `from_checkpoint`

Inherited from `BasePreTrainedModel.from_checkpoint`; see [base_models.md](./base_models.md).

### Tokenization

Integer-to-token mapping:

```python
def tokenize_number(n: int) -> int:
    if 0 <= n <= VANILLA_VOCAB_SIZE - 4:
        return n + 3  # offset for PAD, MASK, UNK
    return VANILLA_UNK_TOKEN_ID
```
