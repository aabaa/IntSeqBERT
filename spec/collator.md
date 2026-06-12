# `src/intseq_bert/collator.py` Implementation Specification

## 1. Overview

This module performs dynamic masking and batch construction for IntSeqBERT and the Vanilla Transformer. It pads variable-length samples loaded by `OEISDataset` and applies masking for masked sequence modelling.

### Design Principles

- **Dynamic Masking**: generate different mask patterns every epoch.
- **Mask Flag Strategy**: distinguish real zero values from masked Magnitude values.
- **Origin Shift Strategy**: use the origin `(0, 0)` as the mask representation for Sin/Cos streams.
- **Dual Model Support**: support both IntSeqBERT and Vanilla Transformer inputs.

---

## 2. Dependencies

```python
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence

from . import config
```

### Config Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `MASK_PROB` | 0.15 | Mask probability |
| `PAD_VALUE_FEATURE` | -9999.0 | Sentinel padding value for features |
| `IGNORE_INDEX` | -100 | Label value ignored by loss functions |
| `MAG_RAW_DIM` | 4 | Input Magnitude dimension |
| `MAG_EXTENDED_DIM` | 5 | Magnitude dimension with mask flag |
| `MOD_FEATURE_DIM` | 200 | Modulo Sin/Cos dimension |
| `NUM_MODULI` | 100 | Number of moduli |
| `KEY_MAG_FEATURES` | `"mag_features"` | Data key |
| `KEY_MOD_FEATURES` | `"mod_features"` | Data key |
| `KEY_MOD_INTEGERS` | `"mod_integers"` | Data key |
| `KEY_OEIS_ID` | `"oeis_id"` | Data key |
| `VANILLA_VOCAB_SIZE` | 20003 | Vanilla token vocabulary size |
| `VANILLA_PAD_TOKEN_ID` | 0 | Padding token ID |
| `VANILLA_MASK_TOKEN_ID` | 1 | Mask token ID |
| `VANILLA_UNK_TOKEN_ID` | 2 | Unknown token ID |

---

## 3. Class Design

### 3.1 `OEISCollator`

Dataclass used as a PyTorch `DataLoader` `collate_fn`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mask_prob` | `float` | `config.MASK_PROB` | Mask probability |

Input contract:

```python
[
    {
        "mag_features": Tensor(L1, 4),
        "mod_features": Tensor(L1, 200),
        "mod_integers": Tensor(L1, 100),
        "oeis_id": "A000045",
    },
    ...
]
```

Output contract:

```python
{
    # IntSeqBERT inputs
    "mag_inputs":     Tensor(B, L, 5),
    "mod_inputs":     Tensor(B, L, 200),
    "mag_labels":     Tensor(B, L, 4),
    "mod_labels":     Tensor(B, L, 100),

    # Vanilla Transformer inputs
    "token_ids":      Tensor(B, L),
    "token_labels":   Tensor(B, L),

    # Shared
    "attention_mask": Tensor(B, L),       # 1=valid, 0=padding
    "mask_matrix":    Tensor(B, L),       # True=masked
    "oeis_ids":       List[str],
}
```

---

## 4. Processing Flow

### Step 1: Input Validation

```python
if not batch:
    raise ValueError("Batch is empty.")

required_keys = [KEY_MAG_FEATURES, KEY_MOD_FEATURES, KEY_MOD_INTEGERS]
for key in required_keys:
    if key not in batch[0]:
        raise KeyError(f"Dataset must provide '{key}' for collator.")
```

### Step 2: Padding

```python
mag_padded = pad_sequence(mag_list, batch_first=True, padding_value=config.PAD_VALUE_FEATURE)
mod_padded = pad_sequence(mod_list, batch_first=True, padding_value=config.PAD_VALUE_FEATURE)
mod_int_padded = pad_sequence(mod_int_list, batch_first=True, padding_value=IGNORE_INDEX)
```

### Step 3: Attention Mask

```python
valid_mask_bool = (mag_padded[..., 0] != config.PAD_VALUE_FEATURE)
attention_mask = valid_mask_bool.long()
```

### Step 4: Mask Matrix

```python
prob_matrix = torch.full((B, L), mask_prob)
prob_matrix[~valid_mask_bool] = 0.0
mask_matrix = torch.bernoulli(prob_matrix).bool()
```

### Step 5: Magnitude Stream

Mask Flag Strategy:

```text
Unmasked (valid): [log_val, sign+, sign-, sign0, 0]
Masked (valid):   [0,       0,     0,     0,     1]
Padding:          [0,       0,     0,     0,     0]
```

Padding positions are explicitly zeroed so sentinel values such as `PAD_VALUE_FEATURE = -9999.0` never reach the model.

```python
is_masked_channel = torch.zeros((B, L, 1))
is_masked_channel[mask_matrix] = 1.0
mag_inputs = torch.cat([mag_padded, is_masked_channel], dim=2)

valid_unmasked = valid_mask_bool & (~mask_matrix)
content_keep_mask = valid_unmasked.unsqueeze(-1).float()
mag_inputs[..., :4] *= content_keep_mask
```

### Step 6: Modulo Stream

Origin Shift Strategy:

```text
Unmasked: [sin(theta), cos(theta), ...]  # on the unit circle
Masked:   [0,          0,          ...]  # origin, outside the unit circle
```

```python
mod_inputs = mod_padded * content_keep_mask
```

### Step 6.5: Token IDs for the Vanilla Transformer

Token ID layout:

```text
0: PAD
1: MASK
2: UNK
3-20002: integers 0-19999
```

Priority:

1. If raw `"numbers"` are available, generate exact token IDs.
2. Otherwise, recover approximate integer values from log magnitude as a fallback.

The exact path clamps integers outside int64 range to a sentinel before converting to `torch.long`, preventing overflow for OEIS values that exceed int64.

```python
max_int = VANILLA_VOCAB_SIZE - 3 - 1  # 19999
in_vocab_mask = (numbers_padded >= 0) & (numbers_padded <= max_int)
token_ids = torch.where(
    in_vocab_mask,
    numbers_padded + 3,
    VANILLA_UNK_TOKEN_ID,
)
token_labels = token_ids.clone()
token_ids = torch.where(mask_matrix, VANILLA_MASK_TOKEN_ID, token_ids)
token_ids = torch.where(valid_mask_bool, token_ids, VANILLA_PAD_TOKEN_ID)
token_labels = torch.where(mask_matrix, token_labels, IGNORE_INDEX)
```

### Step 7: Labels

```python
mag_labels = mag_padded.clone()

mod_labels = mod_int_padded.clone()
mod_labels[~mask_matrix] = IGNORE_INDEX
```

---

## 5. Masking Strategy Details

### 5.1 Why the Mask Flag Is Needed

Magnitude has valid zeros, for example `x=0` gives `log_val=0`. Plain zero padding cannot distinguish the value zero from padding, and sentinel values do not define how to represent masked tokens.

The fifth channel `is_masked` solves this:

- `is_masked=0`: valid unmasked data.
- `is_masked=1`: masked position.

### 5.2 Why Origin Shift Works

Sin/Cos embeddings live on the unit circle. The zero vector is the origin and is never a valid unit-circle point. Setting masked positions to `(0, 0)` therefore creates an unambiguous mask representation.

---

## 6. Error Handling

| Situation | Exception | Message |
|-----------|-----------|---------|
| Empty batch | `ValueError` | `"Batch is empty."` |
| Missing key | `KeyError` | `"Dataset must provide 'mag_features' for collator."` |

---

## 7. Usage Example

```python
from torch.utils.data import DataLoader
from intseq_bert.collator import OEISCollator
from intseq_bert.loader import load_dataset

dataset = load_dataset("std", "train")
collator = OEISCollator(mask_prob=0.15)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collator)

for batch in dataloader:
    mag_inputs = batch["mag_inputs"]
    mod_inputs = batch["mod_inputs"]
    attention_mask = batch["attention_mask"]

    outputs = model(
        mag_inputs,
        mod_inputs,
        src_key_padding_mask=(attention_mask == 0),
    )
```

---

## 8. Design Decisions

| Decision | Rationale |
|----------|-----------|
| `dataclass` | Simple state management; only `mask_prob` is stored |
| Dynamic masking | Improves generalization compared with static masking |
| Never mask padding positions | Predicting padding is meaningless |
| `IGNORE_INDEX` for unmasked Modulo labels | Lets CrossEntropy ignore those positions automatically |
| Keep all Magnitude labels | Regression loss filters with `mask_matrix` |
| Vectorized token ID generation | Avoids Python loops |
| Integer fallback from log magnitude | Provides approximate behavior when raw integers are unavailable |
| `VOCAB_SIZE = 20003` | Practical upper bound for an 8 GB GPU environment |
