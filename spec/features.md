# `src/intseq_bert/features.py` Implementation Specification

## 1. Overview

This module converts raw integer sequences into tensors used for model training. It sits at the upstream end of the data pipeline and is called by `preprocess.py`.

### Output Streams

| Stream | Content | Shape |
|--------|---------|-------|
| **Magnitude** | Log10-scale magnitude + sign one-hot | `(L, 4)` |
| **Modulo Sin/Cos** | Unit-circle residue embeddings | `(L, 200)` |
| **Modulo Integers** | Integer residues for classification labels | `(L, 100)` |

---

## 2. Dependencies

```python
import math
from typing import Dict, List

import torch

from . import config
```

### Config Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_SEQUENCE_LENGTH` | 128 | Truncation limit |
| `MAG_RAW_DIM` | 4 | Magnitude output dimension |
| `MOD_FEATURE_DIM` | 200 | Modulo Sin/Cos output dimension |
| `NUM_MODULI` | 100 | Number of moduli |
| `MOD_RANGE` | `list(range(2, 102))` | Moduli `[2, 3, ..., 101]` |
| `KEY_MAG_FEATURES` | `"mag_features"` | Output key |
| `KEY_MOD_FEATURES` | `"mod_features"` | Output key |
| `KEY_MOD_INTEGERS` | `"mod_integers"` | Output key |

---

## 3. Function Design

### 3.1 `compute_magnitude_features`

Converts an integer sequence into Magnitude features.

```python
def compute_magnitude_features(sequence: List[int]) -> torch.Tensor
```

| Item | Type | Description |
|------|------|-------------|
| Input | `List[int]` | Integer sequence |
| Output | `Tensor(L, 4)` | Magnitude features |

For each integer `x`, the function creates:

```text
[log_val, sign_plus, sign_minus, sign_zero]
```

| Field | Formula | Description |
|-------|---------|-------------|
| `log_val` | `1.0 + log10(abs(x))` | Log-scale absolute value |
| `sign_plus` | `1.0 if x > 0 else 0.0` | Positive-sign flag |
| `sign_minus` | `1.0 if x < 0 else 0.0` | Negative-sign flag |
| `sign_zero` | `1.0 if x == 0 else 0.0` | Zero flag |

Special cases:

| Case | `log_val` | Signs |
|------|-----------|-------|
| `x = 0` | `0.0` | `[0, 0, 1]` |
| `x = 1` | `1.0` | `[1, 0, 0]` |
| `x = -10` | `2.0` | `[0, 1, 0]` |
| `x = 10^1000` | `1001.0` through fallback | `[1, 0, 0]` |

Overflow protection:

```python
try:
    log_val = 1.0 + math.log10(val_abs)
except OverflowError:
    log_val = float(len(str(val_abs)))
```

Empty inputs return `torch.zeros((0, config.MAG_RAW_DIM), dtype=torch.float32)`.

### 3.2 `compute_modulo_features`

Converts an integer sequence into Modulo features and integer labels.

```python
def compute_modulo_features(sequence: List[int]) -> tuple[torch.Tensor, torch.Tensor]
```

| Item | Type | Description |
|------|------|-------------|
| Input | `List[int]` | Integer sequence |
| Output 1 | `Tensor(L, 200)` | Sin/Cos embeddings |
| Output 2 | `Tensor(L, 100)` | Integer residue labels |

For each integer `x` and modulus `m in [2, 101]`:

```text
r = x % m
theta = (2*pi*r) / m
features = [sin(theta), cos(theta)]
```

`mod_features` has shape `(L, 200)`:

```text
[sin_m2, cos_m2, sin_m3, cos_m3, ..., sin_m101, cos_m101]
```

`mod_integers` has shape `(L, 100)`:

```text
[r_m2, r_m3, r_m4, ..., r_m101]
```

Python's `%` operator returns a non-negative residue:

```python
-5 % 3  # => 1, not -2
```

Empty inputs return zero tensors with shapes `(0, MOD_FEATURE_DIM)` and `(0, NUM_MODULI)`.

### 3.3 `process_sequence`

Main entry point for processing one sequence.

```python
def process_sequence(sequence: List[int]) -> Dict[str, torch.Tensor]
```

Processing flow:

1. Truncate to `MAX_SEQUENCE_LENGTH` if necessary.
2. Compute Magnitude features.
3. Compute Modulo features.
4. Pack the outputs into a dictionary.

Output:

```python
{
    "mag_features": Tensor(L, 4),
    "mod_features": Tensor(L, 200),
    "mod_integers": Tensor(L, 100),
}
```

Padding is intentionally not performed here; the collator handles batching.

---

## 4. Data Types

| Tensor | dtype | Rationale |
|--------|-------|-----------|
| `mag_features` | `float32` | Continuous values |
| `mod_features` | `float32` | Continuous Sin/Cos values |
| `mod_integers` | `long` (`int64`) | Labels for CrossEntropy |

---

## 5. Mathematical Background for Unit-Circle Embeddings

If residues are represented as categorical labels, `0` and `m-1` look far apart. Under modular arithmetic they are adjacent because `0 == m (mod m)`.

Sin/Cos embeddings represent periodicity naturally on the unit circle:

```text
r = 0    -> theta = 0      -> (sin=0, cos=1)
r = m/4  -> theta = pi/2   -> (sin=1, cos=0)
r = m/2  -> theta = pi     -> (sin=0, cos=-1)
r = m-1  -> theta ~= 2*pi  -> close to r=0
```

---

## 6. Usage Example

```python
from intseq_bert.features import process_sequence

sequence = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
result = process_sequence(sequence)

print(result["mag_features"].shape)   # torch.Size([10, 4])
print(result["mod_features"].shape)   # torch.Size([10, 200])
print(result["mod_integers"].shape)   # torch.Size([10, 100])

print(result["mag_features"][0])      # tensor([0., 0., 0., 1.])
```

---

## 7. Design Decisions

| Decision | Rationale |
|----------|-----------|
| `1 + log10(abs(x))` | Avoids `log=0` for `x=1` and keeps positive values positive |
| One-hot sign | Represents sign as an explicit independent channel |
| Sin/Cos embedding | Represents periodicity continuously and preserves adjacency across the wrap boundary |
| Keep integer residues | Required for classification losses |
| Truncate only; no padding | Keeps feature extraction separate from batching |
