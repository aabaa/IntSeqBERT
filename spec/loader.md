# `src/intseq_bert/loader.py` Implementation Specification

## 1. Overview

This module loads OEIS feature files (`.pt`) and manages static dataset splits.

### Design Principles

| Principle | Description |
|-----------|-------------|
| **Separation of responsibilities** | Split generation (admin) and dataset loading (runtime) are separate |
| **Physical isolation** | Split results are stored as static text files |
| **No runtime shuffle** | `load_dataset` only reads pre-generated split files |
| **Deterministic splitting** | Splits are reproducible through `config.SEED` |
| **Fail fast** | Missing files and required keys raise immediately |

---

## 2. Dependencies

```python
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from . import config, schemas
```

### Config Constants

| Constant | Example | Purpose |
|----------|---------|---------|
| `DATA_ROOT` | `"data/oeis"` | Data root path |
| `SPLIT_DIR_NAME` | `"splits"` | Split directory name |
| `FEATURES_DIR_NAME` | `"features"` | Feature directory name |
| `SEED` | `42` | Random seed |
| `VAL_RATIO` | `0.10` | Validation ratio |
| `TEST_RATIO` | `0.10` | Test ratio |
| `KEY_MAG_FEATURES` | `"mag_features"` | Required key |
| `KEY_MOD_FEATURES` | `"mod_features"` | Required key |
| `KEY_OEIS_ID` | `"oeis_id"` | ID key |

---

## 3. Class Design

### 3.1 `OEISDataset`

`torch.utils.data.Dataset` implementation that lazily loads individual `.pt` files.

Constructor arguments:

| Argument | Type | Description |
|----------|------|-------------|
| `oeis_ids` | `List[str]` | OEIS IDs to load |
| `features_dir` | `Path` | Directory containing `.pt` files |

`__getitem__` output:

```python
{
    "mag_features": Tensor(L, MAG_RAW_DIM),      # (L, 4)
    "mod_features": Tensor(L, MOD_FEATURE_DIM),  # (L, 200)
    "mod_integers": Tensor(L, NUM_MODULI),       # (L, 100)
    "oeis_id": str,
}
```

Validation:

1. Missing feature files raise `FileNotFoundError`.
2. `KEY_MAG_FEATURES` and `KEY_MOD_FEATURES` are required.
3. `KEY_OEIS_ID` is injected into the loaded sample.

Example:

```python
dataset = OEISDataset(
    oeis_ids=["A000045", "A000040"],
    features_dir=Path("data/oeis/features"),
)
sample = dataset[0]
```

---

## 4. Function Design

### 4.1 `load_dataset`

Runtime-facing function that loads a dataset from pre-generated split files.

```python
def load_dataset(
    split_type: str,
    split_name: str,
    *,
    data_root: Optional[str] = None,
) -> OEISDataset
```

| Argument | Type | Example | Description |
|----------|------|---------|-------------|
| `split_type` | `str` | `"std"` | Split type / directory name |
| `split_name` | `str` | `"train"` | Split name: `train`, `val`, or `test` |
| `data_root` | `Optional[str]` | | Override for tests |

Processing flow:

1. Build `{data_root}/splits/{split_type}/{split_name}.txt`.
2. Raise `FileNotFoundError` if it does not exist.
3. Read one OEIS ID per line.
4. Return an `OEISDataset`.

Example:

```python
train_dataset = load_dataset("std", "train")
val_dataset = load_dataset("std", "val")
test_dataset = load_dataset("std", "test")
```

### 4.2 `create_splits`

Admin-facing function that filters JSONL records by tag and writes static split files.

```python
def create_splits(
    source_jsonl: str,
    output_split_type: str,
    include_tags: Optional[List[str]] = None,
    exclude_tags: Optional[List[str]] = None,
    *,
    data_root: Optional[str] = None,
)
```

| Argument | Type | Example | Description |
|----------|------|---------|-------------|
| `source_jsonl` | `str` | `"data/oeis/data.jsonl"` | Source JSONL |
| `output_split_type` | `str` | `"std"` | Output split directory |
| `include_tags` | `Optional[List[str]]` | `["core", "easy"]` | Include tags, OR semantics |
| `exclude_tags` | `Optional[List[str]]` | `["cons", "base"]` | Exclude tags, OR semantics |
| `data_root` | `Optional[str]` | | Override for tests |

Processing flow:

1. Load JSONL with strict `schemas.OEISRecord` parsing.
2. Apply tag filters:
   - `exclude_tags`: remove records matching any tag.
   - `include_tags`: keep records matching at least one tag.
3. Check that feature files exist.
4. Shuffle deterministically with `config.SEED`.
5. Split into test, validation, and train by `TEST_RATIO` and `VAL_RATIO`.
6. Write `test.txt`, `val.txt`, and `train.txt`.

Output format:

```text
data/oeis/splits/std/
├── test.txt
├── val.txt
└── train.txt
```

---

## 5. Directory Layout

```text
data/oeis/
├── data.jsonl
├── features/
│   ├── A000001.pt
│   ├── A000002.pt
│   └── ...
└── splits/
    ├── std/
    │   ├── train.txt
    │   ├── val.txt
    │   └── test.txt
    ├── easy/
    └── all/
```

---

## 6. Error Handling

### `OEISDataset.__getitem__`

| Situation | Exception |
|-----------|-----------|
| Missing `.pt` file | `FileNotFoundError` |
| Missing required key | `ValueError` |
| File loading failure | Original exception is re-raised |

### `load_dataset`

| Situation | Exception |
|-----------|-----------|
| Missing split file | `FileNotFoundError` |

### `create_splits`

| Situation | Exception |
|-----------|-----------|
| Missing JSONL | `FileNotFoundError` |
| No IDs after filtering | `ValueError` |
| Missing feature directory | `FileNotFoundError` |
| No matching `.pt` files | `ValueError` |

---

## 7. Design Decisions

| Decision | Rationale |
|----------|-----------|
| Static split files | Eliminates run-to-run shuffle differences |
| Text split format | Readable, debuggable, and Git-manageable |
| `data_root` parameter | Enables dependency injection in tests |
| One file per ID | Supports parallel preprocessing and memory-efficient lazy loading |
| Use `schemas.OEISRecord` | Ensures strict tag parsing |

---

## 8. Notes

`load_dataset` does not shuffle. Use `DataLoader(shuffle=True)` when shuffling is needed:

```python
from torch.utils.data import DataLoader

train_dataset = load_dataset("std", "train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

For very large datasets, `create_splits` performs O(n) filesystem checks. If the number of feature files grows into the millions, preloading the file list with `glob()` may be worth considering.
