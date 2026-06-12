# `src/intseq_bert/preprocess.py` Implementation Specification

## 1. Overview

This module is the CLI entry point for the OEIS data pipeline. It parses raw data, builds structured records, extracts features, and creates dataset splits.

### Three-Layer Architecture

| Layer | Responsibility | Characteristics |
|-------|----------------|-----------------|
| **Layer 1** | Pure functions | Side-effect-free and unit-testable |
| **Layer 2** | Workers and helpers | File I/O and parallel processing |
| **Layer 3** | Command handlers | CLI flow control |

---

## 2. Dependencies

```python
import argparse
import gzip
import logging
import multiprocessing
import os
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from . import config, features, schemas
```

### Config Constants

| Constant | Purpose |
|----------|---------|
| `MIN_SEQUENCE_LENGTH` | Minimum sequence length |
| `SEED` | Deterministic shuffle |
| `TEST_RATIO`, `VAL_RATIO` | Split ratios |
| `KEY_OEIS_ID` | Output key |

---

## 3. CLI Commands

| Command | Description |
|---------|-------------|
| `build-jsonl` | Build JSONL from raw OEIS data |
| `extract-features` | Generate `.pt` feature files from JSONL |
| `split-dataset` | Create split files with optional tag filtering |

---

## 4. Layer 1: Pure Logic Functions

### 4.1 `_parse_stripped_line`

Parses one line from `stripped.gz`.

```python
def _parse_stripped_line(line: str) -> Optional[Tuple[str, List[int]]]
```

Input format:

```text
A000045 ,0,1,1,2,3,5,8,13,
```

The ID and sequence are separated by `" ,"`, and a trailing comma is allowed.

| Condition | Return value |
|-----------|--------------|
| Success | `(oeis_id, sequence)` |
| Empty/comment/malformed line | `None` |
| Non-integer value | `None` |

### 4.2 `_parse_names_line`

Parses one line from `names.gz`.

```python
def _parse_names_line(line: str) -> Optional[Tuple[str, str]]
```

Input format:

```text
A000045 Fibonacci numbers
```

The first space separates the ID from the name.

### 4.3 `_parse_seq_content`

Extracts metadata from `.seq` files in OEIS internal format.

```python
def _parse_seq_content(lines: List[str]) -> Dict[str, Any]
```

| Tag | Format | Extracted value |
|-----|--------|-----------------|
| `%K` | `%K A000045 nonn,easy` | `keywords` list |
| `%O` | `%O A000045 0,2` | integer `offset_a` |

Return value:

```python
{
    "keywords": ["nonn", "easy"],
    "offset_a": 0,
}
```

---

## 5. Layer 2: Workers and Helpers

### 5.1 `_load_names_map`

Scans `names.gz` and builds an ID-to-name map.

```python
def _load_names_map(names_path: Path) -> Dict[str, str]
```

### 5.2 `_scan_seq_files`

Recursively scans a directory and builds an ID-to-`.seq` path map.

```python
def _scan_seq_files(seq_dir: Path) -> Dict[str, Path]
```

### 5.3 `_worker_extract_features`

Multiprocessing worker that processes a chunk of JSONL lines.

```python
def _worker_extract_features(chunk: List[str], output_dir: Path) -> int
```

Processing flow:

1. Deserialize JSONL lines as `schemas.OEISRecord`.
2. Filter by `MIN_SEQUENCE_LENGTH`.
3. Extract features with `features.process_sequence()`.
4. Add `KEY_OEIS_ID`.
5. Add the raw integer sequence as `"numbers"` for the Vanilla Transformer.
6. Save the result as a `.pt` file.
7. Return the number of processed records.

---

## 6. Layer 3: Command Handlers

### 6.1 `cmd_build_jsonl`

Command: `build-jsonl`

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--stripped` | | yes | Path to `stripped.gz` |
| `--names` | | no | Path to `names.gz` |
| `--seq-dir` | | no | Directory containing `.seq` files |
| `--output` | `-o` | yes | Output JSONL path |

Processing flow:

1. Load `names.gz` into memory if provided.
2. Scan `.seq` files if provided.
3. Stream `stripped.gz`:
   - parse with `_parse_stripped_line`,
   - merge names,
   - merge metadata on demand,
   - create `OEISRecord`,
   - write JSONL.

Example:

```bash
uv run python -m intseq_bert.preprocess build-jsonl \
  --stripped data/oeis/raw/stripped.gz \
  --names data/oeis/raw/names.gz \
  --seq-dir data/oeis/seq \
  -o data/oeis/data.jsonl
```

### 6.2 `cmd_extract_features`

Command: `extract-features`

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--input` | `-i` | yes | | Input JSONL |
| `--output-dir` | `-o` | yes | | Output directory |
| `--workers` | | no | `4` | Number of workers |
| `--chunk-size` | | no | `1000` | Chunk size |

Processing flow:

1. Read JSONL and split it into chunks.
2. Create `multiprocessing.Pool`.
3. Run `_worker_extract_features` in parallel.
4. Aggregate processed counts.

Example:

```bash
uv run python -m intseq_bert.preprocess extract-features \
  -i data/oeis/data.jsonl \
  -o data/oeis/features \
  --workers 8
```

### 6.3 `cmd_split_dataset`

Command: `split-dataset`

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--jsonl` | `-j` | yes | JSONL source for tag information |
| `--features-dir` | `-f` | yes | Directory containing `.pt` files |
| `--output-dir` | `-o` | yes | Output directory for split lists |
| `--include-tags` | | no | Comma-separated include tags (OR semantics) |
| `--exclude-tags` | | no | Comma-separated exclude tags (OR semantics) |

Processing flow:

1. Load JSONL and filter by tags.
2. Check that corresponding `.pt` files exist.
3. Shuffle with `config.SEED`.
4. Split by `TEST_RATIO` and `VAL_RATIO`.
5. Write `train.txt`, `val.txt`, and `test.txt`.

Example:

```bash
uv run python -m intseq_bert.preprocess split-dataset \
  -j data/oeis/data.jsonl \
  -f data/oeis/features \
  -o data/oeis/splits/std \
  --exclude-tags cons,base,word,fini
```

---

## 7. Output File Formats

### `data.jsonl`

```json
{"oeis_id": "A000045", "sequence": [0, 1, 1], "name": "Fibonacci", "keywords": ["core"]}
{"oeis_id": "A000040", "sequence": [2, 3, 5]}
```

### `features/{ID}.pt`

```python
{
    "mag_features": Tensor(L, 4),
    "mod_features": Tensor(L, 200),
    "mod_integers": Tensor(L, 100),
    "oeis_id": "A000045",
    "numbers": [0, 1, 1, 2, 3, 5],
}
```

### `splits/{type}/train.txt`

```text
A000045
A000040
A000079
```

---

## 8. Error Handling

| Situation | Handling |
|-----------|----------|
| Parse failure | Return `None` and skip |
| Feature extraction failure | Log at DEBUG level and skip |
| No IDs after tag filtering | Log warning and exit |
| Missing `.pt` file | Log warning and exit |

---

## 9. Logging

| Level | Use |
|-------|-----|
| `INFO` | Progress and summaries |
| `DEBUG` | Per-record errors |

`tqdm` is managed in the parent process to avoid garbled worker output.

---

## 10. Design Decisions

| Decision | Rationale |
|----------|-----------|
| Three-layer architecture | Improves testability and separates responsibilities |
| Pure parsing functions | Avoid side effects and simplify unit tests |
| Multiprocessing | Speeds up CPU-bound feature extraction |
| Chunk processing | Improves memory efficiency and progress granularity |
| CLI tag filtering | Allows one-line dataset construction |
