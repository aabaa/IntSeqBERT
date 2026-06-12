# `src/intseq_bert/schemas.py` Implementation Specification

## 1. Overview

This module defines the single canonical data structure for OEIS records. The same schema is used across preprocessing, feature extraction, and data loading so that malformed or inconsistent data is rejected early.

### Design Principles

- **Single Source of Truth**: record structure is defined only in this module.
- **Strict Validation**: invalid records fail immediately.
- **JSONL Compatible**: records can be serialized to and deserialized from JSON Lines.

---

## 2. Dependencies

```python
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List
```

The module depends only on the Python standard library.

---

## 3. Data Structure

### 3.1 `OEISRecord` Dataclass

Represents one OEIS integer sequence.

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `oeis_id` | `str` | - | yes | OEIS ID, e.g. `"A000045"` |
| `sequence` | `List[int]` | - | yes | Integer sequence |
| `name` | `str` | `""` | no | Sequence name |
| `offset_a` | `int` | `0` | no | Starting offset from the OEIS `%O` field |
| `keywords` | `List[str]` | `[]` | no | OEIS keywords, e.g. `["core", "easy"]` |
| `related` | `List[str]` | `[]` | no | Related OEIS IDs |
| `metadata` | `Dict[str, Any]` | `{}` | no | Extension metadata |

Example JSON representation:

```json
{
  "oeis_id": "A000045",
  "sequence": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34],
  "name": "Fibonacci numbers",
  "offset_a": 0,
  "keywords": ["core", "nonn", "easy"],
  "related": ["A000032", "A000073"],
  "metadata": {}
}
```

---

## 4. Methods

### 4.1 Instance Methods

#### `to_dict() -> Dict[str, Any]`

Converts a record to a dictionary.

```python
record.to_dict()
# => {"oeis_id": "A000045", "sequence": [0, 1, 1, ...], ...}
```

#### `to_json_line() -> str`

Serializes a record to one JSON line.

```python
record.to_json_line()
# => '{"oeis_id": "A000045", "sequence": [0, 1, 1, ...], ...}'
```

`ensure_ascii=False` is used so UTF-8 text is preserved.

#### `__str__() -> str`

Returns a compact debug representation.

```python
str(record)
# => "[A000045] Fibonacci numbers (Offset:0) Seq:[0, 1, 1, 2, 3]..."
```

### 4.2 Class Methods

#### `from_dict(data: Dict[str, Any]) -> OEISRecord`

Constructs a record from a dictionary and performs strict validation.

| Check | Condition | Exception |
|-------|-----------|-----------|
| `oeis_id` exists | required key | `ValueError` |
| `sequence` exists | required key | `ValueError` |
| `sequence` type | must be `list` or `str` | `TypeError` |

Sequence conversion rules:

| Input type | Behavior |
|------------|----------|
| `list` | Used as-is |
| `str` | Parsed as a comma-separated sequence, e.g. `"1, 2, 3"` -> `[1, 2, 3]` |
| other | `TypeError` |

Legacy `"id"` keys are not supported. Only `"oeis_id"` is accepted.

#### `from_json_line(line: str) -> OEISRecord`

Constructs a record from a one-line JSON string.

```python
OEISRecord.from_json_line('{"oeis_id": "A000045", "sequence": [0, 1, 1]}')
```

---

## 5. I/O Helper Functions

### `save_records(records: List[OEISRecord], filepath: str)`

Writes records to a JSONL file, one record per line.

```python
save_records([record1, record2], "data.jsonl")
```

### `load_records(filepath: str) -> List[OEISRecord]`

Loads records from a JSONL file.

```python
records = load_records("data.jsonl")
```

Behavior:

- Missing files return an empty list.
- Invalid lines raise exceptions in strict mode.

---

## 6. Error Handling

| Situation | Exception | Example message |
|-----------|-----------|-----------------|
| Missing `oeis_id` | `ValueError` | `"Missing required key: 'oeis_id'"` |
| Missing `sequence` | `ValueError` | `"Missing required key: 'sequence' for ID A000045"` |
| Invalid `sequence` type | `TypeError` | `"Invalid type for 'sequence' in ID A000045: <class 'int'>"` |
| Malformed sequence string | `ValueError` | `"Malformed sequence string for ID A000045: ..."` |

---

## 7. Usage Examples

```python
from intseq_bert.schemas import OEISRecord, load_records, save_records

record = OEISRecord(
    oeis_id="A000045",
    sequence=[0, 1, 1, 2, 3, 5, 8],
    name="Fibonacci numbers",
    keywords=["core", "easy"],
)

json_line = record.to_json_line()
restored = OEISRecord.from_json_line(json_line)

save_records([record], "output.jsonl")
records = load_records("output.jsonl")
```

Pipeline usage:

```python
from intseq_bert.schemas import OEISRecord

with open("data.jsonl") as f:
    for line in f:
        record = OEISRecord.from_json_line(line)
        # Access record.sequence, record.keywords, etc.
```

---

## 8. Design Decisions

| Decision | Rationale |
|----------|-----------|
| Use `dataclass` | Reduces boilerplate and enables simple serialization through `asdict()` |
| Do not support legacy `id` | Keeps a clear boundary after data migration |
| Accept string sequences | Improves compatibility with external sources such as CSV files |
| Strict `load_records` | Detects malformed data early |
