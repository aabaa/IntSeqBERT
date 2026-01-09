# `src/intseq_bert/schemas.py` 実装仕様書

## 1. 概要

本モジュールは、OEIS レコードの**唯一の正規データ構造**を定義する。
パイプライン全体（前処理、特徴抽出、データローダー）で共通のスキーマを強制し、データ不整合を防止する。

### 設計原則

- **Single Source of Truth**: データ構造の定義はこのモジュールのみ
- **Strict Validation**: 不正なデータは即座にエラーを発生させる（Fail Fast）
- **JSONL 互換**: JSON Lines 形式でのシリアライズ/デシリアライズをサポート

---

## 2. 依存関係

```python
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from pathlib import Path
```

外部ライブラリへの依存なし（標準ライブラリのみ）。

---

## 3. データ構造

### 3.1. `OEISRecord` (Dataclass)

OEIS の単一数列を表す構造体。

#### フィールド定義

| フィールド | 型 | デフォルト | 必須 | 説明 |
|-----------|-----|-----------|------|------|
| `oeis_id` | `str` | - | ✅ | OEIS ID (例: `"A000045"`) |
| `sequence` | `List[int]` | - | ✅ | 整数数列 |
| `name` | `str` | `""` | | 数列の名称 |
| `offset_a` | `int` | `0` | | 開始オフセット (OEIS %O フィールド) |
| `keywords` | `List[str]` | `[]` | | OEIS キーワード (例: `["core", "easy"]`) |
| `related` | `List[str]` | `[]` | | 関連数列 ID リスト |
| `metadata` | `Dict[str, Any]` | `{}` | | 拡張用メタデータ |

#### JSON 表現例

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

## 4. メソッド詳細

### 4.1. インスタンスメソッド

#### `to_dict() -> Dict[str, Any]`

レコードを辞書に変換。

```python
record.to_dict()
# => {"oeis_id": "A000045", "sequence": [0, 1, 1, ...], ...}
```

#### `to_json_line() -> str`

レコードを JSON 文字列（1行）にシリアライズ。

```python
record.to_json_line()
# => '{"oeis_id": "A000045", "sequence": [0, 1, 1, ...], ...}'
```

- `ensure_ascii=False` で UTF-8 文字をそのまま出力

#### `__str__() -> str`

デバッグ用の短縮表現。

```python
str(record)
# => "[A000045] Fibonacci numbers (Offset:0) Seq:[0, 1, 1, 2, 3]..."
```

---

### 4.2. クラスメソッド

#### `from_dict(data: Dict[str, Any]) -> OEISRecord`

辞書からインスタンスを生成。**厳格なバリデーション**を実行。

**バリデーションルール:**

| チェック | 条件 | 例外 |
|---------|------|------|
| `oeis_id` 存在 | `"oeis_id"` キー必須 | `ValueError` |
| `sequence` 存在 | `"sequence"` キー必須 | `ValueError` |
| `sequence` 型 | `list` または `str` | `TypeError` |

**sequence の型変換:**

| 入力型 | 処理 |
|--------|------|
| `list` | そのまま使用 |
| `str` | カンマ区切りでパース (`"1, 2, 3"` → `[1, 2, 3]`) |
| その他 | `TypeError` |

**Legacy 非サポート:**
- `"id"` キー（旧形式）は**サポートしない**
- `"oeis_id"` のみを受け付ける

#### `from_json_line(line: str) -> OEISRecord`

JSON 文字列（1行）からインスタンスを生成。

```python
OEISRecord.from_json_line('{"oeis_id": "A000045", "sequence": [0, 1, 1]}')
```

---

## 5. I/O ヘルパー関数

### `save_records(records: List[OEISRecord], filepath: str)`

レコードリストを JSONL ファイルに保存。

```python
save_records([record1, record2], "data.jsonl")
```

**出力形式:** 1行1レコードの JSON Lines

### `load_records(filepath: str) -> List[OEISRecord]`

JSONL ファイルからレコードリストを読み込み。

```python
records = load_records("data.jsonl")
```

**動作:**
- ファイルが存在しない場合: 空リスト `[]` を返す
- 不正な行が含まれる場合: 例外を発生させる（Strict Mode）

---

## 6. エラーハンドリング

| 状況 | 例外 | メッセージ例 |
|------|------|-------------|
| `oeis_id` 欠落 | `ValueError` | `"Missing required key: 'oeis_id'"` |
| `sequence` 欠落 | `ValueError` | `"Missing required key: 'sequence' for ID A000045"` |
| `sequence` 型不正 | `TypeError` | `"Invalid type for 'sequence' in ID A000045: <class 'int'>"` |
| `sequence` 文字列パース失敗 | `ValueError` | `"Malformed sequence string for ID A000045: ..."` |

---

## 7. 使用例

### 基本的な使用

```python
from intseq_bert.schemas import OEISRecord, save_records, load_records

# 作成
record = OEISRecord(
    oeis_id="A000045",
    sequence=[0, 1, 1, 2, 3, 5, 8],
    name="Fibonacci numbers",
    keywords=["core", "easy"]
)

# シリアライズ
json_line = record.to_json_line()

# デシリアライズ
restored = OEISRecord.from_json_line(json_line)

# ファイル I/O
save_records([record], "output.jsonl")
records = load_records("output.jsonl")
```

### パイプラインでの使用

```python
# preprocess.py 内
from intseq_bert.schemas import OEISRecord

with open("data.jsonl") as f:
    for line in f:
        record = OEISRecord.from_json_line(line)
        # record.sequence, record.keywords などにアクセス
```

---

## 8. 設計上の決定事項

| 決定 | 理由 |
|------|------|
| `dataclass` 使用 | ボイラープレート削減、`asdict()` による簡易シリアライズ |
| Legacy `id` キー非サポート | データ移行完了後の明確な境界線 |
| `sequence` 文字列パース対応 | 外部データソース（CSV等）との互換性 |
| `load_records` での Strict Mode | 不正データの早期検出 |
