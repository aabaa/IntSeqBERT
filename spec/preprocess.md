# `src/intseq_bert/preprocess.py` 実装仕様書

## 1. 概要

本モジュールは、OEIS データパイプラインの**CLIエントリポイント**として機能する。
生データのパース、構造化、特徴量抽出、データセット分割を担当する。

### 3層アーキテクチャ

| Layer | 責務 | 特徴 |
|-------|------|------|
| **Layer 1** | Pure Logic Functions | 副作用なし、単体テスト可能 |
| **Layer 2** | Worker & Helper | ファイルI/O、並列処理 |
| **Layer 3** | Command Handlers | CLIフロー制御 |

---

## 2. 依存関係

```python
import argparse
import gzip
import logging
import multiprocessing
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from functools import partial

import torch
from tqdm import tqdm

from . import config
from . import schemas
from . import features
```

### 使用する config 定数

| 定数 | 用途 |
|------|------|
| `MIN_SEQUENCE_LENGTH` | 最小シーケンス長フィルタ |
| `SEED` | 決定論的シャッフル |
| `TEST_RATIO`, `VAL_RATIO` | 分割比率 |
| `KEY_OEIS_ID` | 出力キー |

---

## 3. CLI コマンド

### サブコマンド一覧

| コマンド | 説明 |
|---------|------|
| `build-jsonl` | 生データから JSONL を構築 |
| `extract-features` | JSONL から .pt ファイルを生成 |
| `split-dataset` | タグフィルタリング付き分割 |

---

## 4. Layer 1: Pure Logic Functions

### 4.1. `_parse_stripped_line`

`stripped.gz` の1行をパース。

#### シグネチャ

```python
def _parse_stripped_line(line: str) -> Optional[Tuple[str, List[int]]]
```

#### 入力フォーマット

```
A000045 ,0,1,1,2,3,5,8,13,
```

- ID と数列の間に ` ,` (スペース+カンマ) がある
- 末尾のカンマは許容

#### 戻り値

| 条件 | 戻り値 |
|------|--------|
| 成功 | `(oeis_id, sequence)` |
| 空行、コメント、不正形式 | `None` |
| 非整数値が含まれる | `None` |

---

### 4.2. `_parse_names_line`

`names.gz` の1行をパース。

#### シグネチャ

```python
def _parse_names_line(line: str) -> Optional[Tuple[str, str]]
```

#### 入力フォーマット

```
A000045 Fibonacci numbers
```

- 最初のスペースで ID と名前を分割

#### 戻り値

| 条件 | 戻り値 |
|------|--------|
| 成功 | `(oeis_id, name)` |
| 空行、`#` で始まる行 | `None` |
| `A` で始まらない | `None` |

---

### 4.3. `_parse_seq_content`

`.seq` ファイル（Internal Format）からメタデータを抽出。

#### シグネチャ

```python
def _parse_seq_content(lines: List[str]) -> Dict[str, Any]
```

#### 対応タグ

| タグ | フォーマット | 抽出内容 |
|------|-------------|---------|
| `%K` | `%K A000045 nonn,easy` | `keywords` リスト |
| `%O` | `%O A000045 0,2` | `offset_a` 整数 |

#### 戻り値

```python
{
    "keywords": ["nonn", "easy"],
    "offset_a": 0
}
```

---

## 5. Layer 2: Worker & Helper Functions

### 5.1. `_load_names_map`

`names.gz` を全走査し、ID→名前マップを作成。

```python
def _load_names_map(names_path: Path) -> Dict[str, str]
```

### 5.2. `_scan_seq_files`

ディレクトリを再帰走査し、ID→.seq パスマップを作成。

```python
def _scan_seq_files(seq_dir: Path) -> Dict[str, Path]
```

### 5.3. `_worker_extract_features`

`multiprocessing` 用ワーカー。JSONL 行のチャンクを処理。

```python
def _worker_extract_features(chunk: List[str], output_dir: Path) -> int
```

#### 処理フロー

```
1. JSONL 行をデシリアライズ (schemas.OEISRecord)
2. MIN_SEQUENCE_LENGTH でフィルタ
3. features.process_sequence() で特徴量抽出
4. KEY_OEIS_ID を追加
5. 生の整数列を "numbers" キーとして追加 (Vanilla Transformer 用)
6. .pt ファイルとして保存
7. 処理件数を返却
```

---

## 6. Layer 3: Command Handlers

### 6.1. `cmd_build_jsonl`

**コマンド:** `build-jsonl`

#### CLI 引数

| 引数 | 短縮 | 必須 | 説明 |
|------|------|------|------|
| `--stripped` | | ✅ | stripped.gz パス |
| `--names` | | | names.gz パス |
| `--seq-dir` | | | .seq ディレクトリ |
| `--output` | `-o` | ✅ | 出力 JSONL パス |

#### 処理フロー

```
1. names.gz をメモリにロード (任意)
2. .seq ファイルをスキャン (任意)
3. stripped.gz をストリーム処理:
   - _parse_stripped_line でパース
   - 名前をマージ
   - メタデータをマージ (オンデマンド読み込み)
   - OEISRecord を生成
   - JSONL に書き出し
```

#### 使用例

```bash
uv run python -m intseq_bert.preprocess build-jsonl \
  --stripped data/oeis/raw/stripped.gz \
  --names data/oeis/raw/names.gz \
  --seq-dir data/oeis/seq \
  -o data/oeis/data.jsonl
```

---

### 6.2. `cmd_extract_features`

**コマンド:** `extract-features`

#### CLI 引数

| 引数 | 短縮 | 必須 | デフォルト | 説明 |
|------|------|------|-----------|------|
| `--input` | `-i` | ✅ | | 入力 JSONL |
| `--output-dir` | `-o` | ✅ | | 出力ディレクトリ |
| `--workers` | | | `4` | ワーカー数 |
| `--chunk-size` | | | `1000` | チャンクサイズ |

#### 処理フロー

```
1. JSONL を読み込みチャンクに分割
2. multiprocessing.Pool を作成
3. _worker_extract_features を並列実行
4. 処理件数を集計
```

#### 使用例

```bash
uv run python -m intseq_bert.preprocess extract-features \
  -i data/oeis/data.jsonl \
  -o data/oeis/features \
  --workers 8
```

---

### 6.3. `cmd_split_dataset`

**コマンド:** `split-dataset`

#### CLI 引数

| 引数 | 短縮 | 必須 | 説明 |
|------|------|------|------|
| `--jsonl` | `-j` | ✅ | タグ情報のソース JSONL |
| `--features-dir` | `-f` | ✅ | .pt ファイルディレクトリ |
| `--output-dir` | `-o` | ✅ | 分割リスト出力先 |
| `--include-tags` | | | 含むタグ (カンマ区切り、OR) |
| `--exclude-tags` | | | 除外タグ (カンマ区切り、OR) |

#### 処理フロー

```
1. JSONL を読み込み、タグでフィルタリング
2. .pt ファイル存在を確認
3. config.SEED でシャッフル
4. TEST_RATIO, VAL_RATIO で分割
5. train.txt, val.txt, test.txt を出力
```

#### 使用例

```bash
uv run python -m intseq_bert.preprocess split-dataset \
  -j data/oeis/data.jsonl \
  -f data/oeis/features \
  -o data/oeis/splits/std \
  --exclude-tags cons,base,word,fini
```

---

## 7. 出力ファイル形式

### data.jsonl

```json
{"oeis_id": "A000045", "sequence": [0, 1, 1, ...], "name": "Fibonacci", "keywords": ["core"], ...}
{"oeis_id": "A000040", "sequence": [2, 3, 5, ...], ...}
```

### features/{ID}.pt

```python
{
    "mag_features": Tensor(L, 4),
    "mod_features": Tensor(L, 200),
    "mod_integers": Tensor(L, 100),
    "oeis_id": "A000045",
    "numbers": [0, 1, 1, 2, 3, 5, ...]  # 生の整数列 (Vanilla Transformer 用)
}
```

### splits/{type}/train.txt

```
A000045
A000040
A000079
...
```

---

## 8. エラーハンドリング

| 状況 | 処理 |
|------|------|
| パース失敗 | `None` を返し、スキップ |
| 特徴量抽出失敗 | DEBUG ログ、スキップ |
| タグフィルタ後 ID なし | 警告ログ、処理終了 |
| .pt ファイルなし | 警告ログ、処理終了 |

---

## 9. ログレベル

| レベル | 用途 |
|--------|------|
| `INFO` | 進捗、サマリー |
| `DEBUG` | 個別レコードのエラー |

`tqdm` は親プロセスで管理し、ワーカー内での表示崩れを防止。

---

## 10. 設計上の決定事項

| 決定 | 理由 |
|------|------|
| 3層アーキテクチャ | テスト容易性、責任分離 |
| Pure Functions でパース | 副作用なし、単体テスト可能 |
| multiprocessing | CPU バウンドな特徴量計算の高速化 |
| チャンク処理 | メモリ効率、進捗表示の粒度制御 |
| タグフィルタを CLI に統合 | ワンライナーでデータセット構築可能 |
