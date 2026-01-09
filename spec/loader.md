# `src/intseq_bert/loader.py` 実装仕様書

## 1. 概要

本モジュールは、OEIS 特徴量ファイル (`.pt`) のローディングと**データ分割管理**を担当する。

### 設計原則

| 原則 | 説明 |
|------|------|
| **責任分離** | Split 生成 (Admin) と Dataset 読み込み (User) を厳密に分離 |
| **Physical Isolation** | 分割結果は静的テキストファイルとして保存 |
| **No Runtime Shuffle** | `load_dataset` は事前生成された分割ファイルを読み込むのみ |
| **Deterministic Splitting** | `config.SEED` による再現可能な分割 |
| **Fail Fast** | 欠損ファイル・キーは即座にエラー |

---

## 2. 依存関係

```python
import torch
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional
from torch.utils.data import Dataset
from . import config
from . import schemas
```

### 使用する config 定数

| 定数 | 例 | 用途 |
|------|---|------|
| `DATA_ROOT` | `"data/oeis"` | データルートパス |
| `SPLIT_DIR_NAME` | `"splits"` | 分割ファイルディレクトリ名 |
| `FEATURES_DIR_NAME` | `"features"` | 特徴量ディレクトリ名 |
| `SEED` | `42` | 乱数シード |
| `VAL_RATIO` | `0.05` | 検証データ比率 |
| `TEST_RATIO` | `0.05` | テストデータ比率 |
| `KEY_MAG_FEATURES` | `"mag_features"` | 必須キー |
| `KEY_MOD_FEATURES` | `"mod_features"` | 必須キー |
| `KEY_OEIS_ID` | `"oeis_id"` | ID キー |

---

## 3. クラス設計

### 3.1. `OEISDataset` (torch.utils.data.Dataset)

個別の `.pt` ファイルからオンデマンドでデータを読み込む Dataset。

#### `__init__` 引数

| 引数 | 型 | 説明 |
|------|------|------|
| `oeis_ids` | `List[str]` | ロード対象の OEIS ID リスト |
| `features_dir` | `Path` | `.pt` ファイルが格納されたディレクトリ |

#### `__getitem__` 出力

```python
{
    "mag_features": Tensor(L, MAG_RAW_DIM),   # (L, 4)
    "mod_features": Tensor(L, MOD_FEATURE_DIM), # (L, 200)
    "mod_integers": Tensor(L, NUM_MODULI),    # (L, 100)
    "oeis_id": str
}
```

#### バリデーション

1. **ファイル存在チェック**: `FileNotFoundError` を発生
2. **必須キーチェック**: `KEY_MAG_FEATURES`, `KEY_MOD_FEATURES` が必須
3. **ID 注入**: `KEY_OEIS_ID` をデータに追加

#### 使用例

```python
dataset = OEISDataset(
    oeis_ids=["A000045", "A000040"],
    features_dir=Path("data/oeis/features")
)
sample = dataset[0]  # {"mag_features": ..., ...}
```

---

## 4. 関数設計

### 4.1. `load_dataset` (Runtime: User 向け)

事前生成された分割ファイルから Dataset を読み込む。

#### シグネチャ

```python
def load_dataset(
    split_type: str,
    split_name: str,
    *,
    data_root: Optional[str] = None
) -> OEISDataset
```

#### 引数

| 引数 | 型 | 例 | 説明 |
|------|------|---|------|
| `split_type` | `str` | `"str"` | 分割タイプ（ディレクトリ名） |
| `split_name` | `str` | `"train"` | 分割名 (`train`, `val`, `test`) |
| `data_root` | `Optional[str]` | | テスト用オーバーライド |

#### 処理フロー

```
1. パス構築: {data_root}/splits/{split_type}/{split_name}.txt
2. ファイル存在確認 (なければ FileNotFoundError)
3. ID リスト読み込み (1行1ID)
4. OEISDataset インスタンス作成・返却
```

#### 使用例

```python
train_dataset = load_dataset("std", "train")
val_dataset = load_dataset("std", "val")
test_dataset = load_dataset("std", "test")
```

---

### 4.2. `create_splits` (Admin: 管理者向け)

JSONL からタグフィルタリング後、静的分割ファイルを生成する。

#### シグネチャ

```python
def create_splits(
    source_jsonl: str,
    output_split_type: str,
    include_tags: Optional[List[str]] = None,
    exclude_tags: Optional[List[str]] = None,
    *,
    data_root: Optional[str] = None
)
```

#### 引数

| 引数 | 型 | 例 | 説明 |
|------|------|---|------|
| `source_jsonl` | `str` | `"data/oeis/data.jsonl"` | ソース JSONL |
| `output_split_type` | `str` | `"std"` | 出力ディレクトリ名 |
| `include_tags` | `Optional[List[str]]` | `["core", "easy"]` | 含むタグ (OR) |
| `exclude_tags` | `Optional[List[str]]` | `["cons", "base"]` | 除外タグ (OR) |
| `data_root` | `Optional[str]` | | テスト用オーバーライド |

#### 処理フロー

```
1. JSONL 読み込み (schemas.OEISRecord で厳密パース)
2. タグフィルタリング
   - exclude_tags: いずれかに一致 → 除外
   - include_tags: いずれにも一致しない → 除外
3. 特徴量ファイル存在確認 (.pt ファイル)
4. 決定論的シャッフル (config.SEED)
5. 分割 (TEST_RATIO → VAL_RATIO → 残りは TRAIN)
6. ファイル書き出し (.txt)
   - test.txt
   - val.txt
   - train.txt
```

#### 出力ファイル形式

```
data/oeis/splits/std/
├── test.txt    # 1行1ID (例: "A000045\n")
├── val.txt
└── train.txt
```

#### 使用例

```python
create_splits(
    source_jsonl="data/oeis/data.jsonl",
    output_split_type="std",
    exclude_tags=["cons", "base", "word"]
)
```

---

## 5. ディレクトリ構造

```
data/oeis/
├── data.jsonl              # ソース JSONL (preprocess で生成)
├── features/               # .pt ファイル群
│   ├── A000001.pt
│   ├── A000002.pt
│   └── ...
└── splits/                 # 分割ファイル群
    ├── std/
    │   ├── train.txt
    │   ├── val.txt
    │   └── test.txt
    ├── easy/
    │   └── ...
    └── all/
        └── ...
```

---

## 6. エラーハンドリング

### `OEISDataset.__getitem__`

| 状況 | 例外 |
|------|------|
| `.pt` ファイル不存在 | `FileNotFoundError` |
| 必須キー欠落 | `ValueError` |
| ファイル読み込み失敗 | 元の例外を再送出 |

### `load_dataset`

| 状況 | 例外 |
|------|------|
| 分割ファイル不存在 | `FileNotFoundError` |

### `create_splits`

| 状況 | 例外 |
|------|------|
| JSONL 不存在 | `FileNotFoundError` |
| フィルタ後 ID なし | `ValueError` |
| features ディレクトリ不存在 | `FileNotFoundError` |
| 対応 .pt ファイルなし | `ValueError` |

---

## 7. 設計上の決定事項

| 決定 | 理由 |
|------|------|
| 静的分割ファイル | 実行ごとのシャッフルを排除、再現性確保 |
| テキストファイル形式 | 可読性、デバッグ容易性、Git 管理可能 |
| `data_root` パラメータ | 依存性注入でテスト容易性向上 |
| ID 単位でファイル分離 | 並列処理対応、メモリ効率的な Lazy Loading |
| `schemas.OEISRecord` 使用 | タグ情報の厳密なパース |

---

## 8. 注意事項

### Runtime Shuffle について

`load_dataset` は**シャッフルを行わない**。
シャッフルが必要な場合は `DataLoader(shuffle=True)` を使用する。

```python
from torch.utils.data import DataLoader

train_dataset = load_dataset("std", "train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### 大規模データでのパフォーマンス

`create_splits` のファイル存在チェックは O(n) のファイルシステムアクセスを行う。
数百万ファイルの場合、`glob()` でファイル一覧を先に取得する最適化を検討可能。
