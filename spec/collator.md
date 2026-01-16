# `src/intseq_bert/collator.py` 実装仕様書

## 1. 概要

本モジュールは、Dual Stream Architecture における**動的マスキング**と**バッチ構築**を担当する。
`OEISDataset` から読み込んだ可変長サンプルをパディングし、Masked Language Modeling のためのマスク処理を適用する。

### 設計原則

- **Dynamic Masking**: エポックごとに異なるマスクパターンを生成（データ拡張効果）
- **Mask Flag Strategy**: 連続値ストリームで「0」と「マスク」を区別
- **Origin Shift Strategy**: Sin/Cos ストリームで原点 (0, 0) をマスク表現に使用

---

## 2. 依存関係

```python
import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import List, Dict, Any
from . import config
```

### 使用する config 定数

| 定数 | 値 | 用途 |
|------|------|------|
| `MASK_PROB` | 0.15 | マスク確率 |
| `PAD_VALUE_FEATURE` | -9999.0 | 特徴量のパディング値 (Sentinel Value) |
| `IGNORE_INDEX` | -100 | 損失計算で無視するラベル値 |
| `MAG_RAW_DIM` | 4 | 入力 Magnitude 次元 |
| `MAG_EXTENDED_DIM` | 5 | マスクフラグ付き Magnitude 次元 |
| `MOD_FEATURE_DIM` | 200 | Modulo Sin/Cos 次元 |
| `NUM_MODULI` | 100 | 法の数 |
| `KEY_MAG_FEATURES` | `"mag_features"` | データキー |
| `KEY_MOD_FEATURES` | `"mod_features"` | データキー |
| `KEY_MOD_INTEGERS` | `"mod_integers"` | データキー |
| `KEY_OEIS_ID` | `"oeis_id"` | データキー |

---

## 3. クラス設計

### 3.1. `OEISCollator` (Dataclass)

PyTorch DataLoader の `collate_fn` として使用するコラータ。

#### フィールド

| フィールド | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `mask_prob` | `float` | `config.MASK_PROB` | マスク確率 |

#### 入力契約

`OEISDataset.__getitem__` が返す辞書のリスト:

```python
[
    {
        "mag_features": Tensor(L1, 4),   # MAG_RAW_DIM
        "mod_features": Tensor(L1, 200), # MOD_FEATURE_DIM
        "mod_integers": Tensor(L1, 100), # NUM_MODULI
        "oeis_id": "A000045"
    },
    {
        "mag_features": Tensor(L2, 4),
        ...
    },
    ...
]
```

#### 出力契約

```python
{
    "mag_inputs":     Tensor(B, L, 5),    # MAG_EXTENDED_DIM
    "mod_inputs":     Tensor(B, L, 200),  # MOD_FEATURE_DIM
    "mag_labels":     Tensor(B, L, 4),    # MAG_RAW_DIM
    "mod_labels":     Tensor(B, L, 100),  # NUM_MODULI, Long
    "attention_mask": Tensor(B, L),       # Long, 1=valid, 0=padding
    "mask_matrix":    Tensor(B, L),       # Bool, True=masked
    "oeis_ids":       List[str]           # バッチ内の ID リスト
}
```

---

## 4. 処理フロー

### Step 1: 入力検証

```python
if not batch:
    raise ValueError("Batch is empty.")

required_keys = [KEY_MAG_FEATURES, KEY_MOD_FEATURES, KEY_MOD_INTEGERS]
for key in required_keys:
    if key not in batch[0]:
        raise KeyError(f"Dataset must provide '{key}' for collator.")
```

### Step 2: パディング

```python
# 特徴量は Sentinel Value でパディング
mag_padded = pad_sequence(mag_list, batch_first=True, padding_value=config.PAD_VALUE_FEATURE)
mod_padded = pad_sequence(mod_list, batch_first=True, padding_value=config.PAD_VALUE_FEATURE)

# 整数ラベルは IGNORE_INDEX でパディング
mod_int_padded = pad_sequence(mod_int_list, batch_first=True, padding_value=-100)
```

### Step 3: Attention Mask 生成

```python
# Sentinel Value をチェックして有効位置を判定
# Magの第1チャネル (log_val) を確認
valid_mask_bool = (mag_padded[..., 0] != config.PAD_VALUE_FEATURE)
attention_mask = valid_mask_bool.long()  # 1=valid, 0=padding
```

### Step 4: マスク行列生成

```python
prob_matrix = torch.full((B, L), mask_prob)
prob_matrix[~valid_mask_bool] = 0.0  # パディング位置はマスクしない
mask_matrix = torch.bernoulli(prob_matrix).bool()
```

### Step 5: Magnitude Stream 処理

**Mask Flag Strategy:**

```
```
Unmasked: [log_val, sign+, sign-, sign0, 0]
Masked:   [0,       0,     0,     0,     1]
Padding:  [-9999.0, -9999.0, ...,        0]  (※ is_masked フラグは 0)
```

```python
# 1. マスクフラグチャネル作成
is_masked_channel = torch.zeros((B, L, 1))
is_masked_channel[mask_matrix] = 1.0

# 2. 連結して 5 次元に拡張
mag_inputs = torch.cat([mag_padded, is_masked_channel], dim=2)

# 3. マスク位置のコンテンツ (0:4) をゼロ化
content_keep_mask = (~mask_matrix).unsqueeze(-1).float()
mag_inputs[..., :4] *= content_keep_mask
```

### Step 6: Modulo Stream 処理

**Origin Shift Strategy:**

```
Unmasked: [sin(θ), cos(θ), ...]  # 単位円上
Masked:   [0,      0,      ...]  # 原点（単位円外）
```

```python
mod_inputs = mod_padded * content_keep_mask
```

### Step 7: ラベル準備

```python
# Magnitude: 元の値を保持（損失計算で mask_matrix を使用）
mag_labels = mag_padded.clone()

# Modulo: 非マスク位置を IGNORE_INDEX に設定
mod_labels = mod_int_padded.clone()
mod_labels[~mask_matrix] = IGNORE_INDEX
```

---

## 5. マスキング戦略の詳細

### 5.1. なぜ Mask Flag が必要か

**問題:** Magnitude Stream で値 `0` が有効なデータ（例: x=0 のとき log_val=0）として存在する。
単純なゼロパディングでは「0という値」と「パディング」が区別できない。また Sentinel Value を導入しても、
「マスクされたトークン」をどう表現するかという問題は残る。

**解決:** 5番目のチャネル `is_masked` を追加:
- `is_masked=0`: この位置は有効なデータ
- `is_masked=1`: この位置はマスクされている

### 5.2. なぜ Origin Shift が有効か

**問題:** Sin/Cos 埋め込みは単位円上の点を表す。ゼロベクトルは原点であり、単位円上には存在しない。

**解決:** マスク位置を原点 `(0, 0)` に設定することで、有効な Sin/Cos 値（常に $\sin^2 + \cos^2 = 1$）と区別可能。

---

## 6. エラーハンドリング

| 状況 | 例外 | メッセージ |
|------|------|-----------|
| 空バッチ | `ValueError` | `"Batch is empty."` |
| キー欠落 | `KeyError` | `"Dataset must provide 'mag_features' for collator."` |

---

## 7. 使用例

```python
from torch.utils.data import DataLoader
from intseq_bert.loader import load_dataset
from intseq_bert.collator import OEISCollator

dataset = load_dataset("strict", "train")
collator = OEISCollator(mask_prob=0.15)

dataloader = DataLoader(dataset, batch_size=32, collate_fn=collator)

for batch in dataloader:
    mag_inputs = batch["mag_inputs"]      # (32, L, 5)
    mod_inputs = batch["mod_inputs"]      # (32, L, 200)
    mask_matrix = batch["mask_matrix"]    # (32, L)
    attention_mask = batch["attention_mask"]  # (32, L)
    
    # Model forward
    outputs = model(mag_inputs, mod_inputs, 
                    src_key_padding_mask=(attention_mask == 0))
```

---

## 8. 設計上の決定事項

| 決定 | 理由 |
|------|------|
| `dataclass` 使用 | シンプルな状態管理、`mask_prob` のみ保持 |
| Dynamic Masking | Static Masking より汎化性能が向上 (RoBERTa 論文) |
| パディング位置は非マスク | パディング位置の予測は無意味 |
| `mod_labels` で非マスク位置を `IGNORE_INDEX` | CrossEntropy で自動的に損失計算から除外 |
| `mag_labels` は全位置保持 | Regression 損失は `mask_matrix` でフィルタリング |
