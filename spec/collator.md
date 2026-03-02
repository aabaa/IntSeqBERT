# `src/intseq_bert/collator.py` 実装仕様書

## 1. 概要

本モジュールは、IntSeqBERT および Vanilla Transformer 向けの**動的マスキング**と**バッチ構築**を担当する。
`OEISDataset` から読み込んだ可変長サンプルをパディングし、Masked Language Modeling のためのマスク処理を適用する。

### 設計原則

- **Dynamic Masking**: エポックごとに異なるマスクパターンを生成（データ拡張効果）
- **Mask Flag Strategy**: 連続値ストリームで「0」と「マスク」を区別
- **Origin Shift Strategy**: Sin/Cos ストリームで原点 (0, 0) をマスク表現に使用
- **Dual Model Support**: IntSeqBERT と Vanilla Transformer の両方をサポート

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
| `VANILLA_VOCAB_SIZE` | 20003 | Vanilla トークン語彙サイズ |
| `VANILLA_PAD_TOKEN_ID` | 0 | パディングトークン ID |
| `VANILLA_MASK_TOKEN_ID` | 1 | マスクトークン ID |
| `VANILLA_UNK_TOKEN_ID` | 2 | 未知語トークン ID |

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
    # IntSeqBERT 入力
    "mag_inputs":     Tensor(B, L, 5),    # MAG_EXTENDED_DIM
    "mod_inputs":     Tensor(B, L, 200),  # MOD_FEATURE_DIM
    "mag_labels":     Tensor(B, L, 4),    # MAG_RAW_DIM
    "mod_labels":     Tensor(B, L, 100),  # NUM_MODULI, Long
    
    # Vanilla Transformer 入力
    "token_ids":      Tensor(B, L),       # Long, トークン ID
    "token_labels":   Tensor(B, L),       # Long, マスク位置のみ有効
    
    # 共通
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
Unmasked (Valid):  [log_val, sign+, sign-, sign0, 0]
Masked (Valid):    [0,       0,     0,     0,     1]
Padding:           [0,       0,     0,     0,     0]  (※ 全てゼロ化される)
```

> **Note:** パディング位置のコンテンツも明示的にゼロ化する。
> これにより、`PAD_VALUE_FEATURE = -9999.0` のようなセンチネル値がモデルに流入することを防ぐ。
```

```python
# 1. マスクフラグチャネル作成
is_masked_channel = torch.zeros((B, L, 1))
is_masked_channel[mask_matrix] = 1.0

# 2. 連結して 5 次元に拡張
mag_inputs = torch.cat([mag_padded, is_masked_channel], dim=2)

# 3. マスク位置およびパディング位置のコンテンツ (0:4) をゼロ化
# 重要: パディング位置も含めてゼロ化することで、センチネル値 (-9999.0) のリークを防ぐ
valid_unmasked = valid_mask_bool & (~mask_matrix)  # 有効かつ非マスクのみ True
content_keep_mask = valid_unmasked.unsqueeze(-1).float()
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

### Step 5.5: Token ID 処理 (Vanilla Transformer 用)

**Vectorized Integer to Token ID Mapping:**

```
Token ID 構成:
  0: PAD  - パディング位置
  1: MASK - マスク位置（入力用）
  2: UNK  - 語彙外の整数 (負の数や大きすぎる数)
  3〜20002: 整数 0〜19999
```

**優先順位:**
1. 生整数列 (`"numbers"`) が存在する場合 → 正確なトークンIDを生成
2. 存在しない場合 → log magnitude から近似整数値を復元（フォールバック）

```python
max_int = VANILLA_VOCAB_SIZE - 3 - 1  # 19999

if "numbers" in batch[0]:
    # 生整数列から正確なトークンID生成
    # 
    # 重要: OEIS には int64 の範囲を超える巨大整数が含まれる。
    # torch.tensor(..., dtype=torch.long) でのオーバーフローを防ぐため、
    # 事前に範囲外の値をセンチネル値に変換する。
    INT64_MAX = 2**63 - 1
    INT64_MIN = -(2**63)
    UNK_SENTINEL = max_int + 1  # 後で UNK にマップされる
    
    def safe_clamp(numbers):
        return [n if INT64_MIN <= n <= INT64_MAX else UNK_SENTINEL for n in numbers]
    
    numbers_list = [torch.tensor(safe_clamp(item["numbers"])) for item in batch]
    numbers_padded = pad_sequence(numbers_list, batch_first=True, padding_value=0)
    
    # 非負かつ語彙内 → 有効トークン、それ以外 → UNK
    in_vocab_mask = (numbers_padded >= 0) & (numbers_padded <= max_int)
    token_ids = torch.where(
        in_vocab_mask,
        numbers_padded + 3,  # オフセット 3 (PAD, MASK, UNK)
        VANILLA_UNK_TOKEN_ID  # 負の数、語彙外、巨大整数 → UNK
    )
else:
    # フォールバック: log magnitude から近似整数値を復元
    # 注: 符号情報が失われ、丸め誤差が発生する
    log_vals = mag_padded[..., 0]
    approx_abs = torch.pow(10.0, log_vals) - 1
    approx_abs = torch.clamp(approx_abs, min=0).long()
    
    in_vocab_mask = (approx_abs >= 0) & (approx_abs <= max_int)
    token_ids = torch.where(in_vocab_mask, approx_abs + 3, VANILLA_UNK_TOKEN_ID)

# マスクトークン適用
token_ids = torch.where(mask_matrix, VANILLA_MASK_TOKEN_ID, token_ids)

# パディング位置を PAD に
token_ids = torch.where(valid_mask_bool, token_ids, VANILLA_PAD_TOKEN_ID)

# ラベル: マスク位置のみ正解トークン、他は IGNORE_INDEX
token_labels = torch.where(mask_matrix, original_token_ids, IGNORE_INDEX)
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
| Vectorized Token ID 生成 | Python ループを避けて高速化 |
| log magnitude からの整数復元 | データセットに生整数がない場合の近似方法 |
| `VOCAB_SIZE = 20003` | GPU メモリ 8GB 環境での実用的な上限 |
