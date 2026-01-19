# `VanillaTransformer` 実装仕様書

## 目次

1. [概要](#1-概要)
2. [依存関係](#2-依存関係)
3. [クラス設計](#3-クラス設計)
4. [入出力形状](#4-入出力形状)
5. [損失計算](#5-損失計算)
6. [共通コンポーネント](#6-共通コンポーネント)
7. [ヘルパーメソッド](#7-ヘルパーメソッド)

---

## 1. 概要

**対象ファイル:** `src/intseq_bert/vanilla_models.py`

**目的:**
IntSeqBERT との比較実験（Baseline）用として、以下の3つのクラスを実装する。
これらは FACT 論文のアプローチ（数値をトークンIDとして扱う）に基づく。

| クラス | 役割 |
|--------|------|
| `VanillaEmbeddings` | 数値IDの埋め込みと位置エンコーディング |
| `VanillaTransformerModel` | Transformer Encoder バックボーン |
| `VanillaTransformerForPreTraining` | 学習用ヘッド（ID予測 + 診断用ヘッド） |

---

## 2. 依存関係

### 2.1. config.py への追加定数

| 定数 | デフォルト値 | 説明 |
|------|------------|------|
| `VANILLA_VOCAB_SIZE` | 30000 | トークン語彙サイズ |
| `VANILLA_PAD_TOKEN_ID` | 0 | パディングトークンID |
| `VANILLA_UNK_TOKEN_ID` | 2 | 未知トークンID |
| `VANILLA_MASK_TOKEN_ID` | 1 | マスクトークンID |

### 2.2. 既存の使用定数

`D_MODEL`, `NHEAD`, `NUM_LAYERS`, `DROPOUT`, `MOD_RANGE`, `MAX_SEQUENCE_LENGTH`

---

## 3. クラス設計

### 3.1. `VanillaEmbeddings`

```python
class VanillaEmbeddings(nn.Module):
    """Token embedding with positional encoding for Vanilla Transformer."""
```

| 引数 | 型 | デフォルト |
|------|------|------|
| `vocab_size` | int | `config.VANILLA_VOCAB_SIZE` |
| `d_model` | int | `config.D_MODEL` |
| `dropout` | float | `config.DROPOUT` |
| `max_len` | int | `config.MAX_SEQUENCE_LENGTH` |

**構成:**

| コンポーネント | 定義 |
|---------------|------|
| `token_embedding` | `nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)` |
| `pos_encoding` | Sinusoidal Encoding (加算) |
| `layer_norm` | `nn.LayerNorm(d_model)` |
| `dropout` | `nn.Dropout(dropout)` |

### 3.2. `VanillaTransformerModel`

```python
class VanillaTransformerModel(nn.Module):
    """Transformer Encoder backbone for Vanilla model."""
```

| 引数 | 型 | デフォルト |
|------|------|------|
| `d_model` | int | `config.D_MODEL` |
| `nhead` | int | `config.NHEAD` |
| `num_layers` | int | `config.NUM_LAYERS` |
| `dropout` | float | `config.DROPOUT` |

**構成:**

| コンポーネント | 定義 |
|---------------|------|
| `embeddings` | `VanillaEmbeddings(...)` |
| `encoder_layer` | `nn.TransformerEncoderLayer(batch_first=True, norm_first=True)` |
| `encoder` | `nn.TransformerEncoder(encoder_layer, num_layers)` |

### 3.3. `VanillaTransformerForPreTraining`

```python
class VanillaTransformerForPreTraining(nn.Module):
    """Vanilla Transformer with multi-task heads for pre-training."""
```

**構成:**

| コンポーネント | 定義 | 説明 |
|---------------|------|------|
| `transformer` | `VanillaTransformerModel(...)` | バックボーン |
| `lm_head` | `nn.Linear(d_model, vocab_size)` | **メイン**: トークン予測 |
| `mag_head` | `nn.Linear(d_model, 2)` | 診断用: Magnitude |
| `sign_head` | `nn.Linear(d_model, 3)` | 診断用: 符号 |
| `mod_head` | `nn.Linear(d_model, sum(MOD_RANGE))` | 診断用: Modulo (5150) |

---

## 4. 入出力形状

### 4.1. Forward 引数

| クラス | 引数 | 形状 |
|--------|------|------|
| `VanillaEmbeddings` | `input_ids` | `(B, L)` LongTensor |
| `VanillaTransformerModel` | `input_ids`, `src_key_padding_mask` | `(B, L)`, `(B, L)` Bool |
| `VanillaTransformerForPreTraining` | `input_ids`, `src_key_padding_mask`, `labels` | 同上 + Optional Dict |

### 4.2. Forward 出力

```python
# VanillaEmbeddings
embeddings: (B, L, d_model)

# VanillaTransformerModel  
last_hidden_state: (B, L, d_model)

# VanillaTransformerForPreTraining
{
    "predictions": {
        "logits": (B, L, vocab_size),      # メイン出力
        "mag_mu": (B, L),                  # 診断用
        "mag_log_var": (B, L),             # 診断用
        "mod_logits": (B, L, 5150),        # 診断用
        "sign_logits": (B, L, 3)           # 診断用
    },
    "loss": Tensor or None,
    "loss_components": {...} or None
}
```

---

## 5. 損失計算

### 5.1. 損失関数

| ヘッド | 損失関数 | 重み | 備考 |
|--------|----------|------|------|
| `lm_head` | `CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)` | 1.0 | **メイン** |
| `mag_head` | `GaussianNLLLoss` | 0.1 | 診断用 |
| `sign_head` | `CrossEntropyLoss` | 0.1 | 診断用 |
| `mod_head` | `CrossEntropyLoss` (per modulus) | 0.1 | 診断用 |

> 診断用ヘッドの重みは 0.0 にすれば純粋な LM として学習可能。

### 5.2. マスキング戦略

IntSeqBERT と同じ **Masked Language Model (MLM)** 方式:

1. 入力の 15% をマスク (`VANILLA_MASK_TOKEN_ID` に置換)
2. マスク位置のトークンを予測
3. 非マスク位置は損失計算から除外

---

## 6. 共通コンポーネント

### 6.1. Positional Encoding

`models.py` 内の `_generate_sinusoidal_encoding()` 関数を両モデルで共有。

### 6.2. `_split_mod_logits`

IntSeqBERT と同一のロジック。Mixin またはコピーで実装。

```python
def _split_mod_logits(self, logits: torch.Tensor) -> List[torch.Tensor]:
    return torch.split(logits, config.MOD_RANGE, dim=-1)
```

---

## 7. ヘルパーメソッド

### 7.1. `from_checkpoint`

```python
@classmethod
def from_checkpoint(cls, path: str, device: str = "cpu"):
    checkpoint = torch.load(path, map_location=device)
    model = cls(**checkpoint.get("config", {}))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()
```

### 7.2. トークン化

数値 → トークンID 変換:

```python
def tokenize_number(n: int) -> int:
    if 0 <= n < VANILLA_VOCAB_SIZE:
        return n
    return VANILLA_UNK_TOKEN_ID  # 範囲外は UNK
```
