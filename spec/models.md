# `src/intseq_bert/models.py` 実装仕様書

## 1. 概要

本モジュールは、IntSeqBERT のニューラルネットワーク定義を担当する。
HuggingFace Transformers の設計思想を踏襲し、**埋め込み層 (`Embeddings`)**、**ベースモデル (`Model`)**、**事前学習用モデル (`ForPreTraining`)** の3層クラス構成とする。

特徴:
- **FiLM (Feature-wise Linear Modulation)** によるデュアルストリーム統合
- **Heteroscedastic Regression** による不確実性推定
- **Automatic Weighted Loss** によるマルチタスク学習の安定化

---

## 2. 依存関係

### ライブラリ
- `torch`, `torch.nn`, `math`

### 設定 (`config.py`)

| 定数 | 値 | 用途 |
|------|------|------|
| `MAG_EXTENDED_DIM` | 5 | 入力 Magnitude 次元（is_masked 含む） |
| `MOD_FEATURE_DIM` | 200 | 入力 Modulo Sin/Cos 次元 |
| `MOD_RANGE` | `list(range(2, 102))` | 法のリスト (2〜101) |
| `NUM_MODULI` | 100 | 法の数 |

### 追加定義（`config.py` に追加）

```python
# Sign class indices (matches MAG_EXTENDED_DIM order)
SIGN_POSITIVE = 0  # sign+ column in features
SIGN_NEGATIVE = 1  # sign- column in features
SIGN_ZERO = 2      # sign0 column in features
```

---

## 3. クラス設計

### 3.1. `IntSeqEmbeddings` (Input Layer)

数値の「大きさ」と「周期性」を融合する層。周期性が大きさを「変調（Modulate）」する FiLM 機構を採用。

#### `__init__` 引数

| 引数 | 型 | デフォルト | 説明 |
|------|------|-----------|------|
| `d_model` | int | `config.D_MODEL` | 隠れ層次元 |
| `dropout` | float | `config.DROPOUT` | ドロップアウト率 |
| `max_len` | int | `config.MAX_SEQUENCE_LENGTH` | 最大系列長 |

#### ネットワーク構成

```python
mag_proj:    Linear(MAG_EXTENDED_DIM, d_model)
mod_proj:    Linear(MOD_FEATURE_DIM, d_model)
film_scale:  Linear(d_model, d_model)  # γ生成
film_shift:  Linear(d_model, d_model)  # β生成
pos_encoding: Sinusoidal (固定、max_len x d_model)
layer_norm:  LayerNorm(d_model)
dropout:     Dropout(dropout)
```

#### 初期化

```python
# FiLM γ を 0 に初期化（学習初期に h_mag を破壊しない）
nn.init.zeros_(self.film_scale.weight)
nn.init.zeros_(self.film_scale.bias)
```

#### `forward` 入出力

**入力:**
- `mag_features`: `(B, L, 5)` - Magnitude stream (with is_masked flag)
- `mod_features`: `(B, L, 200)` - Modulo Sin/Cos stream

**出力:**
- `embeddings`: `(B, L, d_model)`

#### 処理フロー

```
1. Projection:
   h_mag = mag_proj(mag_features)          # (B, L, d_model)
   h_mod = ReLU(mod_proj(mod_features))    # (B, L, d_model)

2. FiLM Parameter Generation:
   γ = film_scale(h_mod)                   # (B, L, d_model)
   β = film_shift(h_mod)                   # (B, L, d_model)

3. Modulation:
   h_fused = (1 + γ) ⊙ h_mag + β           # Element-wise

4. Post-Process:
   h_out = LayerNorm(h_fused + PosEncoding[:L])
   h_out = Dropout(h_out)
```

---

### 3.2. `IntSeqModel` (Base Backbone)

Transformer Encoder をラップするメインモデル。

#### `__init__` 引数

| 引数 | 型 | 説明 |
|------|------|------|
| `d_model` | int | 隠れ層次元 |
| `nhead` | int | Attention ヘッド数 |
| `num_layers` | int | Encoder 層数 |
| `dropout` | float | ドロップアウト率 |

#### 構成

```python
embeddings: IntSeqEmbeddings(d_model, dropout)
encoder:    nn.TransformerEncoder(
              nn.TransformerEncoderLayer(
                d_model, nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True  # Pre-LN for training stability
              ),
              num_layers=num_layers
            )
```

#### `forward` 入出力

**入力:**
- `mag_features`: `(B, L, 5)`
- `mod_features`: `(B, L, 200)`
- `src_key_padding_mask`: `(B, L)` - BoolTensor, `True` = Padding

**出力:**
- `last_hidden_state`: `(B, L, d_model)`

---

### 3.3. `IntSeqForPreTraining` (Heads & Loss)

事前学習（Masked Modeling）用のヘッドと損失計算を持つラッパークラス。

#### 予測ヘッド構成

| ヘッド | 構造 | 出力 |
|--------|------|------|
| `mag_head` | `Linear(d_model, d_model) → ReLU → Linear(d_model, 2)` | `[μ, log(σ²)]` |
| `sign_head` | `Linear(d_model, 3)` | Logits for `[Positive, Negative, Zero]` |
| `mod_head` | `Linear(d_model, sum(MOD_RANGE))` | 全 Modulo ロジット結合 (~5150次元) |

> **Note:** `sign_head` のクラス順序は `config.SIGN_POSITIVE=0`, `SIGN_NEGATIVE=1`, `SIGN_ZERO=2` に対応。

#### 学習可能損失パラメータ

```python
loss_log_vars: nn.Parameter(torch.zeros(3))  # [s_mag, s_sign, s_mod]
```

Automatic Weighted Loss (Kendall et al., 2018) 用のノイズレベルパラメータ。

#### `forward` 入出力

**入力:**

| 引数 | 型 | 必須 | 説明 |
|------|------|------|------|
| `mag_features` | `(B, L, 5)` | ✅ | Magnitude stream |
| `mod_features` | `(B, L, 200)` | ✅ | Modulo stream |
| `src_key_padding_mask` | `(B, L)` | ✅ | Padding mask |
| `labels` | Dict | | 学習時のみ |

**labels 辞書:**

| キー | 型 | 説明 |
|------|------|------|
| `mag_targets` | `(B, L)` Float | 元の `MAG_FEATURES[:, :, 0]` (= `1 + log10(\|x\|)`) |
| `sign_targets` | `(B, L)` Long | クラスインデックス (0=Pos, 1=Neg, 2=Zero) |
| `mod_targets` | `(B, L, 100)` Long | 各法の剰余値 |
| `mask_map` | `(B, L)` Bool | `True` = マスク位置（損失計算対象） |

**出力:**

```python
{
  "loss": Tensor (scalar),  # 学習時のみ
  "predictions": {
    "mag_mu": (B, L),
    "mag_log_var": (B, L),
    "sign_logits": (B, L, NUM_SIGN_CLASSES),
    "mod_logits": (B, L, ~5150)
  },
  "loss_breakdown": {  # 学習時のみ、モニタリング用
    "raw_mag": Tensor (scalar),   # Magnitude 損失
    "raw_sign": Tensor (scalar),  # Sign 損失
    "raw_mod": Tensor (scalar),   # Modulo 損失
    "s_mag": Tensor (scalar),     # Mag の学習済み重み
    "s_sign": Tensor (scalar),    # Sign の学習済み重み
    "s_mod": Tensor (scalar)      # Mod の学習済み重み
  }
}
```

---

## 4. 損失計算

マスクされた位置 (`mask_map == True`) のみを対象に計算。

### 4.1. Magnitude Loss (Heteroscedastic Gaussian NLL)

```
L_mag = (1/2) * log(σ²) + (μ - y)² / (2σ²)
```

σ² = exp(log_var) として数値安定性を確保。

### 4.2. Sign Loss

```
L_sign = CrossEntropyLoss(sign_logits, sign_targets)
```

### 4.3. Modulo Loss

```python
# 各法の損失を、その法の最大エントロピー log(m) で正規化する
# これにより、ランダム予測時の損失が全ての法で 1.0 に揃う
L_mod_list = []
for m in MOD_RANGE:
    loss_m = CrossEntropy(logits_m, targets[:, m])
    norm_loss_m = loss_m / log(m)  # normalize by natural log
    L_mod_list.append(norm_loss_m)

L_mod = mean(L_mod_list)
```

`mod_head` 出力を `_split_mod_logits()` で各法にスライス。

### 4.4. 統合損失 (Automatic Weighted Loss)

```
L_total = Σ_i [ (1/2) * exp(-s_i) * L_i + (1/2) * s_i ]
```

ここで `s_i = loss_log_vars[i]` は学習可能パラメータ。

---

## 5. Helper Methods

### `_split_mod_logits(logits: Tensor) -> List[Tensor]`

巨大な `mod_head` 出力を各法に対応するロジットに分割。

```python
def _split_mod_logits(self, logits: Tensor) -> List[Tensor]:
    # logits: (*, sum(MOD_RANGE))
    return torch.split(logits, config.MOD_RANGE, dim=-1)
    # Returns: List of (*, 2), (*, 3), ..., (*, 101)
```

### `_generate_sinusoidal_encoding(max_len: int, d_model: int) -> Tensor`

固定 Sinusoidal Positional Encoding を生成。

```python
def _generate_sinusoidal_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (max_len, d_model)
```

---

## 6. 将来の拡張検討 (Phase 2)

| 項目 | 内容 | 優先度 |
|------|------|--------|
| Relative Positional Encoding | 隣接項関係の学習強化 (RoPE 等) | 中 |
| Gradient Checkpointing | メモリ効率化 | 低 |
