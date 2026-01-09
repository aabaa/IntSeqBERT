# `src/intseq_bert/features.py` 実装仕様書

## 1. 概要

本モジュールは、生の整数列をモデル学習用のテンソルに変換する**特徴量抽出ロジック**を担当する。
データパイプラインの最上流に位置し、`preprocess.py` から呼び出される。

### 出力ストリーム

| ストリーム | 内容 | 次元 |
|-----------|------|------|
| **Magnitude** | Log10 スケール + 符号 One-hot | `(L, 4)` |
| **Modulo Sin/Cos** | 単位円上の埋め込み | `(L, 200)` |
| **Modulo Integers** | 整数剰余（分類ラベル用） | `(L, 100)` |

---

## 2. 依存関係

```python
import math
import torch
from typing import List, Dict
from . import config
```

### 使用する config 定数

| 定数 | 値 | 用途 |
|------|------|------|
| `MAX_SEQUENCE_LENGTH` | 128 | 切り詰め上限 |
| `MAG_RAW_DIM` | 4 | Magnitude 出力次元 |
| `MOD_FEATURE_DIM` | 200 | Modulo Sin/Cos 出力次元 |
| `NUM_MODULI` | 100 | 法の数 |
| `MOD_RANGE` | `list(range(2, 102))` | 法のリスト [2, 3, ..., 101] |
| `KEY_MAG_FEATURES` | `"mag_features"` | 出力キー |
| `KEY_MOD_FEATURES` | `"mod_features"` | 出力キー |
| `KEY_MOD_INTEGERS` | `"mod_integers"` | 出力キー |

---

## 3. 関数設計

### 3.1. `compute_magnitude_features`

整数列を Magnitude 特徴量に変換する。

#### シグネチャ

```python
def compute_magnitude_features(sequence: List[int]) -> torch.Tensor
```

#### 入出力

| 項目 | 型 | 説明 |
|------|------|------|
| **入力** | `List[int]` | 整数列 |
| **出力** | `Tensor(L, 4)` | Magnitude 特徴量 |

#### 特徴量フォーマット

各整数 `x` に対して 4 次元ベクトルを生成:

```
[log_val, sign_plus, sign_minus, sign_zero]
```

| フィールド | 計算式 | 説明 |
|-----------|--------|------|
| `log_val` | `1.0 + log10(\|x\|)` | 対数スケール絶対値 |
| `sign_plus` | `1.0 if x > 0 else 0.0` | 正符号フラグ |
| `sign_minus` | `1.0 if x < 0 else 0.0` | 負符号フラグ |
| `sign_zero` | `1.0 if x == 0 else 0.0` | ゼロフラグ |

#### 特殊ケース

| ケース | log_val | signs |
|--------|---------|-------|
| `x = 0` | `0.0` | `[0, 0, 1]` |
| `x = 1` | `1.0` | `[1, 0, 0]` |
| `x = -10` | `2.0` | `[0, 1, 0]` |
| `x = 10^1000` (巨大数) | `1001.0` (フォールバック) | `[1, 0, 0]` |

#### オーバーフロー保護

```python
try:
    log_val = 1.0 + math.log10(val_abs)
except OverflowError:
    # float64 範囲を超える巨大整数
    log_val = float(len(str(val_abs)))
```

#### 空入力

```python
if not sequence:
    return torch.zeros((0, config.MAG_RAW_DIM), dtype=torch.float32)
```

---

### 3.2. `compute_modulo_features`

整数列を Modulo 特徴量と整数ラベルに変換する。

#### シグネチャ

```python
def compute_modulo_features(sequence: List[int]) -> tuple[torch.Tensor, torch.Tensor]
```

#### 入出力

| 項目 | 型 | 説明 |
|------|------|------|
| **入力** | `List[int]` | 整数列 |
| **出力1** | `Tensor(L, 200)` | Sin/Cos 埋め込み |
| **出力2** | `Tensor(L, 100)` | 整数剰余ラベル |

#### 計算式

各整数 `x` と各法 `m ∈ [2, 101]` に対して:

```
r = x % m                    # Python の正剰余
θ = (2π × r) / m             # 角度
features = [sin(θ), cos(θ)]  # 単位円埋め込み
```

#### 出力構造

**mod_features (L, 200):**

```
[sin_m2, cos_m2, sin_m3, cos_m3, ..., sin_m101, cos_m101]
```

100 個の法 × 2 (sin, cos) = 200 次元

**mod_integers (L, 100):**

```
[r_m2, r_m3, r_m4, ..., r_m101]
```

各法に対する剰余値（分類ラベル用）

#### 負数の剰余

Python の `%` 演算子は正の剰余を返す:

```python
-5 % 3  # => 1 (not -2)
```

#### 空入力

```python
if not sequence:
    return (
        torch.zeros((0, config.MOD_FEATURE_DIM), dtype=torch.float32),
        torch.zeros((0, config.NUM_MODULI), dtype=torch.long)
    )
```

---

### 3.3. `process_sequence` (メインエントリポイント)

単一シーケンスの処理パイプライン。

#### シグネチャ

```python
def process_sequence(sequence: List[int]) -> Dict[str, torch.Tensor]
```

#### 入出力

| 項目 | 型 | 説明 |
|------|------|------|
| **入力** | `List[int]` | 生の整数列 |
| **出力** | `Dict[str, Tensor]` | 特徴量辞書 |

#### 処理フロー

```
1. 切り詰め: len > MAX_SEQUENCE_LENGTH なら先頭 128 要素のみ
2. Magnitude 特徴量計算
3. Modulo 特徴量計算
4. 辞書にパック
```

#### 出力辞書

```python
{
    "mag_features": Tensor(L, 4),    # Magnitude
    "mod_features": Tensor(L, 200),  # Sin/Cos
    "mod_integers": Tensor(L, 100)   # Labels
}
```

> **Note:** パディングは行わない（Collator が担当）

---

## 4. データ型

| テンソル | dtype | 理由 |
|---------|-------|------|
| `mag_features` | `float32` | 連続値 |
| `mod_features` | `float32` | Sin/Cos 連続値 |
| `mod_integers` | `long` (int64) | CrossEntropy 用ラベル |

---

## 5. 単位円埋め込みの数学的背景

### なぜ Sin/Cos を使うか

剰余 `r` をカテゴリカルラベルとして扱うと、`0` と `m-1` が「遠い」と見なされる。
しかし mod 演算では `0 ≡ m`（周期性）なので、これらは「近い」べきである。

**Sin/Cos 埋め込み**により、単位円上で周期性を自然に表現:

```
r = 0    → θ = 0      → (sin=0, cos=1)
r = m/4  → θ = π/2    → (sin=1, cos=0)
r = m/2  → θ = π      → (sin=0, cos=-1)
r = m-1  → θ ≈ 2π     → (sin≈0, cos≈1)  ← r=0 と近い！
```

---

## 6. 使用例

```python
from intseq_bert.features import process_sequence

# Fibonacci sequence
sequence = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
result = process_sequence(sequence)

print(result["mag_features"].shape)   # torch.Size([10, 4])
print(result["mod_features"].shape)   # torch.Size([10, 200])
print(result["mod_integers"].shape)   # torch.Size([10, 100])

# 最初の要素 (x=0) の Magnitude
print(result["mag_features"][0])      # tensor([0., 0., 0., 1.])
```

---

## 7. 設計上の決定事項

| 決定 | 理由 |
|------|------|
| `1 + log10(\|x\|)` | `x=1` で `log=0` を避け、正の値域を確保 |
| One-hot 符号 | 符号を明示的に独立したチャネルで表現 |
| Sin/Cos 埋め込み | 周期性を連続的に表現、mod 間の類似性学習 |
| 整数ラベル保持 | 分類損失計算で必要 |
| 切り詰めのみ・パディングなし | 責任分離（Collator がバッチ処理時にパディング） |
