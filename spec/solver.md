# `src/intseq_bert/solver.py` 実装仕様書

## 目次

1. [概要](#1-概要)
2. [依存関係](#2-依存関係)
3. [クラス設計](#3-クラス設計)
4. [アルゴリズム詳細フロー](#4-アルゴリズム詳細フロー)
5. [各モードの実装ロジック](#5-各モードの実装ロジック)
6. [ヘルパー関数](#6-ヘルパー関数)
7. [注意事項](#7-注意事項)

---

## 1. 概要

IntSeqBERTの推論結果（Magnitudeの平均と分散、Sign、Moduloの確率分布）を入力とし、元の整数系列を復元する `IntegerSolver` クラスを実装する。
探索範囲の広さに応じて3つの探索モード（Mode A, AB, B）を動的に切り替えるハイブリッド・アルゴリズムを採用する。

---

## 2. 依存関係

### ライブラリ

- `torch`, `math`, `typing`

### 設定 (`config.py`)

| 定数 | 値 | 用途 |
|------|------|------|
| `MOD_RANGE` | `list(range(2, 102))` | 法のリスト (2〜101) |
| `NUM_MODULI` | 100 | 法の数 |
| `EPSILON` | `1e-6` | ゼロ除算防止 |

### 追加定義（`config.py` に追加）

```python
# Solver thresholds
SOLVER_DENSE_THRESHOLD = 1_000_000          # Mode A → AB 切替閾値
SOLVER_SIEVE_THRESHOLD = 100_000_000_000_000  # Mode AB → B 切替閾値 (10^14)
SOLVER_SIEVE_TARGET = 100_000               # Anchored Sieve の候補数目標
SOLVER_BEAM_WIDTH = 10                      # CRT Beam Search のビーム幅
SOLVER_MAX_ANCHORS = 20                     # アンカーの最大数

# Scoring Weights (Modulo重複計上によるバイアス防止)
SOLVER_MAG_WEIGHT = 1.0                     # Magnitude スコアの重み
SOLVER_MOD_WEIGHT = 0.3                     # Modulo スコアの重み (2,4,8等の冗長法を割引)
```

### 関連モジュール

- `intseq_models.py` / `vanilla_models.py`: モデルの出力を入力として使用
- `base_models.py`: `BaseForPreTraining` 共通インターフェース
- `features.py`: 特徴量計算（参照用）

---

## 3. クラス設計

### クラス名: `IntegerSolver`

### コンストラクタ

```python
def __init__(self, config=None):
    """
    Args:
        config: 設定オブジェクト (省略時は config.py のデフォルト値を使用)
    
    Attributes:
        mod_range: List[int] - 法のリスト (2〜101)
        dense_threshold: int - Mode A → AB 切替閾値 (default: 1,000,000)
        sieve_threshold: int - Mode AB → B 切替閾値 (default: 10^14)
        sieve_target: int - Anchored Sieve の候補数目標 (default: 100,000)
        beam_width: int - CRT Beam Search のビーム幅 (default: 10)
    """
```

### 主要メソッド

```python
def solve(
    self,
    mag_mu: float,
    mag_log_var: float,
    sign_idx: int,
    mod_log_probs: List[torch.Tensor],
    top_k: int = 5
) -> List[Dict]:
    """
    モデル予測から元の整数を推定する。
    
    Args:
        mag_mu (float): Magnitudeの予測平均 (log10スケール, 1 + log10(|x|))
        mag_log_var (float): Magnitudeの予測対数分散 (不確実性)
        sign_idx (int): 符号インデックス (0=Positive, 1=Negative, 2=Zero)
        mod_log_probs (List[Tensor]): 各法の対数確率分布リスト
            - mod_log_probs[i] は形状 (m,) で m = MOD_RANGE[i]
            - log_softmax 済みの対数確率
        top_k (int): 返却する候補数 (default: 5)
    
    Returns:
        List[Dict]: 候補リスト（スコア降順）
            - value (int): 推定された整数値
            - score (float): 対数尤度スコア（大きいほど良い）
            - method (str): 使用したモード ("dense", "sieve", "crt")
    
    Raises:
        ValueError: sign_idx が 0, 1, 2 以外の場合
    """
```

### モデル出力からの変換

`IntSeqForPreTraining` の出力を `solve()` の入力形式に変換するヘルパーメソッド。

```python
@staticmethod
def from_model_output(
    predictions: Dict,
    position: int,
    model: "BaseForPreTraining"  # IntSeqForPreTraining or VanillaTransformerForPreTraining
) -> Tuple[float, float, int, List[torch.Tensor]]:
    """
    モデルの predictions 辞書から solve() の入力形式に変換する。
    
    Args:
        predictions: model.forward() の返り値["predictions"]
            - mag_mu: (B, L) Magnitude 予測平均
            - mag_log_var: (B, L) Magnitude 予測対数分散
            - sign_logits: (B, L, 3) Sign ロジット
            - mod_logits: (B, L, ~5150) 全 Modulo ロジット結合
        position: 系列内の位置インデックス (0-based)
        model: _split_mod_logits メソッドを持つモデルインスタンス
    
    Returns:
        Tuple of (mag_mu, mag_log_var, sign_idx, mod_log_probs)
    
    Example:
        >>> outputs = model(mag_features, mod_features, mask)
        >>> args = IntegerSolver.from_model_output(outputs["predictions"], pos=5, model=model)
        >>> candidates = solver.solve(*args)
    """
    # 実装:
    # 1. mag_mu[0, position].item()
    # 2. mag_log_var[0, position].item()
    # 3. sign_logits[0, position].argmax().item()
    # 4. mod_logits を _split_mod_logits で分割し、各法に log_softmax 適用
```

---

## 4. アルゴリズム詳細フロー

メソッド `solve` 内で以下の手順を実行する。

### Step 1: 前処理

1. **Sign処理:**
   - `sign_idx == 2` (Zero) なら即座に `[{"value": 0, "score": 0.0, "method": "zero"}]` を返す
   - `sign_idx == 1` (Negative) なら最終結果を負にするフラグを立てる

2. **探索範囲の決定:**
   ```python
   sigma = math.exp(0.5 * mag_log_var)
   # mag_mu は 1 + log10(|x|) なので、実際の log10(|x|) は mag_mu - 1
   log10_center = mag_mu - 1
   n_min = max(1, math.floor(10 ** (log10_center - 3 * sigma)))
   n_max = math.ceil(10 ** (log10_center + 3 * sigma))
   width = n_max - n_min
   ```

3. **オーバーフロー対策:**
   - `mag_mu > 100` の場合、`n_max` を `10^100` に制限するか、強制的に Mode B へ移行

### Step 2: モード分岐

`width` (範囲の広さ) に応じて以下のメソッドに分岐する。

| 条件 | モード | メソッド | 説明 |
|------|--------|----------|------|
| `width <= DENSE_THRESHOLD` | Mode A | `_solve_dense()` | 全探索 |
| `DENSE_THRESHOLD < width <= SIEVE_THRESHOLD` | Mode AB | `_solve_sieve()` | アンカー・シーブ |
| `width > SIEVE_THRESHOLD` | Mode B | `_solve_crt()` | Sparse CRT |

---

## 5. 各モードの実装ロジック

全てのモードで共通のスコアリング関数を使用する。

### スコアリング関数

```python
def _compute_score(
    self, n: int, mag_mu: float, sigma: float, mod_log_probs: List[Tensor],
    mag_weight: float = config.SOLVER_MAG_WEIGHT,
    mod_weight: float = config.SOLVER_MOD_WEIGHT
) -> float:
    """
    候補整数 n のスコア（対数尤度）を計算する。
    
    Total Score = mag_weight × LogLikelihood(Magnitude) + mod_weight × Sum(LogLikelihood(Mods))
    
    Args:
        n: 候補整数 (正の整数)
        mag_mu: Magnitude 予測平均 (1 + log10(|x|) スケール)
        sigma: Magnitude の標準偏差
        mod_log_probs: 各法の対数確率分布リスト
        mag_weight: Magnitude スコアの重み (デフォルト: 1.0)
        mod_weight: Modulo スコアの重み (デフォルト: 0.3、冗長法による偏り防止)
    
    Returns:
        float: 対数尤度スコア（大きいほど良い）
    """
    # Magnitude項: Gaussian log-likelihood
    log10_n = math.log10(n) if n > 0 else 0
    mag_target = 1 + log10_n  # 1 + log10(|x|) スケールに合わせる
    mag_score = -((mag_target - mag_mu) ** 2) / (2 * sigma ** 2 + EPSILON)
    
    # Modulo項: 各法の対数確率の和
    mod_score = 0.0
    for i, m in enumerate(self.mod_range):
        remainder = n % m
        mod_score += mod_log_probs[i][remainder].item()
    
    return (mag_weight * mag_score) + (mod_weight * mod_score)
```

### (1) Mode A: Dense Search (全探索)

```python
def _solve_dense(self, n_min, n_max, mag_mu, sigma, mod_log_probs, top_k) -> List[Dict]:
    """
    範囲内の全整数を総当たりで評価する。
    
    計算量: O(width × NUM_MODULI)
    """
    candidates = []
    for n in range(n_min, n_max + 1):
        score = self._compute_score(n, mag_mu, sigma, mod_log_probs)
        candidates.append({"value": n, "score": score, "method": "dense"})
    
    # スコア降順でソートし、上位 top_k 件を返す
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]
```

### (2) Mode AB: Anchored Sieve (アンカー・シーブ)

```python
def _solve_sieve(self, n_min, n_max, mag_mu, sigma, mod_log_probs, top_k) -> List[Dict]:
    """
    信頼できる少数の法（アンカー）で候補を絞り込んでから評価する。
    """
```

**1. アンカー選定:**

```python
# 各法の「信頼度」= 最大確率 (確信度が高いほど良い)
confidences = [(i, mod_log_probs[i].max().item()) for i in range(len(self.mod_range))]
confidences.sort(key=lambda x: x[1], reverse=True)

anchors = []
lcm = 1
for idx, _ in confidences:
    m = self.mod_range[idx]
    new_lcm = math.lcm(lcm, m)
    anchors.append(idx)
    lcm = new_lcm
    
    # 終了条件: 候補数が目標以下になるまで
    expected_candidates = (n_max - n_min) // lcm + 1
    if expected_candidates <= self.sieve_target:
        break
    
    # アンカー数の上限
    if len(anchors) >= config.SOLVER_MAX_ANCHORS:
        break
```

**2. 候補生成 (Beam Search CRT):**

```python
# 各アンカーの確率上位 k 個の余りを取得（ビームサーチ）
def get_top_remainders(log_probs: Tensor, k: int) -> List[Tuple[int, float]]:
    """(remainder, log_prob) のリストを返す"""
    values, indices = log_probs.topk(min(k, len(log_probs)))
    return [(idx.item(), val.item()) for idx, val in zip(indices, values)]

# Beam Search で CRT の組み合わせを探索
beams = [(0, 1, 0.0)]  # (current_x, current_M, cum_log_prob)
for anchor_idx in anchors:
    m = self.mod_range[anchor_idx]
    top_rems = get_top_remainders(mod_log_probs[anchor_idx], self.beam_width)
    
    new_beams = []
    for (x, M, cum_prob) in beams:
        for (r, log_p) in top_rems:
            try:
                new_x, new_M = solve_crt([(x, M), (r, m)])
                new_beams.append((new_x, new_M, cum_prob + log_p))
            except ValueError:
                continue  # CRT 解なし（矛盾する余り）
    
    # 上位ビーム幅のみ保持
    new_beams.sort(key=lambda x: x[2], reverse=True)
    beams = new_beams[:self.beam_width]

# ビームから範囲内の候補を列挙
candidate_set = set()
for (x, M, _) in beams:
    # x + k*M が [n_min, n_max] に入る k を全て列挙
    k_start = max(0, (n_min - x + M - 1) // M)
    k_end = (n_max - x) // M
    for k in range(k_start, k_end + 1):
        n = x + k * M
        if n_min <= n <= n_max:
            candidate_set.add(n)
```

**3. 詳細スコアリング:**

```python
# 候補に対して全法でスコアリング
candidates = []
for n in candidate_set:
    score = self._compute_score(n, mag_mu, sigma, mod_log_probs)
    candidates.append({"value": n, "score": score, "method": "sieve"})

candidates.sort(key=lambda x: x["score"], reverse=True)
return candidates[:top_k]
```

### (3) Mode B: Sparse CRT (巨大数向け)

```python
def _solve_crt(self, n_min, n_max, mag_mu, sigma, mod_log_probs, top_k) -> List[Dict]:
    """
    CRT で構築された候補のみを評価する（巨大範囲向け）。
    """
```

**1. 基底選定:**

```python
# 信頼度順にソート
confidences = [(i, mod_log_probs[i].max().item()) for i in range(len(self.mod_range))]
confidences.sort(key=lambda x: x[1], reverse=True)

basis = []
lcm = 1
width = n_max - n_min
for idx, _ in confidences:
    m = self.mod_range[idx]
    basis.append(idx)
    lcm = math.lcm(lcm, m)
    
    # 終了条件: LCM が探索範囲を超えるまで (解が一意に定まる)
    if lcm > width:
        break
```

**2. 候補生成 (Beam Search):**

Mode AB と同様の Beam Search を実行。

**3. スコアリング & 範囲補正:**

```python
candidates = []
for (x, M, cum_prob) in beams:
    # x が範囲外でも、周期 M を加減算して範囲内に入れる
    if x < n_min:
        k = (n_min - x + M - 1) // M
        x = x + k * M
    elif x > n_max:
        k = (x - n_max + M - 1) // M
        x = x - k * M
    
    if n_min <= x <= n_max:
        score = self._compute_score(x, mag_mu, sigma, mod_log_probs)
        candidates.append({"value": x, "score": score, "method": "crt"})

candidates.sort(key=lambda x: x["score"], reverse=True)
return candidates[:top_k]
```

---

## 6. ヘルパー関数

以下の数学関数を実装する。

### `extended_gcd(a, b)`

```python
def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    拡張ユークリッドの互除法。
    
    ax + by = gcd(a, b) となる (g, x, y) を返す。
    """
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1
```

### `solve_crt(equations)`

```python
def solve_crt(equations: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    中国剰余定理を解く。
    
    Args:
        equations: [(remainder_1, modulus_1), (remainder_2, modulus_2), ...] のリスト
    
    Returns:
        (x, M): x ≡ r_i (mod m_i) を満たす最小非負解 x と周期 M = lcm(m_1, ..., m_n)
    
    Raises:
        ValueError: 解が存在しない場合（法が互いに素でなく、余りが矛盾する場合）
    """
    x, M = 0, 1
    for r, m in equations:
        g, p, q = extended_gcd(M, m)
        if (r - x) % g != 0:
            raise ValueError(f"No solution: {x} mod {M} and {r} mod {m} are inconsistent")
        lcm = M * m // g
        x = (x + M * ((r - x) // g) * p) % lcm
        M = lcm
    return x, M
```

### `lcm(a, b)`

```python
def lcm(a: int, b: int) -> int:
    """最小公倍数を計算する。"""
    import math
    return abs(a * b) // math.gcd(a, b)
```

> **Note:** Python 3.9+ では `math.lcm()` が利用可能。

---

## 7. 注意事項

### 7.1. 数値オーバーフロー

- 周期 `M` や候補 `n` は非常に大きな整数（Pythonの多倍長整数）になる可能性がある
- `log10(n)` の計算時は `n <= 0` のガードを入れること
- `math.log10()` は `n > 0` でのみ有効

### 7.2. 対数確率の使用

- 確率の積はアンダーフローするため、必ず `log(prob)` の和として計算する
- **重要:** `mod_log_probs` は `log_softmax` 適用済みの対数確率として受け取る
  - モデル出力の `mod_logits` から変換する際は `F.log_softmax(logits, dim=-1)` を使用
- 対数確率が `-inf` になる場合は、`torch.clamp(log_probs, min=-100)` 等で下限を設定

### 7.3. ビームサーチ

- CRT の候補生成時、全ての組み合わせを試すと爆発する（指数的）
- 上位 `BEAM_WIDTH` 個の有望なパスだけを残すビームサーチを行う
- デフォルトのビーム幅は `config.SOLVER_BEAM_WIDTH = 10`

### 7.4. CRT の解なしケース

- 余りが互いに矛盾する場合（例: `x ≡ 0 (mod 4)` かつ `x ≡ 3 (mod 2)`）は CRT で解が存在しない
- Beam Search 内で `ValueError` をキャッチし、その組み合わせをスキップする

### 7.5. Magnitude スケールの変換

- モデルの `mag_mu` 出力は `1 + log10(|x|)` スケール
- 探索範囲計算時は `log10(|x|) = mag_mu - 1` として扱う
- スコアリング時も同様に `1 + log10(n)` と比較する

---

## 8. 将来の拡張検討

| 項目 | 内容 | 優先度 |
|------|------|--------|
| バッチ処理 | 複数位置を並列に solve する | 中 |
| GPU 高速化 | Dense Search のベクトル化 | 低 |
| キャッシュ | LCM / CRT 結果のメモ化 | 低 |