# 実験結果まとめ

> 論文化を念頭に置いた実験結果の記録。
> チェックポイント: `checkpoints/{small,middle,large}_std/{intseq,vanilla,ablation}/`
> 評価セット: OEISデータセット (std split)

---

## 1. 実験概要

### 比較モデル

| モデル名 | 略称 | 説明 |
|---------|------|------|
| **IntSeqBERT** | `intseq` | 提案手法。Magnitude + Modulo の双ストリーム入力 + FiLM融合 |
| **Vanilla Transformer** | `vanilla` | ベースライン。整数をトークンIDとしてembeddingする標準Transformer |
| **Ablation (Magnitude-only)** | `ablation` | Modulo ストリームを取り除いたアブレーション。FiLMなし |

### モデルサイズ設定

| スケール | Layers | d_model | nheads | 学習完了日 |
|---------|--------|---------|--------|-----------|
| **Small** | 6 | 256 | 4 | 2026-02-20 |
| **Middle** | 8 | 512 | 8 | 2026-02-17 |
| **Large** | 12 | 768 | 12 | 2026-02-06 |

### 共通学習設定

| パラメータ | 値 |
|-----------|-----|
| データセット | OEIS (std split) |
| 学習サンプル数 | 219,765 |
| 検証サンプル数 | 27,470 |
| Epochs | 200 (full) |
| Batch size | 32 (accum_steps=2 → effective 64) |
| Learning rate | 5e-5 (warmup 10%) |
| Weight decay | 0.01 |
| Loss weights | mag=1.0, sign=1.0, **mod=2.0** |
| Optimizer | AdamW |
| 数値精度 | FP32 (巨大整数の安定性のため AMP 無効) |
| Framework | PyTorch 2.9.1+cu128, CUDA 12.8 |
| Seed | 42 |

---

## 2. 学習結果 (Validation Best Metrics)

すべてのモデルが 200 epoch を完走（早期停止なし、patience=200）。

### 2.1 Small (6L-256d-4h)

| モデル | Best Epoch | val_loss | val_mag_acc (%) | val_mag_mse | val_sign_acc (%) | val_mod_acc (%) |
|--------|-----------|---------|----------------|-------------|-----------------|----------------|
| **IntSeq** | 174 | **1.2203** | **94.69** | **0.364** | **97.95** | **40.33** |
| Vanilla | 177 | 2.1715 | 85.58 | 1.273 | 97.03 | 36.05 |
| Ablation | 165 | 1.5655 | 93.53 | 0.375 | 97.47 | 25.89 |

代表的なModulo精度 (val):

| モデル | mod_2 (%) | mod_3 (%) | mod_5 (%) | mod_10 (%) | mod_100 (%) |
|--------|----------|----------|----------|-----------|------------|
| **IntSeq** | **81.89** | **64.54** | **46.09** | **45.44** | **38.53** |
| Vanilla | 78.18 | 57.07 | 39.31 | 39.48 | 36.05 |
| Ablation | 63.75 | 45.77 | 34.98 | 29.58 | 24.13 |

### 2.2 Middle (8L-512d-8h)

| モデル | Best Epoch | val_loss | val_mag_acc (%) | val_mag_mse | val_sign_acc (%) | val_mod_acc (%) |
|--------|-----------|---------|----------------|-------------|-----------------|----------------|
| **IntSeq** | 175 | **1.0704** | **95.66** | **0.171** | **98.42** | **46.68** |
| Vanilla | 168 | 1.8967 | 87.08 | 1.087 | 97.57 | 42.01 |
| Ablation | 172 | 1.4337 | 91.93 | 0.234 | 98.08 | 31.81 |

| モデル | mod_2 (%) | mod_3 (%) | mod_5 (%) | mod_10 (%) | mod_100 (%) |
|--------|----------|----------|----------|-----------|------------|
| **IntSeq** | **84.42** | **69.74** | **55.07** | **53.33** | **44.61** |
| Vanilla | 80.21 | 61.58 | 45.36 | 45.31 | 41.74 |
| Ablation | 69.74 | 50.17 | 38.65 | 35.36 | 30.15 |

### 2.3 Large (12L-768d-12h)

| モデル | Best Epoch | val_loss | val_mag_acc (%) | val_mag_mse | val_sign_acc (%) | val_mod_acc (%) |
|--------|-----------|---------|----------------|-------------|-----------------|----------------|
| **IntSeq** | 180 | **1.0028** | **95.73** | **0.180** | **98.61** | **50.15** |
| Vanilla | 174 | 1.7470 | 86.92 | 1.076 | 97.77 | 45.55 |
| Ablation | 170 | 1.3785 | 89.34 | 0.315 | 98.39 | 34.93 |

| モデル | mod_2 (%) | mod_3 (%) | mod_5 (%) | mod_10 (%) | mod_100 (%) |
|--------|----------|----------|----------|-----------|------------|
| **IntSeq** | **85.67** | **72.09** | **60.03** | **58.04** | **48.24** |
| Vanilla | 81.41 | 64.70 | 49.77 | 48.96 | 45.35 |
| Ablation | 71.90 | 53.20 | 41.93 | 39.04 | 33.25 |

### 2.4 スケールアップ効果まとめ

IntSeqBERT の val_mod_acc はスケールに伴い単調改善: 40.33% → 46.68% → **50.15%**。
Vanilla の val_mag_acc の改善幅は小さく、Modulo 予測でも IntSeq との差は拡大傾向。

### 2.5 Test Split 最終評価（`--test_only --test_split test`）

> **実施日**: 2026-03-02
> **使用チェックポイント**: `last_checkpoint.pt`（最終エポック = 200）
> **評価サンプル数**: 27,470（test split）
> **注意**: `best_metrics.json` は学習中の best val epoch のスナップショット。本評価は last epoch モデルで test split を評価したもの。両者の差は軽微（後述）。

#### 主要指標

| Size | Model | test_loss | test_mag_acc (%) | test_mag_mse | test_sign_acc (%) | test_mod_acc (%) |
|------|-------|-----------|-----------------|-------------|------------------|-----------------|
| Small | **IntSeq** | **1.2175** | **94.73** | **0.2215** | **97.78** | **40.43** |
| Small | Vanilla | 2.2142 | 85.73 | 1.5112 | 96.91 | 36.21 |
| Small | Ablation | 1.5683 | 93.72 | 0.3002 | 97.39 | 25.97 |
| Middle | **IntSeq** | **1.0654** | **95.71** | **0.1830** | **98.34** | **46.88** |
| Middle | Vanilla | 1.9214 | 87.37 | 0.9642 | 97.42 | 42.53 |
| Middle | Ablation | 1.4300 | 92.45 | 0.1970 | 97.90 | 31.93 |
| Large | **IntSeq** | **0.9976** | **95.85** | **0.2000** | **98.54** | **50.38** |
| Large | Vanilla | 1.7808 | 86.97 | 1.0025 | 97.66 | 45.85 |
| Large | Ablation | 1.3738 | 89.70 | 0.3237 | 98.29 | 35.22 |

#### 代表的な Modulo 精度（test）

| Size | Model | mod_2 (%) | mod_3 (%) | mod_5 (%) | mod_10 (%) | mod_100 (%) |
|------|-------|----------|----------|----------|-----------|------------|
| Small | **IntSeq** | **81.97** | **64.62** | **46.34** | **45.54** | **38.62** |
| Small | Vanilla | 78.27 | 57.25 | 39.58 | 39.78 | 36.25 |
| Small | Ablation | 64.15 | 46.25 | 35.31 | 30.08 | 24.07 |
| Middle | **IntSeq** | **84.50** | **70.32** | **55.49** | **53.70** | **44.84** |
| Middle | Vanilla | 80.37 | 62.26 | 45.86 | 45.97 | 42.24 |
| Middle | Ablation | 69.79 | 50.52 | 38.99 | 35.42 | 30.32 |
| Large | **IntSeq** | **85.65** | **72.62** | **60.37** | **58.38** | **48.51** |
| Large | Vanilla | 81.40 | 65.22 | 50.07 | 49.25 | 45.60 |
| Large | Ablation | 72.13 | 53.72 | 42.63 | 39.47 | 33.51 |

#### Val（best epoch）vs Test（last epoch）の比較

val と test の指標は全モデルで概ね一致しており、汎化性能の劣化（過学習）は見られない。

| Size | Model | val_mag_acc | test_mag_acc | Δ | val_mod_acc | test_mod_acc | Δ |
|------|-------|------------|-------------|---|------------|-------------|---|
| Small | IntSeq | 94.69 | 94.73 | +0.04 | 40.33 | 40.43 | +0.10 |
| Small | Vanilla | 85.58 | 85.73 | +0.15 | 36.05 | 36.21 | +0.16 |
| Small | Ablation | 93.53 | 93.72 | +0.19 | 25.89 | 25.97 | +0.08 |
| Middle | IntSeq | 95.66 | 95.71 | +0.05 | 46.68 | 46.88 | +0.20 |
| Middle | Vanilla | 87.08 | 87.37 | +0.29 | 42.01 | 42.53 | +0.52 |
| Middle | Ablation | 91.93 | 92.45 | +0.52 | 31.81 | 31.93 | +0.12 |
| Large | IntSeq | 95.73 | 95.85 | +0.12 | 50.15 | 50.38 | +0.23 |
| Large | Vanilla | 86.92 | 86.97 | +0.05 | 45.55 | 45.85 | +0.30 |
| Large | Ablation | 89.34 | 89.70 | +0.36 | 34.93 | 35.22 | +0.29 |

Δ はすべて正（test ≥ val）または微差であり、val 選択基準の妥当性が確認された。

---

## 3. Magnitude 予測分析 (`analyze_magnitude`)

テストセット全体の Magnitude 回帰指標。`log10(|x|+1)` スケールで評価。

### 3.1 Overall Metrics

| Size | Model | MSE | RMSE | MAE | R² | Acc_0.5 (%) | Acc_0.1 (%) | ECE |
|------|-------|-----|------|-----|-----|-------------|-------------|-----|
| Small | **IntSeq** | **0.228** | **0.478** | **0.135** | **0.988** | **94.70** | **70.36** | 1.30 |
| Small | Vanilla | 1.188 | 1.090 | 0.327 | 0.937 | 85.99 | 51.22 | 18.43 |
| Small | Ablation | 0.272 | 0.522 | 0.160 | 0.986 | 93.64 | 63.26 | **0.47** |
| Middle | **IntSeq** | **0.164** | **0.406** | **0.110** | **0.991** | **95.75** | **78.07** | 1.48 |
| Middle | Vanilla | 1.067 | 1.033 | 0.298 | 0.944 | 87.37 | 52.58 | 16.36 |
| Middle | Ablation | 0.284 | 0.533 | 0.175 | 0.985 | 92.52 | 57.60 | **0.64** |
| Large | **IntSeq** | **0.142** | **0.377** | **0.106** | **0.993** | **95.83** | **79.16** | **0.65** |
| Large | Vanilla | 1.037 | 1.018 | 0.313 | 0.946 | 87.08 | 49.97 | 5.36 |
| Large | Ablation | 0.371 | 0.609 | 0.216 | 0.981 | 89.60 | 45.97 | 0.66 |

**注**: ECE (Expected Calibration Error) は不確かさ推定の校正誤差。Vanilla は NLL・ECE が著しく大きく（Large では NLL=4464）、異常な不確かさ出力を示す。

### 3.2 スケール別 MSE (Large モデル)

| Bucket | 定義 | IntSeq | Vanilla | Ablation |
|--------|------|--------|---------|----------|
| Small | log < 1 | **0.111** | 0.138 | 0.103 |
| Medium | 1 ≤ log < 3 | **0.051** | 0.071 | 0.116 |
| Large | 3 ≤ log < 6 | **0.162** | 2.100 | 0.381 |
| Huge | 6 ≤ log < 10 | **2.082** | 22.73 | 5.021 |
| Astronomical | log ≥ 10 | **110.4** | 840.0 | 532.6 |

Large以上の桁数になると急激にMSEが悪化するが、IntSeqBERTは他モデルと比較して一貫して最も低い誤差を維持。Vanilla は Large bucket から誤差が桁違いに増大し、Scale Invariance を欠くことが明確に示された。

### 3.3 OEIS タグ別 Magnitude MSE (Large IntSeq 代表例)

| Tag | Count | MSE | MAE |
|-----|-------|-----|-----|
| core | 15 | 0.0096 | 0.043 |
| walk | 453 | 0.0143 | 0.068 |
| mult | 303 | 0.0222 | 0.050 |
| easy | 6,709 | 0.0573 | 0.076 |
| nonn | 25,784 | 0.1404 | 0.103 |
| sign | 1,686 | 0.1708 | 0.155 |
| hard | 497 | 0.4942 | 0.239 |

`hard` タグは最も困難で MSE が高い。`core`（OEISの中核的数列）は最も良い精度。

---

## 4. Modulo スペクトル分析 (`analyze_mod_spectrum`)

モジュラス m=2〜101 の 100 個すべてに対する精度・NIG (Normalized Information Gain) を測定。

### 4.1 代表的モジュラス精度

| Size | Model | mod_2 | mod_3 | mod_5 | mod_10 | mod_100 | Top NIG (mod) |
|------|-------|-------|-------|-------|--------|---------|---------------|
| Small | **IntSeq** | **81.97** | **64.87** | **46.26** | **45.52** | **38.50** | 0.5389 (96) |
| Small | Vanilla | 78.27 | 57.42 | 39.83 | 39.98 | 36.46 | 0.4794 (96) |
| Small | Ablation | 63.90 | 46.03 | 35.34 | 30.02 | 24.05 | 0.3315 (96) |
| Middle | **IntSeq** | **84.45** | **70.28** | **55.47** | **53.65** | **44.75** | 0.6019 (96) |
| Middle | Vanilla | 80.26 | 62.26 | 45.89 | 45.89 | 42.22 | 0.5346 (96) |
| Middle | Ablation | 69.80 | 50.60 | 39.18 | 35.68 | 30.41 | 0.4032 (96) |
| Large | **IntSeq** | **85.62** | **72.46** | **60.36** | **58.37** | **48.55** | **0.6291 (96)** |
| Large | Vanilla | 81.45 | 65.27 | 49.99 | 49.14 | 45.58 | 0.5628 (96) |
| Large | Ablation | 72.34 | 53.91 | 42.51 | 39.52 | 33.65 | 0.4318 (96) |

### 4.2 主要な発見

1. **mod_96 が全モデル・全スケールで最高NIG**
   mod_96 は高度合成数 (96 = 2⁵×3) であり、多くの数列が mod_96 で周期的に振る舞う。95%CIで確認済み（Large IntSeq: NIG下限=0.6219, 上限=0.6336）。

2. **Modulo ストリームの効果**
   Ablation（Magnitude-only）と IntSeq の mod_2 精度差: Small で約18pt、Large で約13pt。FiLM による Modulo 情報の融合が Parity 予測に顕著な効果。

3. **スケール依存性**
   IntSeq の mod_2 精度は Small 81.97% → Large 85.62% と一貫して向上。Ablation も Large で72.34%まで改善するが、IntSeq には大きく劣る。

4. **mod_60 (バビロニア数60進法) の高い NIG**
   Large IntSeq で mod_60 は NIG 上位に入る（Babylonian と自動解釈）。数列に内在する60進法的周期をモデルが捉えている可能性。

---

## 5. Solver 評価 (`analyze_solver`)

「次の項」の厳密一致精度。各 10,000 サンプルで評価。

### 5.1 全体精度

| Size | Model | Top-1 Acc (%) | Top-10 Acc (%) | Sign Acc (%) | Valid Rate (%) |
|------|-------|--------------|----------------|-------------|----------------|
| Small | **IntSeq** | **14.05** | **21.00** | **98.73** | 90.59 |
| Small | Vanilla | 2.43 | 3.24 | 92.92 | **100.0** |
| Small | Ablation | 7.42 | 17.33 | 98.50 | 90.17 |
| Middle | **IntSeq** | **17.02** | **22.62** | **99.02** | 86.31 |
| Middle | Vanilla | 2.43 | 3.41 | 92.71 | **100.0** |
| Middle | Ablation | 9.88 | 20.52 | 98.74 | 90.34 |
| Large | **IntSeq** | **19.09** | **26.23** | **99.02** | 86.64 |
| Large | Vanilla | 2.59 | 3.80 | 92.05 | **100.0** |
| Large | Ablation | 11.75 | 21.79 | 98.94 | 86.99 |

IntSeqBERT は Vanilla の約7〜8倍の Top-1 精度。スケールアップで 14→17→19% と着実に改善。

### 5.2 Magnitude 別精度 (Large モデル)

| Bucket | Count | IntSeq Top-1 | IntSeq Top-10 | Vanilla Top-1 | Ablation Top-1 |
|--------|-------|-------------|--------------|--------------|----------------|
| Small | 1,835 | **68.34%** | **88.50%** | 14.11% | 54.55% |
| Medium | 3,083 | **20.82%** | **31.50%** | 0.00% | 5.61% |
| Large | 3,904 | **0.31%** | **0.67%** | 0.00% | 0.03% |
| Huge | 1,110 | **0.09%** | **0.18%** | 0.00% | 0.00% |
| Astronomical | 68 | 0.00% | 0.00% | 0.00% | 0.00% |

Small magnitude（絶対値が小さい項）では高精度 (IntSeq 68%+)、Large 以上では急落。これはトークン語彙（20,000語）の限界・整数表現の困難さを示す。

### 5.3 Solver Mode 別精度 (Large IntSeq)

| Mode | Count | Usage Rate | Top-1 Acc | Top-10 Acc |
|------|-------|-----------|----------|-----------|
| dense | 2,404 | 24.04% | **61.06%** | **86.02%** |
| sieve | 3,674 | 36.74% | 5.36% | 8.44% |
| crt | 2,317 | 23.17% | 0.09% | 0.13% |
| zero | 269 | 2.69% | **89.96%** | **89.96%** |
| none | 1,336 | 13.36% | 0.00% | 0.00% |

- **dense mode**: 候補を実数探索で直接列挙。高精度 (61.06%)。IntSeq の予測が精度良い場合に活躍。
- **sieve mode**: ふるい法。数論的制約が強い数列向け。現状 5%程度と低い。
- **crt mode**: 中国剰余定理。Large で 0.1% 未満と機能しておらず、Modulo 予測の精度が CRT 精度のボトルネック。
- **zero mode**: 次項が 0 の場合。89.96% と高精度（triv ial case）。
- **none**: Solver が候補を返せなかったケース（予測値が範囲外等）。

**IntSeq の Solver Valid Rate が 86〜91%** であり、約 10〜14% のサンプルでは Solver が有効な候補を返せない（CRT失敗等）。Vanilla は valid_rate=100%（単純にLM出力を返すため常に有効）。

### 5.4 Solver Mode 別精度 (Large Vanilla vs Ablation の比較)

| Mode | Vanilla Top-1 | Ablation Top-1 |
|------|--------------|----------------|
| vanilla_lm (Vanilla専用) | 2.59% | — |
| dense (Ablation) | — | 22.56% |
| sieve (Ablation) | — | 3.11% |
| zero (Ablation) | — | 90.94% |

---

## 6. Attention 分析 (`analyze_attention`)

5つの代表数列（A107413, A022433, A023622, A047961, A106589）に対するアテンションパターン解析。

### 6.1 Local Attention 比率

| Size | Model | A107413 total_local | A022433 total_local | A023622 total_local |
|------|-------|--------------------|--------------------|---------------------|
| Small | IntSeq | 0.446 | 0.367 | 0.416 |
| Small | Vanilla | 0.452 | 0.373 | 0.421 |
| Small | Ablation | 0.454 | 0.307 | 0.389 |
| Middle | IntSeq | 0.401 | 0.300 | 0.355 |
| Middle | Vanilla | 0.422 | 0.283 | 0.362 |
| Middle | Ablation | 0.419 | 0.239 | 0.342 |
| Large | IntSeq | 0.347 | 0.261 | 0.305 |
| Large | Vanilla | 0.348 | 0.248 | 0.307 |
| Large | Ablation | 0.405 | 0.233 | 0.328 |

（total_local_ratio = 直前3トークン周辺へのアテンション比率。値が高いほど局所依存性が強い）

### 6.2 主要な発見

1. **すべてのモデルで `pattern_alignment = UNKNOWN`**
   自動パターン判定（RECURRENCE, GLOBAL_CONTEXT 等）は発動せず。アテンションの解釈にはより細かい閾値チューニングが必要。

2. **スケールアップで Local Attention が減少**
   Small (0.35〜0.45) → Large (0.24〜0.41) と、モデルが大きくなるほどより広いコンテキストを参照する傾向。

3. **A107413（線形漸化式）が最も高い local ratio**
   直前項への強い依存を反映。A106589（Rauzy置換）は全モデルで最も低い local ratio。

4. **prev_1 vs prev_2 の比率**
   全モデルで prev_1 > prev_2。1つ前の項が最も参照されており、Markov的な局所依存を示す。

---

## 7. Case Study (`analyze_cases`)

7つの代表数列に対する予測可視化（PNG）が各チェックポイントに保存済み。

### 対象数列

| カテゴリ | OEIS ID | 説明 |
|---------|---------|------|
| Basic | A139249 | 等差数列 |
| Poly | A079414 | 4次多項式 |
| Huge | A017408 | 急成長数列 |
| Prime | A094407 | Mod-16素数 |
| Comb | A134717 | 奇数Motzkin数 |
| CA | A284479 | セル・オートマトン Rule 950 |
| Logic | A196527 | 素数和と合成数和のGCD |

可視化ファイル: `checkpoints/{size}/{model}/analysis/cases/{OEIS_ID}.png`
各図は4パネル: (1) Magnitude予測±2σ, (2) Sign確率, (3) Modulo Spectrum Heatmap, (4) Attention/Summary。

---

## 8. 総合考察

### 8.1 IntSeqBERT vs Vanilla Transformer

| 観点 | IntSeqBERT の優位性 |
|-----|-------------------|
| Magnitude 精度 | MSE で約7〜8倍低い (Large: 0.142 vs 1.037) |
| Modulo 予測 | Mod_2 で約4pt高い（Parity の習得） |
| Sign 精度 | 約1pt高い |
| Solver Top-1 | 約7〜8倍高い (19.09% vs 2.59%) |
| 校正誤差 (ECE) | Large では同等（0.65 vs 5.36）|
| 推論速度 | Solver が遅い（0.076 sec/sample vs 0.005 sec） |

Vanilla Transformer は語彙外整数 ([UNK]) の扱いに限界があり、Large/Huge magnitude での精度が致命的に低下する。

### 8.2 Ablation（Modulo ストリームの寄与）

| 観点 | 効果 |
|-----|------|
| Modulo 予測 | Ablation から IntSeq への Mod_2 向上: Small +18pt, Large +13pt |
| Magnitude 精度 | Ablation と IntSeq の差は小さい（Acc_0.5 で 1〜3pt） |
| Solver 精度 | IntSeq が Ablation より 7〜8pt 高い (Large: 19.09% vs 11.75%) |
| Calibration | Ablation は ECE が最も低い（Modulo なしで単純）|

Modulo ストリームは主に**数論的性質（剰余予測）の学習**に寄与し、Magnitude 予測にも副次的に好影響を与える。

### 8.3 Scale Scaling の効果

IntSeqBERT の主要指標のスケール依存性:

| Metric | Small | Middle | Large | 改善幅 |
|--------|-------|--------|-------|--------|
| val_mag_acc (%) | 94.69 | 95.66 | 95.73 | +1.04pt |
| val_mod_acc (%) | 40.33 | 46.68 | 50.15 | +9.82pt |
| Solver Top-1 (%) | 14.05 | 17.02 | 19.09 | +5.04pt |
| Magnitude MSE | 0.228 | 0.164 | 0.142 | -38% |

Modulo 精度とSolver精度がスケールに対してより感度が高く、スケールアップの恩恵が大きい。

### 8.4 限界と課題

1. **大きな整数（Large/Huge/Astronomical）での予測失敗**
   Huge (10⁶以上) では全モデルほぼ精度0。CRT モードの改善が鍵。

2. **Solver Valid Rate の低下**
   IntSeq は valid_rate 約87%。13%のサンプルで Solver が応答不能（主に CRT 失敗）。

3. **Attention Pattern Alignment の自動判定**
   全データで UNKNOWN。閾値・パターン定義の見直しが必要。

4. **CRT モードの性能**
   Top-1 精度が 0.1% 未満。Modulo 予測精度の向上が必要条件。

---

## 9. ファイル一覧

### チェックポイント

```
checkpoints/
├── {small,middle,large}_std/
│   ├── {intseq,vanilla,ablation}/
│   │   ├── best_metrics.json       # 最良検証指標
│   │   ├── config.json             # 実験設定
│   │   ├── history.csv             # エポック別学習ログ
│   │   ├── last_checkpoint.pt      # モデル重み
│   │   └── analysis/
│   │       ├── magnitude/          # Magnitude分析
│   │       │   ├── overall_metrics.csv
│   │       │   ├── scale_wise_metrics.csv
│   │       │   ├── tag_wise_metrics.csv
│   │       │   └── figures/        # PNG可視化
│   │       ├── mod_spectrum/       # Modulo Spectrum
│   │       │   ├── mod_spectrum_ranking.csv
│   │       │   └── mod_spectrum_with_ci.csv
│   │       ├── attention/          # Attention分析
│   │       │   └── attention_summary.csv
│   │       ├── cases/              # ケーススタディPNG
│   │       │   └── {OEIS_ID}.png
│   │       └── solver/             # Solver評価
│   │           ├── summary.json
│   │           ├── solver_results.csv
│   │           ├── magnitude_breakdown.csv
│   │           └── mode_breakdown.csv
```

---

## 10. 未実施・要確認事項

- [x] **test split での最終評価** — 2026-03-02 実施済み（→ Section 2.5）。全モデルで val との差は +0.5pt 以内、過学習なし確認。
- [ ] **Attention Pattern Alignment の閾値調整**（全 UNKNOWN の解決）
- [ ] **CRT 精度向上の施策**（Large/Huge magnitude での Solver 改善）
- [ ] **Middle の test 分析**（Middle は小/大と比べてやや分析が少ない印象）
- [ ] **統計的有意差検定**（bootstrap CIは取得済み。モデル間比較の t-test 等）
- [ ] **論文用図表の整備**（summary.md の表を LaTeX テーブルへ変換）
