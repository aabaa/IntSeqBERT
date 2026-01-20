# `src/intseq_bert/analysis/analyze_magnitude.py` 実装仕様書

## 1. 概要

本スクリプトは、モデルの **Magnitude (数値の大きさ) 予測能力** を多角的に分析する。
単なる精度評価だけでなく、数値のスケール（桁数）ごとの性能変化や、予測不確実性の妥当性（Calibration）を検証する。

### 主要機能

1. **Scale-wise Analysis:** 対数スケール（桁数）ごとの誤差分析（巨大数への汎化性能を確認）。
2. **Uncertainty Calibration:** 予測分散 `sigma^2` と実際の誤差 `MSE` の相関分析。
3. **Tag-Stratified Analysis:** 数列の種類（`poly`, `exp` 等）ごとの増大則の理解度比較。
4. **Vanilla Comparison:** Vanilla Transformer (Token base) との限界性能比較。
5. **Error Distribution Analysis:** 予測誤差の分布形状と外れ値パターンの特定。
6. **Sign Consistency Check:** Magnitude 予測と Sign 予測の整合性検証。

---

## 2. 依存関係

* `analyze_mod_spectrum.py` と同様のライブラリ構成。
* `scipy.stats` (相関係数計算用)
* `seaborn` (分布プロット用)

---

## 3. コマンドライン引数

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--checkpoint` | str | 必須 | モデルチェックポイントのパス |
| `--split_type` | str | 必須 | データ分割タイプ (std, easy, all) |
| `--split_name` | str | `"test"` | 評価する分割 (train/val/test) |
| `--output_dir` | str | 必須 | 結果出力ディレクトリ |
| `--model_type` | str | `"intseq"` | モデルタイプ (intseq/vanilla/ablation) |
| `--batch_size` | int | `64` | 推論バッチサイズ |
| `--seed` | int | `42` | 乱数シード |
| `--bootstrap_samples` | int | `1000` | Bootstrap サンプル数 (信頼区間算出用) |
| `--worst_k` | int | `100` | 最悪ケースとして抽出するサンプル数 |

```bash
python -m intseq_bert.analysis.analyze_magnitude \
    --checkpoint checkpoints/intseq_std/last_checkpoint.pt \
    --split_type std \
    --split_name test \
    --output_dir results/mag_analysis \
    --model_type intseq
```

---

## 4. 分析指標 (Metrics)

### 4.1. Accuracy Metrics

| 指標 | 説明 | 備考 |
|------|------|------|
| **MSE** | Mean Squared Error | Loss と同等 |
| **RMSE** | Root Mean Squared Error | 単位を揃えた誤差 |
| **MAE** | Mean Absolute Error | 外れ値の影響を受けにくい |
| **MedAE** | Median Absolute Error | 外れ値に対して頑健 |
| **R²** | 決定係数 | 説明変量の表現力。1に近いほど良い |
| **Pearson ρ** | ピアソン相関係数 | GT vs Pred の線形相関 |
| **Spearman ρ** | スピアマン順位相関係数 | 順序関係の保持度合い |

### 4.2. Tolerance Accuracy

許容誤差 `delta` 以内の割合を算出する。

| 指標 | 許容誤差 | 実スケールでの意味 |
|------|----------|-------------------|
| **Acc_0.5** | `abs(y - y_pred) < 0.5` | 対数スケールで約 3.16倍以内 |
| **Acc_0.1** | `abs(y - y_pred) < 0.1` | 対数スケールで約 1.25倍以内 |
| **Acc_0.05** | `abs(y - y_pred) < 0.05` | 対数スケールで約 1.12倍以内 |

### 4.3. Uncertainty Metrics (IntSeq Only)

| 指標 | 説明 |
|------|------|
| **Calibration Error** | 予測された標準偏差 `sigma` が、実際の誤差残差 `abs(y - y_pred)` とどれだけ一致しているか |
| **Negative Log Likelihood (NLL)** | ガウス分布を仮定した尤度（小さいほど良い） |
| **Expected Calibration Error (ECE)** | ビン分割での校正誤差の加重平均 |

### 4.4. Consistency Metrics

| 指標 | 説明 |
|------|------|
| **Sign-Magnitude Consistency** | Magnitude 予測 > 0 かつ Sign 予測が負（またはその逆）となる矛盾サンプルの割合 |

---

## 5. 分析ロジック

### 5.1. Scale-wise Analysis (桁数別評価)

データを正解値 `y` (log scale) の大きさに応じてバケット分けし、各バケットでの MSE を算出する。

| Bucket (Log10) | Description | Example Values |
| --- | --- | --- |
| **0 - 2** | Small | `1 ~ 100` |
| **2 - 5** | Medium | `100 ~ 100,000` |
| **5 - 20** | Large | `10^5 ~ 10^20` (64bit int 範囲) |
| **20 - 50** | Huge | `10^20 ~ 10^50` |
| **50+** | Astronomical | `10^50` 以上 |

**目的:** Vanilla Transformer は "Huge" 以降で `[UNK]` となり崩壊するが、IntSeqBERT はトレンドを維持できることを示す。

#### サンプル数の記録と警告

出力CSV (`scale_wise_metrics.csv`) には以下のカラムを必須で含める：

| カラム | 説明 |
|--------|------|
| `bucket` | バケット名 (e.g., "Small", "Huge") |
| `count` | そのバケットに含まれるサンプル数 |
| `mse` | Mean Squared Error |
| `mae` | Mean Absolute Error |
| `mse_ci_lower` | MSE の 95% 信頼区間下限 |
| `mse_ci_upper` | MSE の 95% 信頼区間上限 |

> [!WARNING]
> **サンプル数不足への対応:**
> - `count < 30` のバケットは統計的有意性が低いため、CSV出力時に `is_reliable` カラムを `False` に設定する
> - 可視化 (`error_vs_scale.png`) では、信頼性の低いバケットは以下のいずれかで表示する：
>   1. 薄い色（alpha=0.3）で表示
>   2. 点線スタイルで接続
>   3. エラーバー（信頼区間）を大きく表示
> - コンソールログに警告を出力: `"Warning: Bucket 'Astronomical' has only N=5 samples (unreliable)"`

### 5.2. Calibration Plot (信頼性評価)

1. 予測データを「予測された不確実性 `sigma`」順にソートし、10個のビンに分割する。
2. 各ビンについて、「平均 `sigma`」と「実際の RMSE (Root Mean Squared Error)」を計算する。
3. 散布図を描画する。
   * **理想:** `y=x` の直線に乗る（「自信がない」と予測した時は、実際に誤差が大きい）。
   * **過信 (Overconfidence):** 実際の誤差 > 予測 `sigma`。
   * **過小評価 (Underconfidence):** 実際の誤差 < 予測 `sigma`。

### 5.3. Error Distribution Analysis (誤差分布分析)

予測誤差 `(y - y_pred)` の分布を分析する。

1. **ヒストグラム:** 誤差分布の形状（正規性、偏り、多峰性）を可視化
2. **QQプロット:** 正規分布からの逸脱度を確認
3. **基本統計量:**
   - Mean, Median, Std
   - Skewness (歪度): 正なら右裾、負なら左裾が長い
   - Kurtosis (尖度): 正なら正規より尖っている

### 5.4. Worst-K Analysis (最悪ケース分析)

予測誤差が最も大きい K 個のサンプルを抽出・記録する。

#### 出力カラム

| カラム | 説明 | 例 |
|--------|------|----|
| `rank` | 誤差の大きさ順位 | `1`, `2`, ... |
| `oeis_id` | 数列ID | `A000045` |
| `position` | 位置 (0-indexed) | `10` |
| `gt_value` | 正解値 (log scale) | `5.678` |
| `pred_value` | 予測値 | `3.210` |
| `error` | 絶対誤差 | `2.468` |
| `tag` | 数列タグ | `exp` |
| `context` | 周辺の値（前後数項） | `"..., 1.2, 3.4, [5.7], 7.8, ..."` |

#### Context カラムの生成

`context` カラムは、対象位置の前後数項を含めることで「なぜそこで間違えたのか？」を定性的に分析可能にする。

```python
def format_context(sequence: List[float], position: int, window: int = 2) -> str:
    """
    Generate context string showing surrounding values.
    
    Args:
        sequence: The full sequence (log scale values)
        position: Target position (0-indexed)
        window: Number of items before/after to include
    
    Returns:
        Formatted string like "..., 1.2, 3.4, [5.6], 7.8, ..."
    """
    start_idx = max(0, position - window)
    end_idx = min(len(sequence), position + window + 1)
    
    parts = []
    
    # Leading ellipsis if truncated
    if start_idx > 0:
        parts.append("...")
    
    # Values before target
    for i in range(start_idx, position):
        parts.append(f"{sequence[i]:.2f}")
    
    # Target value (highlighted with brackets)
    parts.append(f"[{sequence[position]:.2f}]")
    
    # Values after target
    for i in range(position + 1, end_idx):
        parts.append(f"{sequence[i]:.2f}")
    
    # Trailing ellipsis if truncated
    if end_idx < len(sequence):
        parts.append("...")
    
    return ", ".join(parts)

# 使用例
# sequence = [0.0, 1.2, 3.4, 5.6, 7.8, 9.0]
# format_context(sequence, position=3, window=2)
# => "..., 1.2, 3.4, [5.6], 7.8, ..."
```

> [!TIP]
> Context を見ることで、以下のような定性的分析が可能：
> - 「急激に増大する特異点だった」（例: `[1.2], 5.0, 15.0, ...`）
> - 「周期的パターンの中で外れた」（例: `..., 2.0, 2.0, [2.0], 2.0, ...` なのに予測が外れた）
> - 「符号反転ポイントだった」

**目的:** モデルの弱点パターン（特定のタグ、特定のスケールなど）を特定する。

### 5.5. Sign-Magnitude Consistency Check

Magnitude Head と Sign Head の予測の整合性を検証する。

```python
# 矛盾パターン
inconsistent = (
    ((pred_mag > 0) & (pred_sign == SIGN_NEGATIVE)) |  # 正の大きさなのに負符号
    ((pred_mag < 0) & (pred_sign == SIGN_POSITIVE))    # 負の大きさなのに正符号
)
consistency_rate = 1.0 - inconsistent.mean()
```

### 5.6. Bootstrap Confidence Intervals

各指標について 95% 信頼区間を算出する。

```python
def bootstrap_ci(data, metric_fn, n_samples=1000, ci=0.95):
    """Bootstrap confidence interval estimation."""
    estimates = []
    for _ in range(n_samples):
        sample = np.random.choice(data, size=len(data), replace=True)
        estimates.append(metric_fn(sample))
    lower = np.percentile(estimates, (1 - ci) / 2 * 100)
    upper = np.percentile(estimates, (1 + ci) / 2 * 100)
    return lower, upper
```

### 5.7. Growth Type Analysis (成長タイプ別分析)

数列が「指数関数的（Log-Linear）」か「それ以外」かで精度に差があるかを分析する機能。

#### `analyze_log_linearity(sequence: List[int]) -> bool`

入力数列が「指数関数的（対数スケールで線形）」に成長しているかを判定するヘルパー関数。

**ロジック:**
1. 系列の絶対値を取り、0 を除外してから log10 をとる。
2. インデックス [0, 1, ..., L-1] と対数値の間で線形回帰を行う。
3. 決定係数 r² > 0.95 (閾値は `config.LOG_LINEARITY_R2_THRESHOLD` で定数化) なら `True` (Log-Linear) を返す。

```python
# config.py
LOG_LINEARITY_R2_THRESHOLD = 0.95
```

#### `compute_growth_type_metrics()`

データセットを走査し、`is_log_linear` でグループ化した MSE/MAE を算出。

**出力カラム:** `growth_type`, `count`, `mse`, `mae`, `mse_ci_lower`, `mse_ci_upper`, `is_reliable`

#### `plot_growth_type_comparison()`

Log-Linear vs Non-Log-Linear の棒グラフ比較。

---

## 6. 出力ファイル構成

```text
results/analysis/mag/
├── overall_metrics.csv        # 全体の MSE, MAE, R², Acc, CI
├── scale_wise_metrics.csv     # 桁数ごとの MSE 推移 (N列を含む)
├── tag_wise_metrics.csv       # タグごとの MSE
├── calibration_data.csv       # Calibration Plot 用データ
├── error_distribution.csv     # 誤差分布の統計量
├── worst_k_samples.csv        # 最悪 K サンプル詳細
├── consistency_report.csv     # Sign-Magnitude 整合性レポート
├── growth_type_metrics.csv    # 成長タイプ別 MSE/MAE
└── figures/
    ├── error_vs_scale.png     # 横軸: log(y), 縦軸: Error
    ├── prediction_scatter.png # 横軸: GT, 縦軸: Pred (対角線付き)
    ├── calibration_plot.png   # 不確実性の信頼性
    ├── error_histogram.png    # 誤差分布ヒストグラム
    ├── error_qq_plot.png      # QQプロット
    └── growth_type_comparison.png  # Log-Linear vs Non-Log-Linear 比較
```

---

## 7. 実装のポイント (Vanilla 対応)

Vanilla Transformer の場合、Magnitude Head は存在しますが、Embedding 層での入力制限により、未知語（大きな数）に対して正しく Magnitude をエンコードできていない可能性があります（または `[UNK]` トークンとして処理される）。

`VanillaWrapper` の `predict` メソッドにおいて、以下の処理が必要です：

* **入力側:** `[UNK]` トークンが含まれる場合、そのサンプルの予測は信頼できないため、分析から除外するか、あるいは「失敗例」として記録するフラグを立てる。
* **出力側:** Vanilla にも `mag_head` は実装されている（はずな）ので、そのまま予測値を取得する。

---

## 8. 使用例

### 8.1. IntSeqBERT の評価

```bash
python -m intseq_bert.analysis.analyze_magnitude \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --split_type std \
    --split_name test \
    --output_dir results/mag_analysis/intseq \
    --model_type intseq \
    --worst_k 100
```

### 8.2. Vanilla Transformer との比較

```bash
python -m intseq_bert.analysis.analyze_magnitude \
    --checkpoint checkpoints/vanilla_std/best_model.pt \
    --split_type std \
    --split_name test \
    --output_dir results/mag_analysis/vanilla \
    --model_type vanilla
```

### 8.3. 結果の比較

両方の `overall_metrics.csv` を読み込み、指標の差分と信頼区間の重複を確認することで、統計的に有意な差があるかを判断できる。

---

## 9. 実装上の注意点

### 9.1. Magnitude の定義

モデル出力 `mag_mu` は `1 + log10(|x|)` として学習されている。評価時は **1を引いて純粋な log10(|x|) に戻してから** 比較・可視化を行う。

### 9.2. 0 の扱い

log10(0) は未定義（-inf）だが、便宜上 0.0 または特別な値（例: -1.0）として扱い、可視化時に除外するか明示する。

### 9.3. プロットスタイル

スタイルには `seaborn` を使用し、論文掲載に耐えうる視認性を確保する。