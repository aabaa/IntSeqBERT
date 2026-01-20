# `src/intseq_bert/analysis/analyze_magnitude.py` 拡張実装指示書

## 1. 概要

本文書は `spec/analyze_magnitude.md` の**拡張**を目的とし、既存実装に不足している機能を追加するための指示を記載します。
学習済み IntSeqBERT/Vanilla モデルの「Magnitude予測ストリーム」の性能を評価し、成長タイプ別分析を追加します。

---

## 2. 実装状況確認

既存の `analyze_magnitude.py` (1115行) には以下が**実装済み**です：

| 機能 | 実装状態 | 対応関数 |
|------|----------|----------|
| Scale-wise Analysis | ✅ | `compute_scale_wise_metrics()` |
| Calibration Analysis | ✅ | `compute_calibration_data()`, `plot_calibration()` |
| Worst-K Analysis | ✅ | `extract_worst_k_samples()` |
| Tag-Stratified Analysis | ✅ | `compute_tag_stratified_metrics()` |
| Error Distribution | ✅ | `compute_error_distribution_stats()`, `plot_error_histogram()` |
| Sign-Magnitude Consistency | ✅ | `compute_sign_magnitude_consistency()` |
| Bootstrap CI | ✅ | `bootstrap_ci()` |
| 全体メトリクス (R², MSE, MAE等) | ✅ | `compute_overall_metrics()` |
| **Growth Type Analysis (Log-Linear判定)** | ❌ | **未実装** |

---

## 3. 追加実装項目

### 3.1. Growth Type Analysis (成長タイプ別分析)

数列が「指数関数的（Log-Linear）」か「それ以外」かで精度に差があるかを分析する機能を追加。

#### `analyze_log_linearity(sequence: List[int]) -> bool`

* 入力数列が「指数関数的（対数スケールで線形）」に成長しているかを判定するヘルパー関数。
* **ロジック:**
    1. 系列の絶対値を取り、0 を除外してから log10 をとる。
    2. インデックス [0, 1, ..., L-1] と対数値の間で線形回帰を行う。
    3. 決定係数 r² > 0.95 (閾値は `config.py` で定数化) なら `True` (Log-Linear) を返す。

```python
LOG_LINEARITY_R2_THRESHOLD = 0.95  # config.py に追加
```

#### `compute_growth_type_metrics()`

* データセットを走査し、`is_log_linear` でグループ化した MSE/MAE を算出。
* **出力カラム:** `growth_type`, `count`, `mse`, `mae`, `mse_ci_lower`, `mse_ci_upper`, `is_reliable`

#### `plot_growth_type_comparison()`

* Log-Linear vs Non-Log-Linear の棒グラフ比較。

---

## 4. CLI 引数 (既存との整合性確認)

既存の `parse_args()` に含まれるべき引数：

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--checkpoint` | str | 必須 | モデルチェックポイントのパス |
| `--split_type` | str | 必須 | データ分割タイプ (std, easy, all) |
| `--split_name` | str | `"test"` | 評価する分割 (train/val/test) |
| `--output_dir` | str | 必須 | 結果出力ディレクトリ |
| `--model_type` | str | `"intseq"` | **モデルタイプ (intseq/vanilla)** |
| `--batch_size` | int | `64` | 推論バッチサイズ |
| `--seed` | int | `42` | 乱数シード |

---

## 5. バケット定義 (config.MAGNITUDE_BUCKETS 準拠)

| バケット名 | LogMag 範囲 | 実数範囲 |
|------------|-------------|----------|
| **Small** | 0 - 2 | 1 〜 100 |
| **Medium** | 2 - 5 | 100 〜 100,000 |
| **Large** | 5 - 20 | 10⁵ 〜 10²⁰ |
| **Huge** | 20 - 50 | 10²⁰ 〜 10⁵⁰ |
| **Astronomical** | 50+ | 10⁵⁰ 以上 |

---

## 6. 出力ファイル (追加分)

既存の出力に加えて以下を追加：

```text
results/analysis/mag/
├── growth_type_metrics.csv   # 成長タイプ別 MSE/MAE
└── figures/
    └── growth_type_comparison.png  # Log-Linear vs Non-Log-Linear 比較
```

---

## 7. 注意点

* **Magnitude の定義:** モデル出力 `mag_mu` は `1 + log10(|x|)` として学習されている。評価時は **1を引いて純粋な log10(|x|) に戻してから** 比較・可視化を行う。
* **0 の扱い:** log10(0) は未定義（-inf）だが、便宜上 0.0 または特別な値（例: -1.0）として扱い、可視化時に除外するか明示する。
* **プロット:** スタイルには `seaborn` を使用し、論文掲載に耐えうる視認性を確保する。