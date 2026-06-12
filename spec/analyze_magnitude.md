# Implementation Specification: `src/intseq_bert/analysis/analyze_magnitude.py`

## 1. Overview

This script analyzes the model's ability to predict numeric magnitude from several angles. It measures not only raw accuracy, but also how performance changes by numeric scale and whether the predicted uncertainty is calibrated.

### Key Features

1. **Scale-wise Analysis:** Error analysis by log-scale bucket to evaluate generalization to very large numbers.
2. **Uncertainty Calibration:** Correlation analysis between predicted variance `sigma^2` and observed error/MSE.
3. **Tag-Stratified Analysis:** Comparison by sequence category such as `poly` and `exp`.
4. **Vanilla Comparison:** Limit-performance comparison against the token-based Vanilla Transformer.
5. **Error Distribution Analysis:** Identification of error distribution shape and outlier patterns.
6. **Sign-consistency report:** Placeholder report for future consistency validation between magnitude and sign predictions.

---

## 2. Dependencies

* Same library stack as `analyze_mod_spectrum.py`.
* `scipy.stats` for correlation coefficients.
* `seaborn` for distribution plots.

---

## 3. Command-Line Arguments

| Argument | Type | Default | Description |
|------|-----|-----------|------|
| `--checkpoint` | str | required | Path to the model checkpoint |
| `--split_type` | str | required | Data split type (`std`, `easy`, `all`) |
| `--split_name` | str | `"test"` | Split to evaluate (`train`/`val`/`test`) |
| `--output_dir` | str | required | Output directory for results |
| `--model_type` | str | `"intseq"` | Model type (`intseq`/`vanilla`/`ablation`) |
| `--jsonl_path` | str | `"data/oeis/data.jsonl"` | OEIS JSONL path for tag metadata |
| `--batch_size` | int | `64` | Inference batch size |
| `--seed` | int | `42` | Random seed |
| `--bootstrap_samples` | int | `1000` | Number of bootstrap samples for confidence intervals |
| `--worst_k` | int | `100` | Number of samples to extract as worst cases |
| `--device` | str | `"auto"` | Device (`cuda`, `cpu`, or `auto`) |

```bash
python -m intseq_bert.analysis.analyze_magnitude \
    --checkpoint checkpoints/intseq_std/last_checkpoint.pt \
    --split_type std \
    --split_name test \
    --output_dir results/mag_analysis \
    --model_type intseq
```

---

## 4. Analysis Metrics

### 4.1. Accuracy Metrics

| Metric | Description | Notes |
|------|------|------|
| **MSE** | Mean Squared Error | Equivalent to the loss |
| **RMSE** | Root Mean Squared Error | Error in the same unit |
| **MAE** | Mean Absolute Error | Less sensitive to outliers |
| **MedAE** | Median Absolute Error | Robust to outliers |
| **R2** | Coefficient of determination | Explanatory power; closer to 1 is better |
| **Pearson rho** | Pearson correlation coefficient | Linear correlation between ground truth and predictions |
| **Spearman rho** | Spearman rank correlation coefficient | Preservation of ordering |

### 4.2. Tolerance Accuracy

Compute the fraction of samples within a tolerance `delta`.

| Metric | Tolerance | Meaning on the original scale |
|------|----------|-------------------|
| **Acc_0.5** | `abs(y - y_pred) < 0.5` | Within about 3.16x on the log scale |
| **Acc_0.1** | `abs(y - y_pred) < 0.1` | Within about 1.25x on the log scale |
| **Acc_0.05** | `abs(y - y_pred) < 0.05` | Within about 1.12x on the log scale |

### 4.3. Uncertainty Metrics (IntSeq Only)

| Metric | Description |
|------|------|
| **Calibration Error** | How well predicted standard deviation `sigma` matches the observed residual `abs(y - y_pred)` |
| **Negative Log Likelihood (NLL)** | Gaussian negative log likelihood; lower is better |
| **Expected Calibration Error (ECE)** | Weighted average calibration error over bins |

### 4.4. Consistency Report

| Metric | Description |
|------|------|
| **Sign-Magnitude Consistency** | Placeholder report in the current implementation; actual consistency scoring requires sign predictions to be collected in this analysis path |

---

## 5. Analysis Logic

### 5.1. Scale-wise Analysis

Bucket data by ground-truth `y` on the log scale and compute MSE in each bucket.

| Bucket (Log10) | Description | Example Values |
| --- | --- | --- |
| **0 - 2** | Small | `1 ~ 100` |
| **2 - 5** | Medium | `100 ~ 100,000` |
| **5 - 20** | Large | `10^5 ~ 10^20` (64-bit integer range) |
| **20 - 50** | Huge | `10^20 ~ 10^50` |
| **50+** | Astronomical | `10^50` and above |

**Goal:** Show that the Vanilla Transformer collapses into `[UNK]` beyond the "Huge" range, while IntSeqBERT can maintain the trend.

#### Sample Counts and Warnings

The output CSV (`scale_wise_metrics.csv`) must include the following columns:

| Column | Description |
|--------|------|
| `bucket` | Bucket name, e.g. `"Small"` or `"Huge"` |
| `count` | Number of samples in the bucket |
| `mse` | Mean Squared Error |
| `mae` | Mean Absolute Error |
| `mse_ci_lower` | Lower bound of the 95% confidence interval for MSE |
| `mse_ci_upper` | Upper bound of the 95% confidence interval for MSE |

> [!WARNING]
> **Handling low sample counts:**
> - For buckets with `count < 30`, set `is_reliable` to `False` in the CSV because statistical reliability is low.
> - In `error_vs_scale.png`, display unreliable buckets using one of the following:
>   1. Light color, e.g. `alpha=0.3`.
>   2. Dotted line style.
>   3. Large error bars showing the confidence interval.
> - Emit a console warning such as `"Warning: Bucket 'Astronomical' has only N=5 samples (unreliable)"`.

### 5.2. Calibration Plot

1. Sort predictions by predicted uncertainty `sigma` and divide them into 10 bins.
2. For each bin, compute the mean `sigma` and the observed RMSE.
3. Draw a scatter plot.
   * **Ideal:** Points lie on the `y=x` line, meaning predictions with lower confidence actually have larger errors.
   * **Overconfidence:** Observed error > predicted `sigma`.
   * **Underconfidence:** Observed error < predicted `sigma`.

### 5.3. Error Distribution Analysis

Analyze the distribution of prediction error `(y - y_pred)`.

1. **Histogram:** Visualize shape, normality, skewness, and multimodality.
2. **QQ plot:** Check deviations from a normal distribution.
3. **Basic statistics:**
   - Mean, median, and standard deviation.
   - Skewness: positive values indicate a longer right tail; negative values indicate a longer left tail.
   - Kurtosis: positive values indicate a sharper peak than a normal distribution.

### 5.4. Worst-K Analysis

Extract and record the `K` samples with the largest prediction errors.

#### Output Columns

| Column | Description | Example |
|--------|------|----|
| `rank` | Rank by error magnitude | `1`, `2`, ... |
| `oeis_id` | Sequence ID | `A000045` |
| `position` | Position, 0-indexed | `10` |
| `gt_value` | Ground-truth value on the log scale | `5.678` |
| `pred_value` | Predicted value | `3.210` |
| `error` | Absolute error | `2.468` |
| `tag` | Sequence tag | `exp` |
| `context` | Neighboring values before and after the target | `"..., 1.2, 3.4, [5.7], 7.8, ..."` |

#### Generating the `context` Column

The `context` column includes a few terms before and after the target position so that failures can be inspected qualitatively.

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

    # Target value, highlighted with brackets
    parts.append(f"[{sequence[position]:.2f}]")

    # Values after target
    for i in range(position + 1, end_idx):
        parts.append(f"{sequence[i]:.2f}")

    # Trailing ellipsis if truncated
    if end_idx < len(sequence):
        parts.append("...")

    return ", ".join(parts)

# Example
# sequence = [0.0, 1.2, 3.4, 5.6, 7.8, 9.0]
# format_context(sequence, position=3, window=2)
# => "..., 1.2, 3.4, [5.6], 7.8, ..."
```

> [!TIP]
> The context makes qualitative analysis possible, for example:
> - A singular point with rapid growth, e.g. `[1.2], 5.0, 15.0, ...`.
> - A failure inside a periodic pattern, e.g. `..., 2.0, 2.0, [2.0], 2.0, ...`.
> - A sign-flip point.

**Goal:** Identify model weakness patterns by tag, scale, or local sequence structure.

### 5.5. Sign-Magnitude Consistency Report

The current implementation writes `consistency_report.csv` with `N/A - requires sign predictions`. The following check documents the intended future validation once sign predictions are collected in this analysis path.

```python
# Inconsistency patterns
inconsistent = (
    ((pred_mag > 0) & (pred_sign == SIGN_NEGATIVE)) |  # Positive magnitude but negative sign
    ((pred_mag < 0) & (pred_sign == SIGN_POSITIVE))    # Negative magnitude but positive sign
)
consistency_rate = 1.0 - inconsistent.mean()
```

### 5.6. Bootstrap Confidence Intervals

Compute 95% confidence intervals for each metric.

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

### 5.7. Growth Type Analysis

Analyze whether accuracy differs between sequences that grow exponentially, meaning approximately linear on the log scale, and all other sequences.

#### `analyze_log_linearity(sequence: List[int]) -> bool`

Helper function that determines whether an input sequence grows exponentially, i.e. linearly on the log scale.

**Logic:**
1. Take absolute values, remove zeros, and compute `log10`.
2. Fit a linear regression between indices `[0, 1, ..., L-1]` and log values.
3. Return `True` for log-linear growth if `r2 > 0.95`, with the threshold defined by `config.LOG_LINEARITY_R2_THRESHOLD`.

```python
# config.py
LOG_LINEARITY_R2_THRESHOLD = 0.95
```

#### `compute_growth_type_metrics()`

Scan the dataset and compute MSE/MAE grouped by `is_log_linear`.

**Output columns:** `growth_type`, `count`, `mse`, `mae`, `mse_ci_lower`, `mse_ci_upper`, `is_reliable`

#### `plot_growth_type_comparison()`

Bar chart comparing log-linear and non-log-linear sequences.

---

## 6. Output Files

```text
results/analysis/mag/
├── overall_metrics.csv        # Overall MSE, MAE, R2, accuracy, and CI
├── scale_wise_metrics.csv     # MSE trend by scale, including N/count columns
├── tag_wise_metrics.csv       # MSE by tag
├── calibration_data.csv       # Data for calibration plot
├── error_distribution.csv     # Error distribution statistics
├── worst_k_samples.csv        # Details for the worst K samples
├── consistency_report.csv     # Placeholder sign-magnitude consistency report
├── growth_type_metrics.csv    # MSE/MAE by growth type
├── analysis_config.json       # Run configuration
└── figures/
    ├── error_vs_scale.png     # X-axis: log(y), Y-axis: error
    ├── prediction_scatter.png # X-axis: ground truth, Y-axis: prediction, with diagonal
    ├── calibration_plot.png   # Reliability of uncertainty
    ├── error_histogram.png    # Error distribution histogram
    ├── error_qq_plot.png      # QQ plot
    └── growth_type_comparison.png  # Log-linear vs non-log-linear comparison
```

---

## 7. Implementation Notes for Vanilla Support

For the Vanilla Transformer, the magnitude head exists, but the embedding layer input constraints may prevent correct magnitude encoding for unknown or large numbers. Such numbers may also be processed as `[UNK]`.

The `VanillaWrapper.predict` method should handle the following:

* **Input side:** If a sample contains `[UNK]` tokens, its prediction is unreliable. Either exclude it from analysis or record a flag marking it as a failure case.
* **Output side:** Vanilla is expected to have a `mag_head`, so use its prediction directly.

---

## 8. Usage Examples

### 8.1. Evaluate IntSeqBERT

```bash
python -m intseq_bert.analysis.analyze_magnitude \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --split_type std \
    --split_name test \
    --output_dir results/mag_analysis/intseq \
    --model_type intseq \
    --worst_k 100
```

### 8.2. Compare Against Vanilla Transformer

```bash
python -m intseq_bert.analysis.analyze_magnitude \
    --checkpoint checkpoints/vanilla_std/best_model.pt \
    --split_type std \
    --split_name test \
    --output_dir results/mag_analysis/vanilla \
    --model_type vanilla
```

### 8.3. Compare Results

Load both `overall_metrics.csv` files and compare metric differences and confidence interval overlap to determine whether the observed difference is statistically significant.

---

## 9. Implementation Caveats

### 9.1. Definition of Magnitude

The model output `mag_mu` is trained on the same scale as `mag_labels[:, :, 0]`: `0` for zero values and `1 + log10(|x|)` for nonzero values. Evaluation compares `mag_mu` directly against this target scale. When reporting pure order-of-magnitude buckets, use `u = mag_target - 1` for nonzero values and `u = 0` for zero values.

### 9.2. Handling Zero

`log10(0)` is undefined (`-inf`). For convenience, treat it as `0.0` or a special value such as `-1.0`, and either exclude it from visualizations or mark it explicitly.

### 9.3. Plot Style

Use `seaborn` styling and ensure figures are legible enough for inclusion in the paper.
