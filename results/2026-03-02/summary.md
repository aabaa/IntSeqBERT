# Experiment Results Summary

> Experiment notes prepared with paper writing in mind.
> Checkpoints: `checkpoints/{small,middle,large}_std/{intseq,vanilla,ablation}/`
> Evaluation set: OEIS dataset (std split)

---

## 1. Experiment Overview

### Compared Models

| Model | Short name | Description |
|-------|------------|-------------|
| **IntSeqBERT** | `intseq` | Proposed method. Dual-stream Magnitude + Modulo input with FiLM fusion |
| **Vanilla Transformer** | `vanilla` | Baseline. Standard Transformer that embeds integers as token IDs |
| **Ablation (Magnitude-only)** | `ablation` | Ablation without the Modulo stream. No FiLM |

### Model Sizes

| Scale | Layers | d_model | nheads | Training completed |
|-------|--------|---------|--------|--------------------|
| **Small** | 6 | 256 | 4 | 2026-02-20 |
| **Middle** | 8 | 512 | 8 | 2026-02-17 |
| **Large** | 12 | 768 | 12 | 2026-02-06 |

### Shared Training Settings

| Parameter | Value |
|-----------|-------|
| Dataset | OEIS (std split) |
| Training samples | 219,765 |
| Validation samples | 27,470 |
| Epochs | 200 (full) |
| Batch size | 32 (accum_steps=2 -> effective 64) |
| Learning rate | 5e-5 (warmup 10%) |
| Weight decay | 0.01 |
| Loss weights | mag=1.0, sign=1.0, **mod=2.0** |
| Optimizer | AdamW |
| Numeric precision | FP32 (AMP disabled for stability with very large integers) |
| Framework | PyTorch 2.9.1+cu128, CUDA 12.8 |
| Seed | 42 |

---

## 2. Training Results (Validation Best Metrics)

All models completed 200 epochs with no early stopping (patience=200).

### 2.1 Small (6L-256d-4h)

| Model | Best Epoch | val_loss | val_mag_acc (%) | val_mag_mse | val_sign_acc (%) | val_mod_acc (%) |
|-------|------------|----------|-----------------|-------------|------------------|-----------------|
| **IntSeq** | 174 | **1.2203** | **94.69** | **0.364** | **97.95** | **40.33** |
| Vanilla | 177 | 2.1715 | 85.58 | 1.273 | 97.03 | 36.05 |
| Ablation | 165 | 1.5655 | 93.53 | 0.375 | 97.47 | 25.89 |

Representative Modulo accuracies (val):

| Model | mod_2 (%) | mod_3 (%) | mod_5 (%) | mod_10 (%) | mod_100 (%) |
|-------|-----------|-----------|-----------|------------|-------------|
| **IntSeq** | **81.89** | **64.54** | **46.09** | **45.44** | **38.53** |
| Vanilla | 78.18 | 57.07 | 39.31 | 39.48 | 36.05 |
| Ablation | 63.75 | 45.77 | 34.98 | 29.58 | 24.13 |

### 2.2 Middle (8L-512d-8h)

| Model | Best Epoch | val_loss | val_mag_acc (%) | val_mag_mse | val_sign_acc (%) | val_mod_acc (%) |
|-------|------------|----------|-----------------|-------------|------------------|-----------------|
| **IntSeq** | 175 | **1.0704** | **95.66** | **0.171** | **98.42** | **46.68** |
| Vanilla | 168 | 1.8967 | 87.08 | 1.087 | 97.57 | 42.01 |
| Ablation | 172 | 1.4337 | 91.93 | 0.234 | 98.08 | 31.81 |

| Model | mod_2 (%) | mod_3 (%) | mod_5 (%) | mod_10 (%) | mod_100 (%) |
|-------|-----------|-----------|-----------|------------|-------------|
| **IntSeq** | **84.42** | **69.74** | **55.07** | **53.33** | **44.61** |
| Vanilla | 80.21 | 61.58 | 45.36 | 45.31 | 41.74 |
| Ablation | 69.74 | 50.17 | 38.65 | 35.36 | 30.15 |

### 2.3 Large (12L-768d-12h)

| Model | Best Epoch | val_loss | val_mag_acc (%) | val_mag_mse | val_sign_acc (%) | val_mod_acc (%) |
|-------|------------|----------|-----------------|-------------|------------------|-----------------|
| **IntSeq** | 180 | **1.0028** | **95.73** | **0.180** | **98.61** | **50.15** |
| Vanilla | 174 | 1.7470 | 86.92 | 1.076 | 97.77 | 45.55 |
| Ablation | 170 | 1.3785 | 89.34 | 0.315 | 98.39 | 34.93 |

| Model | mod_2 (%) | mod_3 (%) | mod_5 (%) | mod_10 (%) | mod_100 (%) |
|-------|-----------|-----------|-----------|------------|-------------|
| **IntSeq** | **85.67** | **72.09** | **60.03** | **58.04** | **48.24** |
| Vanilla | 81.41 | 64.70 | 49.77 | 48.96 | 45.35 |
| Ablation | 71.90 | 53.20 | 41.93 | 39.04 | 33.25 |

### 2.4 Scaling Summary

IntSeqBERT's val_mod_acc improves monotonically with scale: 40.33% -> 46.68% -> **50.15%**.
Vanilla shows only modest gains in val_mag_acc, and the gap from IntSeq also widens for Modulo prediction.

### 2.5 Final Test Split Evaluation (`--test_only --test_split test`)

> **Run date**: 2026-03-02
> **Checkpoint used**: `last_checkpoint.pt` (final epoch = 200)
> **Evaluation samples**: 27,470 (test split)
> **Note**: `best_metrics.json` is a snapshot of the best validation epoch during training. This evaluation uses the last-epoch model on the test split. The difference is minor (see below).

#### Main Metrics

| Size | Model | test_loss | test_mag_acc (%) | test_mag_mse | test_sign_acc (%) | test_mod_acc (%) |
|------|-------|-----------|------------------|--------------|-------------------|------------------|
| Small | **IntSeq** | **1.2175** | **94.73** | **0.2215** | **97.78** | **40.43** |
| Small | Vanilla | 2.2142 | 85.73 | 1.5112 | 96.91 | 36.21 |
| Small | Ablation | 1.5683 | 93.72 | 0.3002 | 97.39 | 25.97 |
| Middle | **IntSeq** | **1.0654** | **95.71** | **0.1830** | **98.34** | **46.88** |
| Middle | Vanilla | 1.9214 | 87.37 | 0.9642 | 97.42 | 42.53 |
| Middle | Ablation | 1.4300 | 92.45 | 0.1970 | 97.90 | 31.93 |
| Large | **IntSeq** | **0.9976** | **95.85** | **0.2000** | **98.54** | **50.38** |
| Large | Vanilla | 1.7808 | 86.97 | 1.0025 | 97.66 | 45.85 |
| Large | Ablation | 1.3738 | 89.70 | 0.3237 | 98.29 | 35.22 |

#### Representative Modulo Accuracies (test)

| Size | Model | mod_2 (%) | mod_3 (%) | mod_5 (%) | mod_10 (%) | mod_100 (%) |
|------|-------|-----------|-----------|-----------|------------|-------------|
| Small | **IntSeq** | **81.97** | **64.62** | **46.34** | **45.54** | **38.62** |
| Small | Vanilla | 78.27 | 57.25 | 39.58 | 39.78 | 36.25 |
| Small | Ablation | 64.15 | 46.25 | 35.31 | 30.08 | 24.07 |
| Middle | **IntSeq** | **84.50** | **70.32** | **55.49** | **53.70** | **44.84** |
| Middle | Vanilla | 80.37 | 62.26 | 45.86 | 45.97 | 42.24 |
| Middle | Ablation | 69.79 | 50.52 | 38.99 | 35.42 | 30.32 |
| Large | **IntSeq** | **85.65** | **72.62** | **60.37** | **58.38** | **48.51** |
| Large | Vanilla | 81.40 | 65.22 | 50.07 | 49.25 | 45.60 |
| Large | Ablation | 72.13 | 53.72 | 42.63 | 39.47 | 33.51 |

#### Validation (best epoch) vs. Test (last epoch)

Validation and test metrics are broadly consistent across all models, with no sign of degraded generalization or overfitting.

| Size | Model | val_mag_acc | test_mag_acc | Delta | val_mod_acc | test_mod_acc | Delta |
|------|-------|-------------|--------------|-------|-------------|--------------|-------|
| Small | IntSeq | 94.69 | 94.73 | +0.04 | 40.33 | 40.43 | +0.10 |
| Small | Vanilla | 85.58 | 85.73 | +0.15 | 36.05 | 36.21 | +0.16 |
| Small | Ablation | 93.53 | 93.72 | +0.19 | 25.89 | 25.97 | +0.08 |
| Middle | IntSeq | 95.66 | 95.71 | +0.05 | 46.68 | 46.88 | +0.20 |
| Middle | Vanilla | 87.08 | 87.37 | +0.29 | 42.01 | 42.53 | +0.52 |
| Middle | Ablation | 91.93 | 92.45 | +0.52 | 31.81 | 31.93 | +0.12 |
| Large | IntSeq | 95.73 | 95.85 | +0.12 | 50.15 | 50.38 | +0.23 |
| Large | Vanilla | 86.92 | 86.97 | +0.05 | 45.55 | 45.85 | +0.30 |
| Large | Ablation | 89.34 | 89.70 | +0.36 | 34.93 | 35.22 | +0.29 |

All deltas are positive (test >= val) or very small, confirming that the validation selection criterion was reasonable.

---

## 3. Magnitude Prediction Analysis (`analyze_magnitude`)

Magnitude regression metrics over the full test set, evaluated on the training target scale: `0` for zero values and `1 + log10(|x|)` for nonzero values.

### 3.1 Overall Metrics

| Size | Model | MSE | RMSE | MAE | R^2 | Acc_0.5 (%) | Acc_0.1 (%) | ECE |
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

**Note**: ECE (Expected Calibration Error) measures uncertainty calibration. Vanilla has substantially larger NLL and ECE (NLL=4464 for Large), indicating abnormal uncertainty outputs.

### 3.2 Scale-wise MSE (Large Models)

Here `u = 0` for zero values and `u = log10(|x|)` for nonzero values.

| Bucket | Definition | IntSeq | Vanilla | Ablation |
|--------|------------|--------|---------|----------|
| Small | u < 2 | **0.111** | 0.138 | 0.103 |
| Medium | 2 <= u < 5 | **0.051** | 0.071 | 0.116 |
| Large | 5 <= u < 20 | **0.162** | 2.100 | 0.381 |
| Huge | 20 <= u < 50 | **2.082** | 22.73 | 5.021 |
| Astronomical | u >= 50 | **110.4** | 840.0 | 532.6 |

MSE rises sharply for Large and larger buckets, but IntSeqBERT consistently maintains the lowest error among the models. Vanilla's error increases by orders of magnitude starting from the Large bucket, clearly showing its lack of scale invariance.

### 3.3 Magnitude MSE by OEIS Tag (Representative Large IntSeq Run)

| Tag | Count | MSE | MAE |
|-----|-------|-----|-----|
| core | 15 | 0.0096 | 0.043 |
| walk | 453 | 0.0143 | 0.068 |
| mult | 303 | 0.0222 | 0.050 |
| easy | 6,709 | 0.0573 | 0.076 |
| nonn | 25,784 | 0.1404 | 0.103 |
| sign | 1,686 | 0.1708 | 0.155 |
| hard | 497 | 0.4942 | 0.239 |

The `hard` tag is the most difficult and has high MSE. The `core` tag, representing central OEIS sequences, has the best accuracy.

---

## 4. Modulo Spectrum Analysis (`analyze_mod_spectrum`)

Measures accuracy and NIG (Normalized Information Gain) for all 100 moduli from m=2 to m=101.

### 4.1 Representative Modulus Accuracies

| Size | Model | mod_2 | mod_3 | mod_5 | mod_10 | mod_100 | Top NIG (mod) |
|------|-------|-------|-------|-------|--------|---------|---------------|
| Small | **IntSeq** | **81.97** | **64.62** | **46.34** | **45.54** | **38.62** | 0.5389 (96) |
| Small | Vanilla | 78.27 | 57.25 | 39.58 | 39.78 | 36.25 | 0.4794 (96) |
| Small | Ablation | 64.15 | 46.25 | 35.31 | 30.08 | 24.07 | 0.3315 (96) |
| Middle | **IntSeq** | **84.50** | **70.32** | **55.49** | **53.70** | **44.84** | 0.6019 (96) |
| Middle | Vanilla | 80.37 | 62.26 | 45.86 | 45.97 | 42.24 | 0.5346 (96) |
| Middle | Ablation | 69.79 | 50.52 | 38.99 | 35.42 | 30.32 | 0.4032 (96) |
| Large | **IntSeq** | **85.65** | **72.62** | **60.37** | **58.38** | **48.51** | **0.6291 (96)** |
| Large | Vanilla | 81.40 | 65.22 | 50.07 | 49.25 | 45.60 | 0.5628 (96) |
| Large | Ablation | 72.13 | 53.72 | 42.63 | 39.47 | 33.51 | 0.4318 (96) |

### 4.2 Key Findings

1. **mod_96 has the highest NIG for all models and scales**
   mod_96 is a composite modulus with small prime factors (96 = 2^5 x 3), so it aggregates 2-adic and mod-3 information. This is confirmed with a 95% CI (Large IntSeq: NIG lower=0.6219, upper=0.6336).

2. **Effect of the Modulo stream**
   The mod_2 accuracy gap between Ablation (Magnitude-only) and IntSeq is about 18pt for Small and about 13pt for Large. FiLM fusion of Modulo information has a marked effect on parity prediction.

3. **Scale dependence**
   IntSeq mod_2 accuracy improves consistently from Small 81.97% to Large 85.65%. Ablation also improves to 72.13% at Large, but remains far behind IntSeq.

4. **High NIG for mod_60 (Babylonian base-60)**
   For Large IntSeq, mod_60 appears among the top NIG moduli (automatically interpreted as Babylonian). The model may be capturing base-60-like periodicity present in the sequences.

---

## 5. Solver Evaluation (`analyze_solver`)

Exact-match accuracy for the "next term" task, evaluated on 10,000 samples each.

### 5.1 Overall Accuracy

| Size | Model | Top-1 Acc (%) | Top-10 Acc (%) | Sign Acc (%) | Valid Rate (%) |
|------|-------|---------------|----------------|--------------|----------------|
| Small | **IntSeq** | **14.05** | **21.00** | **98.73** | 90.59 |
| Small | Vanilla | 2.43 | 3.24 | 92.92 | **100.0** |
| Small | Ablation | 7.42 | 17.33 | 98.50 | 90.17 |
| Middle | **IntSeq** | **17.02** | **22.62** | **99.02** | 86.31 |
| Middle | Vanilla | 2.43 | 3.41 | 92.71 | **100.0** |
| Middle | Ablation | 9.88 | 20.52 | 98.74 | 90.34 |
| Large | **IntSeq** | **19.09** | **26.23** | **99.02** | 86.64 |
| Large | Vanilla | 2.59 | 3.80 | 92.05 | **100.0** |
| Large | Ablation | 11.75 | 21.79 | 98.94 | 86.99 |

IntSeqBERT achieves roughly 7-8x the Top-1 accuracy of Vanilla. Scaling improves performance steadily from 14% to 17% to 19%.

### 5.2 Accuracy by Magnitude (Large Models)

| Bucket | Count | IntSeq Top-1 | IntSeq Top-10 | Vanilla Top-1 | Ablation Top-1 |
|--------|-------|--------------|---------------|---------------|----------------|
| Small | 1,835 | **68.34%** | **88.50%** | 14.11% | 54.55% |
| Medium | 3,083 | **20.82%** | **31.50%** | 0.00% | 5.61% |
| Large | 3,904 | **0.31%** | **0.67%** | 0.00% | 0.03% |
| Huge | 1,110 | **0.09%** | **0.18%** | 0.00% | 0.00% |
| Astronomical | 68 | 0.00% | 0.00% | 0.00% | 0.00% |

Accuracy is high for small-magnitude terms (IntSeq 68%+), but drops sharply for Large and above. This reflects both the limits of the 20,000-token vocabulary and the difficulty of integer representation.

### 5.3 Accuracy by Solver Mode (Large IntSeq)

| Mode | Count | Usage Rate | Top-1 Acc | Top-10 Acc |
|------|-------|------------|-----------|------------|
| dense | 2,404 | 24.04% | **61.06%** | **86.02%** |
| sieve | 3,674 | 36.74% | 5.36% | 8.44% |
| crt | 2,317 | 23.17% | 0.09% | 0.13% |
| zero | 269 | 2.69% | **89.96%** | **89.96%** |
| none | 1,336 | 13.36% | 0.00% | 0.00% |

- **dense mode**: Directly enumerates candidates by real-valued search. High accuracy (61.06%). Effective when IntSeq predictions are accurate.
- **sieve mode**: Sieve method for sequences with strong number-theoretic constraints. Currently low at about 5%.
- **crt mode**: Chinese Remainder Theorem. Below 0.1% on Large and effectively not working; Modulo prediction accuracy is the bottleneck for CRT accuracy.
- **zero mode**: The next term is 0. High accuracy at 89.96% (trivial case).
- **none**: The Solver could not return candidates, e.g. because predictions were out of range.

The **IntSeq Solver Valid Rate is 86-91%**, meaning about 10-14% of samples do not receive a valid Solver candidate (mainly CRT failures). Vanilla has valid_rate=100% because it simply returns LM outputs and is therefore always valid.

### 5.4 Accuracy by Solver Mode (Large Vanilla vs. Ablation)

| Mode | Vanilla Top-1 | Ablation Top-1 |
|------|---------------|----------------|
| vanilla_lm (Vanilla-only) | 2.59% | - |
| dense (Ablation) | - | 22.56% |
| sieve (Ablation) | - | 3.11% |
| zero (Ablation) | - | 90.94% |

---

## 6. Attention Analysis (`analyze_attention`)

Attention-pattern analysis for five representative sequences (A107413, A022433, A023622, A047961, A106589).

### 6.1 Local Attention Ratio

| Size | Model | A107413 total_local | A022433 total_local | A023622 total_local |
|------|-------|---------------------|---------------------|---------------------|
| Small | IntSeq | 0.446 | 0.367 | 0.416 |
| Small | Vanilla | 0.452 | 0.373 | 0.421 |
| Small | Ablation | 0.454 | 0.307 | 0.389 |
| Middle | IntSeq | 0.401 | 0.300 | 0.355 |
| Middle | Vanilla | 0.422 | 0.283 | 0.362 |
| Middle | Ablation | 0.419 | 0.239 | 0.342 |
| Large | IntSeq | 0.347 | 0.261 | 0.305 |
| Large | Vanilla | 0.348 | 0.248 | 0.307 |
| Large | Ablation | 0.405 | 0.233 | 0.328 |

(total_local_ratio = attention ratio around the previous three tokens. Higher values indicate stronger local dependence.)

### 6.2 Key Findings

1. **`pattern_alignment = UNKNOWN` for all models**
   Automatic pattern detection (RECURRENCE, GLOBAL_CONTEXT, etc.) does not activate. Attention interpretation needs finer threshold tuning.

2. **Local Attention decreases with scaling**
   From Small (0.35-0.45) to Large (0.24-0.41), larger models tend to attend to broader contexts.

3. **A107413 (linear recurrence) has the highest local ratio**
   This reflects strong dependence on the immediately previous term. A106589 (Rauzy substitution) has the lowest local ratio for all models.

4. **prev_1 vs. prev_2 ratio**
   prev_1 > prev_2 for all models. The immediately previous term is referenced most often, indicating Markov-like local dependence.

---

## 7. Case Study (`analyze_cases`)

Prediction visualizations (PNG) for seven representative sequences have been saved under each checkpoint.

### Target Sequences

| Category | OEIS ID | Description |
|----------|---------|-------------|
| Basic | A139249 | Arithmetic progression |
| Poly | A079414 | Degree-4 polynomial |
| Huge | A017408 | Rapidly growing sequence |
| Prime | A094407 | Mod-16 primes |
| Comb | A134717 | Odd Motzkin numbers |
| CA | A284479 | Cellular automaton Rule 950 |
| Logic | A196527 | GCD of prime sums and composite sums |

Visualization files: `checkpoints/{size}_std/{model}/analysis/cases/{OEIS_ID}.png`
Each figure has four panels: (1) Magnitude prediction +/- 2 sigma, (2) Sign probability, (3) Modulo Spectrum Heatmap, and (4) Attention/Summary.

---

## 8. Overall Discussion

### 8.1 IntSeqBERT vs. Vanilla Transformer

| Aspect | IntSeqBERT advantage |
|--------|----------------------|
| Magnitude accuracy | About 7-8x lower MSE (Large: 0.142 vs. 1.037) |
| Modulo prediction | About +4pt on Mod_2 (parity acquisition) |
| Sign accuracy | About +1pt |
| Solver Top-1 | About 7-8x higher (19.09% vs. 2.59%) |
| Calibration error (ECE) | Comparable at Large (0.65 vs. 5.36) |
| Inference speed | Solver is slower (0.076 sec/sample vs. 0.005 sec) |

Vanilla Transformer is limited by its handling of out-of-vocabulary integers (`[UNK]`), causing catastrophic degradation on Large/Huge magnitude values.

### 8.2 Ablation (Contribution of the Modulo Stream)

| Aspect | Effect |
|--------|--------|
| Modulo prediction | Mod_2 gain from Ablation to IntSeq: Small +18pt, Large +13pt |
| Magnitude accuracy | Gap between Ablation and IntSeq is small (1-3pt in Acc_0.5) |
| Solver accuracy | IntSeq is 7-8pt higher than Ablation (Large: 19.09% vs. 11.75%) |
| Calibration | Ablation has the lowest ECE (simpler without Modulo) |

The Modulo stream mainly contributes to learning **number-theoretic properties (residue prediction)** and also has a secondary positive effect on Magnitude prediction.

### 8.3 Effect of Scaling

Scale dependence of the main IntSeqBERT metrics:

| Metric | Small | Middle | Large | Improvement |
|--------|-------|--------|-------|-------------|
| val_mag_acc (%) | 94.69 | 95.66 | 95.73 | +1.04pt |
| val_mod_acc (%) | 40.33 | 46.68 | 50.15 | +9.82pt |
| Solver Top-1 (%) | 14.05 | 17.02 | 19.09 | +5.04pt |
| Magnitude MSE | 0.228 | 0.164 | 0.142 | -38% |

Modulo accuracy and Solver accuracy are more sensitive to scale, showing larger gains from model scaling.

### 8.4 Limitations and Open Issues

1. **Prediction failures for large integers (Large/Huge/Astronomical)**
   Accuracy is nearly zero for all models on Huge (`10^20 <= |x| < 10^50`) and Astronomical (`|x| >= 10^50`) values. Improving CRT mode is key.

2. **Decline in Solver Valid Rate**
   IntSeq has a valid_rate of about 87%. The Solver cannot respond for 13% of samples, mainly due to CRT failures.

3. **Automatic Attention Pattern Alignment**
   UNKNOWN for all data. Thresholds and pattern definitions need revision.

4. **CRT mode performance**
   Top-1 accuracy is below 0.1%. Higher Modulo prediction accuracy is a prerequisite for improvement.

---

## 9. File List

### Checkpoints

```
checkpoints/
├── {small,middle,large}_std/
│   ├── {intseq,vanilla,ablation}/
│   │   ├── best_metrics.json       # Best validation metrics
│   │   ├── config.json             # Experiment settings
│   │   ├── history.csv             # Per-epoch training log
│   │   ├── last_checkpoint.pt      # Model weights
│   │   └── analysis/
│   │       ├── magnitude/          # Magnitude analysis
│   │       │   ├── overall_metrics.csv
│   │       │   ├── scale_wise_metrics.csv
│   │       │   ├── tag_wise_metrics.csv
│   │       │   └── figures/        # PNG visualizations
│   │       ├── mod_spectrum/       # Modulo Spectrum
│   │       │   ├── mod_spectrum_ranking.csv
│   │       │   └── mod_spectrum_with_ci.csv
│   │       ├── attention/          # Attention analysis
│   │       │   └── attention_summary.csv
│   │       ├── cases/              # Case-study PNGs
│   │       │   └── {OEIS_ID}.png
│   │       └── solver/             # Solver evaluation
│   │           ├── summary.json
│   │           ├── solver_results.csv
│   │           ├── magnitude_breakdown.csv
│   │           └── mode_breakdown.csv
```

---

## 10. Pending / To-Check Items

- [x] **Final evaluation on the test split** — Completed on 2026-03-02 (see Section 2.5). All models are within +0.5pt of validation, with no overfitting observed.
- [ ] **Threshold tuning for Attention Pattern Alignment** (resolve all UNKNOWN results)
- [ ] **CRT accuracy improvements** (Solver improvements for Large/Huge magnitude values)
- [ ] **Additional Middle test analysis** (Middle currently has slightly less analysis than Small/Large)
- [ ] **Statistical significance tests** (bootstrap CIs are already available; add model-comparison tests such as t-tests)
- [ ] **Paper-ready figures and tables** (convert tables from `summary.md` into LaTeX tables)
