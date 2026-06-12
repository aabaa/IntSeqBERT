# 5. Experiments

<!-- Target: ~4 pages (~1800 words + Table 1 + Table 2 + Figure 2) -->

## 5.1 Main Results

Table~\ref{tab:main} shows test performance for all three model sizes and all variants. IntSeqBERT consistently outperforms both baselines across scales and metrics.

<!-- Table 1 -->
**Table 1.** Test results. Mag Acc = Accuracy$_{0.5}$ (%), Sign Acc (%), and MMA = Mean Modulo Accuracy (%). The best value within each size group is shown in **bold**.

| Size | Model | Mag Acc | Sign Acc | MMA |
|--------|------------|--------:|---------:|-------:|
| Small  | **IntSeq** | **94.73** | **97.78** | **40.43** |
| Small  | Vanilla    | 85.73   | 96.91    | 36.21  |
| Small  | Ablation   | 93.72   | 97.39    | 25.97  |
| Middle | **IntSeq** | **95.71** | **98.34** | **46.88** |
| Middle | Vanilla    | 87.37   | 97.42    | 42.53  |
| Middle | Ablation   | 92.45   | 97.90    | 31.93  |
| Large  | **IntSeq** | **95.85** | **98.54** | **50.38** |
| Large  | Vanilla    | 86.97   | 97.66    | 45.85  |
| Large  | Ablation   | 89.70   | 98.29    | 35.22  |

At the Large scale, IntSeqBERT improves over Vanilla by **+8.9pt** in Mag Acc and **+4.5pt** in MMA. Removing the Modulo stream causes the largest MMA drop, reaching -15.2pt for Large, directly quantifying the contribution of periodic arithmetic features. The ablation still retains competitive Mag Acc, suggesting that sign and modulo information are essential for modulo prediction but have a more limited effect on magnitude regression.

**Learning curves.** Figure~\ref{fig:learning_curve} shows validation-loss curves for the Large variants. Large IntSeqBERT decreases steadily from Val Loss = 2.17 at epoch 1 to 1.70 at epoch 10, 1.15 at epoch 50, 1.05 at epoch 100, and 1.01 at epoch 200. Training and validation losses track each other closely (epoch 200: Train 1.00 / Val 1.01), and we observe no overfitting. Vanilla ends at Val Loss = 1.77, higher than both IntSeqBERT (1.01) and Ablation (1.39), indicating that the presence of the Modulo stream materially affects optimization even at the same model size.

<!-- Figure 4 (paper/2026-03/figures/fig4_learning_curves.{pdf,png})
     \begin{figure}[t]
       \centering
       \includegraphics[width=\linewidth]{figures/fig4_learning_curves}
       \caption{Validation-loss learning curves for all scales and variants.
                IntSeqBERT (solid blue) consistently stays below Vanilla (dashed orange)
                and Ablation (dash-dot green), converging to Val Loss = 1.01 at epoch 200
                for the Large scale. See supplementary Figure~\ref{fig:train_val_curve}
                for training vs. validation losses.}
       \label{fig:learning_curve}
     \end{figure}

     Supplementary Figure 4b (paper/2026-03/figures/fig4b_train_val_curves.{pdf,png})
     \begin{figure}[t]
       \centering
       \includegraphics[width=\linewidth]{figures/fig4b_train_val_curves}
       \caption{Training loss (dotted) and validation loss (solid). Each panel is a model
                variant; colors indicate Small (green), Middle (orange), and Large (blue).
                Train and validation curves remain close across all variants and scales,
                showing no visible overfitting.}
       \label{fig:train_val_curve}
     \end{figure} -->

## 5.2 Magnitude Prediction

**Scale-wise analysis.** Table~\ref{tab:scale} reports magnitude-bucket MSE on the test set for Large models. Buckets are defined by $u=\log_{10}(|x|)$, with $u=0$ for $x=0$.

<!-- Table 2 -->
**Table 2.** Scale-wise MSE on the test set (Large models). Lower is better.

| Bucket | IntSeq | Vanilla | Ablation |
|----------------|--------:|---------:|---------:|
| Small ($u<2$) | 0.111 | 0.138 | **0.103** |
| Medium ($2\le u<5$) | **0.051** | 0.071 | 0.116 |
| Large ($5\le u<20$) | **0.162** | 2.100 | 0.381 |
| Huge ($20\le u<50$) | **2.082** | 22.73 | 5.021 |
| Astronomical ($u\ge50$) | **110.4** | 840.0 | 532.6 |

<!-- Note: scale-wise data from checkpoints/large_std/{model}/analysis/magnitude/scale_wise_metrics.csv. -->

Vanilla collapses in the Large bucket (MSE = 2.10, 13x IntSeq), mainly because unseen integers are absorbed into the `UNK` token. IntSeqBERT obtains the best MSE in every bucket except Small, and its advantage is clearest for Medium and above. The ablation degrades in the Medium bucket due to the absence of modulo context (0.116 vs. 0.051 for IntSeq). It also degrades substantially for Huge and Astronomical values (Huge: 5.02 vs. 2.08; Astronomical: 533 vs. 110). This suggests that while magnitude can be estimated reasonably for small integers without modular information, FiLM modulation from the Modulo stream acts as an arithmetic structural constraint for very large values.

Figure~\ref{fig:scatter} shows predicted versus true magnitude for the three Large variants. IntSeqBERT attains the highest linear agreement ($R^2=0.988$), and its deviations from the diagonal in the Large and Huge buckets are much smaller than Vanilla ($R^2=0.943$). This visually confirms the scale-wise MSE results and the failure mode of tokenized baselines in high-scale regions.

<!-- Figure 5 (paper/2026-03/figures/fig5_magnitude_scatter.{pdf,png})
     \begin{figure}[t]
       \centering
       \includegraphics[width=\linewidth]{figures/fig5_magnitude_scatter}
       \caption{Predicted versus true magnitude for Large models on the $\log_{10}$ scale.
                Points are colored by bucket. IntSeqBERT reaches $R^2=0.988$ and Vanilla
                reaches $R^2=0.943$, with clear dispersion for Vanilla above the Large bucket.}
       \label{fig:scatter}
     \end{figure} -->

**Calibration.** Figure~\ref{fig:calibration} shows uncertainty calibration curves for Large models. The x-axis is predicted uncertainty $\sigma$ averaged within a bin, and the y-axis is the observed RMSE in that bin. Perfect calibration lies on $y=x$. Vanilla has an extremely wide $\sigma$ range (0.007 to 46.7) and a large ECE of 5.36. It is overconfident in low-scale regions, where $\sigma \approx 0$ but RMSE is 1 to 2, while high-scale uncertainty can explode. IntSeqBERT has a much better ECE of 0.65, comparable to Ablation (0.66). This suggests that uncertainty calibration depends more strongly on the continuous numeric representation than on the presence of the Modulo stream alone.

<!-- Figure 6 (paper/2026-03/figures/fig6_calibration.{pdf,png})
     \begin{figure}[t]
       \centering
       \includegraphics[width=\linewidth]{figures/fig6_calibration}
       \caption{Uncertainty calibration curves for Large models. The x-axis is predicted
                $\sigma$ on a log scale; the y-axis is observed RMSE. IntSeqBERT (ECE = 0.648)
                and Ablation (ECE = 0.662) stay close to the diagonal, while Vanilla
                (ECE = 5.360) shows severe overconfidence at low $\sigma$.}
       \label{fig:calibration}
     \end{figure} -->

## 5.3 Modulo Spectrum Analysis

We evaluate Normalized Information Gain (NIG) for each modulus $m \in \{2,\ldots,101\}$ using Large IntSeqBERT.

Figure~\ref{fig:nig_spectrum} shows the NIG spectrum for the three Large variants. IntSeqBERT (solid blue) outperforms Vanilla (dashed orange) and Ablation (dash-dot green) across the full range. The contrast between prime moduli (gray background) and composite moduli is also visually clear.

<!-- Figure 2 (paper/2026-03/figures/fig2_nig_spectrum.{pdf,png})
     \begin{figure}[t]
       \centering
       \includegraphics[width=\linewidth]{figures/fig2_nig_spectrum}
       \caption{NIG spectrum for moduli $m=2,\ldots,101$ (Large models).
                Gray background marks prime moduli. The light-blue band is the 95\% CI
                for IntSeqBERT from bootstrapping.}
       \label{fig:nig_spectrum}
     \end{figure} -->

**Finding 1: NIG is strongly negatively correlated with Euler's totient ratio $\varphi(m)/m$.** We observe Pearson $r=-0.851$ ($p<10^{-28}$) between NIG and $\varphi(m)/m=\prod_{p\mid m}(1-1/p)$ (Figure~\ref{fig:nig_phi}). Composite moduli with many small prime factors tend to have higher NIG. This can be interpreted as a CRT aggregation effect: if $m$ is a common multiple of smaller moduli $m_1,m_2,\ldots$, then $x \bmod m$ jointly preserves information for those moduli. The highest NIG across all models and scales is achieved by $m=96=2^5\times3$ ($\varphi(96)/96=1/3$, Large IntSeq NIG = 0.629, 95% CI [0.622, 0.634]). An exception is $m=2$, which reaches NIG = 0.628 despite being prime; this reflects the corpus-wide importance of parity in OEIS.

<!-- Figure 2b (paper/2026-03/figures/fig2b_nig_vs_phi.{pdf,png})
     \begin{figure}[t]
       \centering
       \includegraphics[width=0.72\linewidth]{figures/fig2b_nig_vs_phi}
       \caption{NIG versus Euler's totient ratio $\varphi(m)/m$ for Large IntSeqBERT.
                Composite moduli are blue circles and prime moduli are red triangles.
                The regression line shows Pearson $r=-0.851$ ($p<10^{-28}$).}
       \label{fig:nig_phi}
     \end{figure} -->

**Finding 2: Parity accuracy stratifies models.** At the Large scale, mod-2 accuracy is 85.65% for IntSeq, 81.40% for Vanilla, and 72.13% for Ablation. Removing the Modulo stream causes a 13.5pt drop, making parity the clearest single-modulus indicator of the stream's effect.

**Representative modulus accuracies (Large models):**

| Modulus | IntSeq | Vanilla | Ablation | Interpretation |
|------:|-------:|--------:|---------:|----------------------|
| 2     | 85.65  | 81.40   | 72.13    | Parity |
| 3     | 72.62  | 65.22   | 53.72    | Ternary residue |
| 5     | 60.37  | 50.07   | 42.63    | Last digit modulo 5 |
| 10    | 58.38  | 49.25   | 39.47    | Last decimal digit |
| 60    | 53.97  | 47.87   | 35.12    | Sexagesimal, highly composite |
| 96    | 51.82  | 47.29   | 34.44    | Composite (2^5 x 3) |
| 100   | 48.51  | 45.60   | 33.51    | Last two decimal digits |

## 5.4 Next-Term Prediction with the Solver

The solver reconstructs candidate integers from the model's magnitude, sign, and modulo predictions and ranks them by likelihood. We evaluate exact-match accuracy on 10,000 test samples. Each sample corresponds to one OEIS sequence; the **last term** is used as the prediction target, and all preceding terms are input context. Test sequences have a minimum length of 10 and a median length of 36 (mean 42.5), so the solver always receives at least 9 preceding terms.

The **valid rate** is the fraction of samples for which the solver returns at least one candidate integer. Vanilla always returns candidates from the token-vocabulary softmax, so its valid rate is 100%. IntSeq and Ablation can fail to produce a valid candidate within the search range, yielding a lower valid rate.

**Table 3.** Solver evaluation: Top-1 and Top-10 exact-match accuracy (%) and valid candidate rate.

| Size | Model | Top-1 | Top-10 | Sign Acc | Valid Rate (%) |
|--------|------------|-------:|-------:|---------:|------------:|
| Small  | **IntSeq** | **14.05** | **21.00** | **98.73** | 90.59 |
| Small  | Vanilla    | 2.43   | 3.24   | 92.92    | 100.0 |
| Small  | Ablation   | 7.42   | 17.33  | 98.50    | 90.17 |
| Middle | **IntSeq** | **17.02** | **22.62** | **99.02** | 86.31 |
| Middle | Vanilla    | 2.43   | 3.41   | 92.71    | 100.0 |
| Middle | Ablation   | 9.88   | 20.52  | 98.74    | 90.34 |
| Large  | **IntSeq** | **19.09** | **26.23** | **99.02** | 86.64 |
| Large  | Vanilla    | 2.59   | 3.80   | 92.05    | 100.0 |
| Large  | Ablation   | 11.75  | 21.79  | 98.94    | 86.99 |

Large IntSeqBERT reaches 19.09% Top-1 accuracy, **7.4x** the Vanilla baseline (2.59%).

**Accuracy by magnitude (Large IntSeqBERT):**

| Bucket | Top-1 | Top-10 |
|--------------|-------:|-------:|
| Small        | 68.34  | 88.50  |
| Medium       | 20.82  | 31.50  |
| Large        | 0.31   | 0.67   |
| Huge         | 0.09   | 0.18   |
| Astronomical | 0.00   | 0.00   |

Solver accuracy collapses for large integers, consistent with the Magnitude MSE results in Section 5.2. In the Small bucket, however, Top-1 accuracy reaches 68.34%, showing that next-term reconstruction is practical for low-scale values.

**Mode breakdown (Large IntSeqBERT):** The solver switches among dense, sieve, and CRT modes according to search width $\Delta n = n_{\max}-n_{\min}$ (Section 3.9). It also uses zero and none as terminal states:

| Mode | Usage | Top-1 | Description |
|--------|--------:|------:|-----------------------------------|
| dense  | 24.0% | 61.06 | Direct enumeration over an integer range |
| sieve  | 36.7% | 5.36  | Enumeration via modular sieving |
| crt    | 23.2% | 0.09  | Reconstruction with CRT |
| zero   | 2.7%  | 89.96 | Predicting the next term as 0 |
| none   | 13.4% | 0.00  | No valid candidate found |

The CRT mode, which attempts to reconstruct integers directly from modulo predictions, currently achieves almost zero accuracy. We discuss this limitation in Section 6.4.

**Sieve mode details.** Sieve mode greedily selects high-confidence moduli as anchors and adds them until their least common multiple keeps the search below the target candidate count of 100,000. It combines top-probability residues by beam search, generates candidates through CRT, and keeps the top-$k$ after scoring.
