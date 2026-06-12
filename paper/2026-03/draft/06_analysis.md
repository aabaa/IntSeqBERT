# 6. Analysis and Discussion

<!-- Target: ~2 pages (~900 words + one figure) -->

## 6.1 Ablation Study: Contribution of the Modulo Stream

To isolate the effect of the Modulo stream and FiLM fusion, we compare IntSeqBERT against the Magnitude-only ablation across all three scales. Table~\ref{tab:ablation} summarizes the Large-scale differences.

**Table 4.** IntSeq vs. ablation at the Large scale on test performance. $\Delta$ = IntSeq - Ablation.

| Metric | IntSeq | Ablation | $\Delta$ |
|-------------------|-------:|---------------:|---------:|
| Mag Acc (%) | 95.85 | 89.70 | +6.15 |
| Sign Acc (%) | 98.54 | 98.29 | +0.25 |
| MMA (%) | 50.38 | 35.22 | +15.16 |
| mod-2 Acc (%) | 85.65 | 72.13 | +13.52 |
| Solver Top-1 (%) | 19.09 | 11.75 | +7.34 |
| Magnitude MSE | 0.142 | 0.371 | -0.228 |

As intended, the Modulo stream produces its largest improvement in **modulo prediction** (MMA +15.2pt, parity +13.5pt). Its contribution to **magnitude prediction** is smaller but still consistent (Acc$_{0.5}$ +6.2pt, MSE -0.228), suggesting that periodic arithmetic structure provides a complementary inductive bias for scale regression. Knowing residues of $x_i$ modulo 2 through 101 substantially narrows the plausible magnitude patterns. The effect on **sign accuracy** is small (+0.25pt), indicating that sign is mostly recoverable from magnitude-only context.

**Solver accuracy** improves by +7.3pt with the Modulo stream. The main reason is that more accurate residue information refines scoring in Dense and Sieve modes. The direct contribution of CRT mode remains limited at present (Top-1 = 0.09%, Section 5.4).

## 6.2 Scaling Behaviour

Table~\ref{tab:scale_trend} shows how IntSeqBERT metrics change with model size.

**Table 5.** Scaling of IntSeqBERT on the test split (Small / Middle / Large).

| Metric | Small | Middle | Large | $\Delta$ (S to L) |
|-------------------|-------:|-------:|-------:|----------------:|
| Mag Acc (%) | 94.73 | 95.71 | 95.85 | +1.12 |
| MMA (%) | 40.43 | 46.88 | 50.38 | +9.95 |
| mod-2 Acc (%) | 81.97 | 84.50 | 85.65 | +3.68 |
| Solver Top-1 (%) | 14.05 | 17.02 | 19.09 | +5.04 |
| Magnitude MSE | 0.228 | 0.164 | 0.142 | -0.086 |

Magnitude accuracy improves only moderately with scale (+1.1pt from Small to Large), while modulo accuracy improves much more strongly (+10.0pt). This is consistent with the intuition that residue arithmetic is more compositional and benefits more from representational capacity. Solver Top-1 accuracy follows the modulo trend, improving steadily by +5.0pt.

Figure~\ref{fig:scaling} visualizes the same trend and contrasts the scaling slopes of modulo and magnitude metrics.

<!-- Figure 3 (paper/2026-03/figures/fig3_scaling.{pdf,png})
     \begin{figure}[t]
       \centering
       \includegraphics[width=\linewidth]{figures/fig3_scaling}
       \caption{Scaling behaviour across model sizes (Small/Middle/Large).
                Left: magnitude accuracy stays high for both IntSeqBERT and Ablation,
                with a mild +1.1pt improvement from Small to Large.
                Middle: MMA improves from 40.3\% to 50.4\% for IntSeqBERT.
                Right: Solver Top-1 accuracy improves from 14.1\% to 19.1\%, while
                Vanilla remains near 2.4\%.}
       \label{fig:scaling}
     \end{figure} -->

## 6.3 Attention Patterns

Using the Large model, we visualize attention weights for five representative sequences: A107413 (linear recurrence), A022433 (Hofstadter-type sequence), A023622 (Lucas convolution), A047961 (coordination sequence), and A106589 (Rauzy substitution).

We quantify **local attention ratio**, the amount of attention concentrated on the previous three positions:

| Sequence | IntSeq | Vanilla | Ablation |
|-----------------------------|-------:|--------:|---------------:|
| A107413 (linear recurrence) | 0.347 | 0.348 | 0.405 |
| A022433 (Hofstadter) | 0.261 | 0.248 | 0.233 |
| A023622 (Lucas convolution) | 0.305 | 0.307 | 0.328 |
| A047961 (zeolite coordination) | 0.245 | 0.237 | 0.229 |
| A106589 (Rauzy substitution) | 0.166 | 0.147 | 0.185 |

Two trends emerge. First, A107413 consistently shows high local attention, reflecting dependence on immediately preceding terms. Second, local attention varies across models for the same sequence. In particular, Ablation concentrates more locally for A107413 and A023622, suggesting that removing the Modulo stream may make the model rely more heavily on nearby references.

<!-- Attention heatmap (supplementary figure, if used)
     Source: paper/2026-03/figures/attention/
     \begin{figure}[t]
       \centering
       \includegraphics[width=0.72\linewidth]{figures/attention/A107413_aggregated}
       \caption{Layer-averaged attention heatmap for Large IntSeqBERT on A107413.
                Rows are query positions and columns are key positions. The local attention
                ratio of 0.347 measures concentration on the previous three positions.}
       \label{fig:attn_heatmap}
     \end{figure} -->

**Limitation.** Automatic pattern-alignment detection, such as checking whether attention reflects the exact recurrence coefficients of A107413, returned `UNKNOWN` for all five case-study sequences across the evaluated models. Qualitative visual alignment did not satisfy the current quantitative thresholds. Developing sharper alignment metrics is future work.

## 6.4 Limitations

**Difficulty with large integers.** Solver accuracy is almost zero for $|x| \geq 10^{20}$, covering the Huge and Astronomical buckets. Even CRT mode, which could in principle reconstruct large integers from modulo predictions, reaches only 0.09% Top-1 accuracy. The bottleneck is insufficient modulo accuracy. CRT reconstruction requires **correct** residues for several moduli; with MMA around 50%, residue errors are frequent enough to break reconstruction.

**Lower valid candidate rate.** About 13% of IntSeqBERT solver calls return no valid candidate (mode = `none`), mainly due to CRT failure. Better fallback strategies, such as relaxed residue constraints or approximate CRT, may recover part of this loss.

**Attention interpretability.** The `UNKNOWN` pattern-alignment results indicate that stronger conclusions about head specialization require more careful thresholding and evaluation.

**Dataset bias.** OEIS contains many sequences tagged `nonn`, meaning nonnegative and not all zero. The train/test distribution inherits this bias, so performance on positive integers may be higher than performance on negative integers. The high sign accuracy of 98.54% may partly reflect this distribution. Astronomical values are also rare, so metrics in that bucket are based on relatively few samples. Future evaluations should introduce stratified sampling by sign and magnitude.
