# 7. Conclusion

<!-- Target: ~0.5 pages (~250 words) -->

We proposed **IntSeqBERT**, a dual-stream Transformer encoder for integer sequences. Each element is represented along two complementary axes: a continuous logarithmic Magnitude embedding that captures scale and growth, and a sin/cos Modulo embedding that captures periodic arithmetic structure. The two streams are fused with FiLM and jointly trained on 219,765 OEIS sequences with a multitask objective combining magnitude regression, sign classification, and residue classification for 100 moduli.

On the test set, Large IntSeqBERT achieves 95.85% magnitude accuracy, 98.54% sign accuracy, and 50.38% mean modulo accuracy, outperforming the tokenized Vanilla Transformer by +8.9pt, +0.9pt, and +4.5pt. In solver-based next-term prediction, IntSeqBERT reaches 19.09% Top-1 exact-match accuracy, 7.4x the Vanilla baseline. Ablation confirms that the Modulo stream accounts for most of the modulo-prediction improvement (+15.2pt) and also provides a secondary gain for magnitude regression (+6.2pt). Across moduli, we observe a strong negative correlation between NIG and Euler's totient ratio $\varphi(m)/m$ ($r=-0.851$, $p<10^{-28}$), providing empirical evidence that composite moduli efficiently capture OEIS arithmetic structure through CRT aggregation.

Future work includes:

- Improving large-integer prediction by combining magnitude uncertainty with approximate CRT.
- Extending the Modulo stream to primes larger than 101 and to algebraic structures beyond $\mathbb{Z}/m\mathbb{Z}$.
- Developing finer attention-alignment metrics to identify which recurrence structures are captured by individual heads.
- Building family-aware splits, since random splits cannot prevent leakage across related sequence families.
- Improving solver evaluation by comparing next-term prediction with masked prediction at interior positions, where bidirectional context can be used more fully.
- Adding synthetic sequence augmentation, as in FACT [cite:zurich-fact], from recurrence templates and algebraic rules to improve generalization in sparse buckets such as Huge and Astronomical.
- Scaling to larger models and larger pretraining corpora. This study is limited to the Large model on one GPU; larger compute may especially improve modulo accuracy and CRT-mode reconstruction.
