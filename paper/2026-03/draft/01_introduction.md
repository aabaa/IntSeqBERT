# 1. Introduction

<!-- Target: ~1.5 pages (~700 words) -->

The On-Line Encyclopedia of Integer Sequences (OEIS) [cite:sloane1996] is the de facto reference for integer sequences in mathematics. As of January 2026, it contains more than 391,000 sequence entries spanning combinatorics, number theory, algebra, and many other areas. Each entry links a finite integer prefix with mathematical descriptions, comments, references, and examples, making OEIS an unusually rich body of machine-readable mathematical knowledge.

The task studied in this paper is **masked sequence modelling** for integer sequences. Given a finite prefix, we randomly mask several positions and train a model to recover the masked values from the surrounding integer context. This task is a direct test of how well a model internalizes the arithmetic and combinatorial laws governing integer sequences. Next-term prediction is treated as an application-oriented special case, evaluated through a solver component. Prior work such as FACT [cite:zurich-fact] defined and evaluated six OEIS tasks, including unmasking, and established what a plain Transformer can achieve when integers are represented as token IDs. However, token-based models have intrinsic difficulty with large unseen integers and multiplicative structure.

The main difficulty is the extreme heterogeneity of integer sequences. Values range from one-digit constants to astronomical factorial and exponential sequences, often differing by dozens or hundreds of decimal digits within one corpus. At the same time, many sequences obey residue constraints that are largely independent of magnitude, such as prime-valued sequences, combinatorial parity, or periodic modular structure. A standard tokenization approach assigns each integer to a discrete vocabulary item, which is poorly suited to this setting: unseen large integers become out-of-vocabulary, arithmetic structure is hidden inside opaque token IDs, and scale generalization breaks down.

We propose **IntSeqBERT**, a Transformer encoder pretrained on masked OEIS sequence modelling. Instead of tokenizing each integer, IntSeqBERT encodes every element along two complementary axes:

- **Magnitude stream:** a continuous embedding of absolute value on the logarithmic scale, capturing growth and numeric scale.
- **Modulo stream:** sin/cos embeddings of residues modulo the 100 moduli from 2 to 101, capturing periodic and number-theoretic structure.

The two streams are fused by FiLM (Feature-wise Linear Modulation) [cite:perez2018film], allowing residue information to scale and shift the magnitude representation elementwise. Training jointly optimizes magnitude regression, sign classification, and modulo prediction for 100 moduli. To recover concrete integers from predicted magnitude, sign, and residue distributions, we use an **IntegerSolver** based on dense search, sieving, and the Chinese Remainder Theorem (CRT).

We evaluate IntSeqBERT on a dataset of 274,705 OEIS sequences and compare three model sizes (Small, Middle, and Large) against two baselines. All experiments are run on a single GeForce RTX 3070 Ti with 8 GB VRAM, so the results should be interpreted under consumer-GPU memory constraints. The main findings are:

- Large IntSeqBERT achieves **95.85%** magnitude accuracy and **50.38%** mean modulo accuracy (MMA), outperforming the Vanilla Transformer baseline by **+8.9pt** and **+4.5pt**, respectively.
- The dual-stream representation improves exact next-term reconstruction through the solver by **7.4x** in Top-1 accuracy (19.09% vs. 2.59%).
- Modulo spectrum analysis reveals a strong negative correlation between NIG (Normalized Information Gain) and Euler's totient ratio $\varphi(m)/m$ ($r=-0.851$, $p<10^{-28}$). Composite moduli efficiently capture arithmetic structure through CRT aggregation.
- Modulo prediction and solver accuracy improve more strongly with model scale than magnitude prediction, suggesting that arithmetic reasoning benefits disproportionately from increased capacity.

**Contributions.** This paper makes four contributions:

1. *IntSeqBERT*: a dual-stream Transformer architecture for integer sequences, combining magnitude and modular arithmetic features through FiLM.
2. *Multitask pretraining on OEIS*: a masked modelling objective combining magnitude regression, sign classification, and residue classification for 100 moduli.
3. *Comprehensive evaluation*: scale-wise magnitude analysis, modulo spectrum analysis, attention-pattern analysis, and solver-integrated next-term prediction.
4. *Empirical insight into arithmetic structure*: the discovery of a strong negative correlation between NIG and $\varphi(m)/m$, showing that composite moduli aggregate periodic arithmetic structure in OEIS sequences.

**Paper organization.** Section 2 reviews related work. Section 3 presents the IntSeqBERT architecture. Section 4 describes the dataset and experimental setup. Section 5 reports experimental results. Section 6 discusses ablations, scaling behaviour, attention patterns, and limitations. Section 7 concludes.
