# 2. Related Work

<!-- Target: ~1.5 pages (~700 words) -->

## 2.1 AI and Mathematical Discovery

### Mathematical Conjectures from Sequence Similarity

History contains many examples in which observing similarities among coefficients or numerical sequences led to major mathematical conjectures.

McKay (1978) noticed that the coefficient 196884 in the expansion
$j(\tau) = q^{-1} + 744 + 196884q + \cdots$
differs by one from 196883, the dimension of the smallest nontrivial irreducible representation of the Monster group. Conway and Norton formulated the **Monstrous Moonshine conjecture** from this observation [cite:conway1979monstrous], and Borcherds' proof in 1992 led to a Fields Medal.

The **Taniyama-Shimura conjecture** (1955) arose from structural similarities between the L-functions of elliptic curves and modular forms at the coefficient level. Wiles' 1994 proof for semistable elliptic curves resolved Fermat's Last Theorem [cite:wiles1995]. The conjecture is now viewed as a special GL(2) case of the Langlands program (1967), whose core is a deep analogy between Galois representations and automorphic forms.

The **Birch and Swinnerton-Dyer conjecture** emerged directly from numerical experimentation in the 1960s. Using the EDSAC-2 computer, Birch and Swinnerton-Dyer observed patterns of the form $\prod_{p \leq N} N_p/p \sim C(\log N)^r$ and conjectured their connection to the rank $r$ of an elliptic curve [cite:birch1965]. This is a classical example of inductive mathematical reasoning from numerical sequences.

In mathematical physics, Candelas et al. (1991) used mirror symmetry from string theory to predict the number of degree-$d$ rational curves on a quintic threefold. Extending the known degree-1 and degree-2 values, they predicted the degree-3 value **317,206,375**, later confirmed by Givental's mathematical proof [cite:candelas1991].

These examples illustrate that observing numerical patterns and sequence similarities can initiate mathematical discovery. If AI can automate this process at scale, it may help identify candidates for new mathematical conjectures.

### AI Progress in Mathematics

AI is increasingly active across mathematical domains. In problem solving and proof, DeepMind's AlphaGeometry [cite:trinh2024alphageometry] solved IMO geometry problems at high accuracy, while AlphaProof [cite:alphaproof2024] solved four of six IMO 2024 problems as Lean formal proofs. In formalization, autoformalization research led by Urban and others [cite:urban-autoformalization] has made progress toward automatically converting natural-language mathematics into machine-checkable formal statements.

For automated conjecture generation from numerical patterns, the **Ramanujan Machine** [cite:raayoni2021ramanujan] generated conjectures about continued fractions for constants such as $\pi$ and $e$ by numerical search alone, and some were later proved. **FunSearch** [cite:romera2023funsearch] combined large language models with evolutionary search and discovered improved constructions for the cap set problem.

These systems show that AI-assisted mathematical discovery is viable, but automatic conjecture generation remains limited. The Ramanujan Machine focuses on continued fractions for specific constants rather than arbitrary integer sequences. FunSearch is effective for combinatorial problems with explicit optimization objectives, but does not target conjecture generation across diverse mathematical areas. Large-scale representation learning over integer sequences remains underexplored, and this paper aims to provide such a foundation.

## 2.2 AI Research on OEIS

One line of OEIS-based AI research is **Alien Coding** [cite:urban-alien-coding] by Urban and collaborators. It seeks algorithms or programs that generate OEIS sequences, searching for short code descriptions of sequences. This views a sequence as the result of computation and studies integer sequences from the perspective of program synthesis. Follow-up work includes **QSynt** [cite:gauthier2023qsynt], which used self-learning tree search to synthesize programs for 43,516 OEIS sequences, and **Learning Conjecturing from Scratch** [cite:gauthier2025conjecturing], which generated inductive conjectures for OEIS-derived arithmetic problems and automatically proved 5,565 problems.

**FACT** [cite:zurich-fact] is a comprehensive OEIS benchmark. It defines six tasks: classification, similarity, sequence-part prediction, next-element extrapolation, **unmasking**, and continuation. FACT evaluates MLP, RNN, CNN, and Transformer baselines. One of its key findings is that unmasking is the hardest of the six tasks, especially for organic OEIS sequences compared with synthetic data. Since unmasking requires recovering arbitrary positions from full context, it demands deeper understanding of sequence rules. A model that solves unmasking well is likely to be useful for classification and similarity estimation as well.

This work directly targets the unmasking task. Our Vanilla baseline follows FACT's plain tokenized-Transformer design at the architectural level, while using a smaller vocabulary under our 8 GB VRAM constraint. IntSeqBERT adds modulo feature engineering and FiLM fusion to push the performance boundary under limited computational resources.

The relation to prior work is as follows. Alien Coding and its successors aim to explicitly describe individual sequence-generation rules by program synthesis. FACT and this work are based on **representation learning**. FACT asks what a plain Transformer can achieve; this work asks how far arithmetic feature engineering can improve it. Instead of deriving an explicit formula for each sequence, IntSeqBERT injects shared arithmetic and periodic structure through a modulo spectrum.

## 2.3 Arithmetic Feature Engineering with Modulo Spectra

Deep neural networks tend to learn low-frequency, nearly linear rules more easily than high-frequency nonlinear rules. This phenomenon is known as spectral bias [cite:rahaman2019spectral]. Networks often learn smooth components first, while sharp changes and multiplicative structure may require greater depth.

This issue is especially relevant for integer sequences. Learning multiplicative growth such as $n^2$ or $n!$ as a pure sequence model may require a deep network. Our goal is to mitigate this by feature engineering. Since multiplication naturally manifests in residues, we explicitly provide a modulo spectrum as input features, reducing the burden on the model to rediscover multiplicative structure from scratch.

### Algebraic Compatibility of Integer Operations and Residues

Many OEIS sequences are generated by finite algorithms combining addition, subtraction, multiplication, and modular operations. Fibonacci numbers, binomial coefficients, and prime sieves are typical examples.

Residue maps are ring homomorphisms for integer addition and multiplication:
$$
(a + b) \bmod m = ((a \bmod m) + (b \bmod m)) \bmod m,
$$
$$
(a \times b) \bmod m = ((a \bmod m) \times (b \bmod m)) \bmod m.
$$
Therefore, the residue sequence $(x_i \bmod m)$ is a compact representation that preserves additive and multiplicative structure, even when the original integers are extremely large.

### Power Sequences and Power Residues

Residues are especially informative for power-like growth such as $a^n$, $n!$, and $\binom{2n}{n}$. For prime $p$ and $\gcd(a,p)=1$, the residues of $a^n$ modulo $p$ form a periodic sequence by Fermat's little theorem. More refined information is provided by reciprocity laws for power residues:

- **Quadratic residues:** Gauss' quadratic reciprocity law [cite:gauss-reciprocity] relates residue behaviour across odd primes.
- **Cubic and quartic residues:** analogous reciprocity laws apply when $p \equiv 1 \pmod{3}$ or $p \equiv 1 \pmod{4}$.
- **Higher power residues:** Kummer and Artin reciprocity generalize these ideas to broader settings.

Although these laws are clearest for prime moduli, composite moduli also preserve factor-wise information through CRT. This motivates using all moduli from 2 to 101, including both primes and composites.

### Composite Moduli and CRT

Residues modulo a composite number preserve information about prime-power components through the Chinese Remainder Theorem. For example, $m=96=2^5 \times 3$ jointly encodes 2-adic information up to $2^5$ and mod-3 information. In our modulo spectrum analysis, $m=96$ obtains the highest Normalized Information Gain (NIG), suggesting that combined 2-adic and mod-3 periodicity is prominent in OEIS.

### Decimal-Representation Sequences

OEIS also contains many sequences defined by decimal representation, such as digit sums, digit products, and palindromes. Moduli such as 10 and 100 directly encode the last one or two decimal digits. Including $m=10$ and $m=100$ allows IntSeqBERT to represent this family of sequences.

## 2.4 Neural Numeric Representations

Neural models represent numbers in several ways. Tokenization, dominant in LLMs, maps each integer to a discrete vocabulary entry, but cannot handle out-of-vocabulary values and hides arithmetic structure in token IDs. Continuous log-scale embeddings improve scale invariance but miss periodic arithmetic. Sin/cos positional encodings [cite:vaswani2017attention] exploit periodicity of positions; here, we repurpose the same idea for residues of values.

FiLM [cite:perez2018film] was proposed for dynamically modulating visual features with language features in visual question answering. We reuse FiLM to modulate the Magnitude stream with the Modulo stream, allowing periodic arithmetic information to condition continuous value prediction.

In summary, this work differs from prior OEIS research in three ways:

- Compared with Alien Coding, QSynt, and Learning Conjecturing, we learn shared arithmetic structure in a latent space rather than synthesizing rules for individual sequences.
- Compared with FACT, we address the limitations of a plain tokenized Transformer by adding modulo-spectrum feature engineering and FiLM fusion.
- The dual-stream design in Section 3 can be viewed as an architectural implementation of the mathematical process of observing numerical patterns and extracting laws.
