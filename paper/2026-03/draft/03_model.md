# 3. IntSeqBERT

<!-- Target: ~3 pages (~1400 words + one architecture figure) -->

## 3.1 Problem Formulation

Let $\mathbf{x} = (x_1, x_2, \ldots, x_L)$ be a finite prefix of an OEIS integer sequence, where $x_i \in \mathbb{Z}$ and $L \leq 128$. We use **masked sequence modelling**: a subset of positions is randomly masked, and the model is trained to predict the masked values.

For each masked position $i$, the model predicts three quantities:

1. **Magnitude**:
$$
v_i =
\begin{cases}
0 & (x_i = 0), \\
1 + \log_{10}(|x_i|) & (x_i \neq 0),
\end{cases}
\in \mathbb{R}_{\geq 0}.
$$
2. **Sign:** $s_i \in \{+, -, 0\}$, represented as a 3-class label.
3. **Residues:** for every $m \in \{2, 3, \ldots, 101\}$, $r_i^{(m)} = x_i \bmod m$, giving 100 independent classification targets.

This decomposition separates scale, sign, and periodic arithmetic structure into complementary supervision signals.

## 3.2 Input Feature Extraction

For each element $x_i$, we compute two feature vectors before learnable embedding.

**Magnitude features** $\mathbf{f}_i^{\text{mag}} \in \mathbb{R}^4$:
$$
\mathbf{f}_i^{\text{mag}} = [v_i,\; \mathbf{1}[x_i > 0],\; \mathbf{1}[x_i < 0],\; \mathbf{1}[x_i = 0]].
$$
The last three components are a one-hot sign representation. For astronomical integers beyond the range of `float64`, we fall back to decimal digit length.

**Modulo features** $\mathbf{f}_i^{\text{mod}} \in \mathbb{R}^{200}$:
For each modulus $m \in \{2, 3, \ldots, 101\}$, let $r = x_i \bmod m$. We embed the residue as a point on the unit circle:
$$
\phi_m(r) = \left[\sin\!\left(\frac{2\pi r}{m}\right),\; \cos\!\left(\frac{2\pi r}{m}\right)\right] \in \mathbb{R}^2.
$$
Concatenating all 100 moduli gives $\mathbf{f}_i^{\text{mod}} \in \mathbb{R}^{200}$. This sin/cos embedding is equivariant to the cyclic structure of $\mathbb{Z}/m\mathbb{Z}$ and avoids discontinuity at the wraparound boundary.

## 3.3 Dual-Stream Embedding

The two feature vectors are projected independently into the hidden dimension $d$. We use a two-layer MLP for the Magnitude stream:
$$
\mathbf{h}_i^{\text{mag}} = \mathrm{MLP}_{\text{mag}}(\mathbf{f}_i^{\text{mag}}), \quad
\mathbf{h}_i^{\text{mod}} = W_{\text{mod}}\,\mathbf{f}_i^{\text{mod}} + \mathbf{b}_{\text{mod}},
$$
where $\mathbf{h}_i^{\text{mag}}, \mathbf{h}_i^{\text{mod}} \in \mathbb{R}^d$.

## 3.4 FiLM Fusion

The streams are fused with Feature-wise Linear Modulation (FiLM) [cite:perez2018film]. The Modulo embedding generates an elementwise scale $\boldsymbol{\gamma}_i$ and shift $\boldsymbol{\beta}_i$ that modulate the Magnitude embedding:
$$
\boldsymbol{\gamma}_i = W_\gamma\,\mathbf{h}_i^{\text{mod}}, \quad
\boldsymbol{\beta}_i = W_\beta\,\mathbf{h}_i^{\text{mod}},
$$
$$
\mathbf{e}_i = (1 + \boldsymbol{\gamma}_i) \odot \mathbf{h}_i^{\text{mag}} + \boldsymbol{\beta}_i.
$$
We apply ReLU after the Modulo projection and dropout before FiLM. This parameter-efficient fusion lets periodic arithmetic information condition continuous magnitude representations. Standard sin/cos positional encodings are added before the Transformer encoder.

Figure~\ref{fig:architecture} shows the overall architecture. Each input element is projected through the Magnitude stream and Modulo stream, fused by FiLM, and processed by a Transformer encoder.

<!-- Figure 1 (paper/2026-03/figures/fig1_architecture.{png,pptx})
     \begin{figure}[t]
       \centering
       \includegraphics[width=\linewidth]{figures/fig1_architecture}
       \caption{IntSeqBERT architecture. The dual-stream embedding block projects Magnitude features
                ($\mathbf{f}^{\mathrm{mag}}\in\mathbb{R}^4$) and Modulo features
                ($\mathbf{f}^{\mathrm{mod}}\in\mathbb{R}^{200}$) into $\mathbb{R}^d$ and fuses
                them with FiLM. Positional encodings are added before a Pre-LN Transformer encoder,
                followed by three prediction heads: magnitude regression, sign classification,
                and 100 modulo classifiers.}
       \label{fig:architecture}
     \end{figure} -->

## 3.5 Transformer Encoder

The fused sequence $(\mathbf{e}_1, \ldots, \mathbf{e}_L)$ is processed by a standard Transformer encoder [cite:vaswani2017attention] with Pre-Layer Normalisation [cite:xiong2020layer]. We evaluate three model sizes:

| Setting | Layers | $d$ | Heads | Approx. Parameters |
|--------|------:|----:|------:|-------------------:|
| Small  | 6    | 256 | 4     | 6.4M               |
| Middle | 8    | 512 | 8     | 29.0M              |
| Large  | 12   | 768 | 12    | 91.5M              |

## 3.6 Prediction Heads

Let $\mathbf{z}_i \in \mathbb{R}^d$ be the encoder output at masked position $i$.

**Magnitude head**:
$$
(\mu_i,\, \log \sigma_i^2) = \mathrm{MLP}_{\text{mag-head}}(\mathbf{z}_i).
$$
The predicted magnitude is $\hat{v}_i=\mu_i$. The head is a two-layer MLP with ReLU activation ($d \to d \to 2$). The variance output is retained as auxiliary uncertainty information.

**Sign head**:
$$
\hat{s}_i = \operatorname{softmax}(W_{\text{sign}}\,\mathbf{z}_i), \quad W_{\text{sign}} \in \mathbb{R}^{3 \times d}.
$$

**Modulo head**:
For each modulus $m \in \{2,\ldots,101\}$, a separate linear classifier predicts logits over $\{0,\ldots,m-1\}$. The 100 classifiers share the same input $\mathbf{z}_i$ but have independent parameters. The total output dimension is $\sum_{m=2}^{101}m=5{,}150$.

## 3.7 Training Objective

The multitask loss is
$$
\mathcal{L} = w_{\text{mag}}\mathcal{L}_{\text{mag}} + w_{\text{sign}}\mathcal{L}_{\text{sign}} + w_{\text{mod}}\mathcal{L}_{\text{mod}},
$$
with $w_{\text{mag}}=1.0$, $w_{\text{sign}}=1.0$, and $w_{\text{mod}}=2.0$. Adaptive schemes such as uncertainty weighting were unstable in preliminary experiments, so we use fixed weights.

$\mathcal{L}_{\text{mag}}$ is the Huber loss between $\hat{v}_i$ and $v_i$. $\mathcal{L}_{\text{sign}}$ is 3-class cross entropy. $\mathcal{L}_{\text{mod}}$ is the average cross entropy over the 100 modulo heads, normalized by $\log m$ to account for class count:
$$
\mathcal{L}_{\text{mod}} = \frac{1}{100}\sum_{m=2}^{101}\frac{1}{\log m}\,\mathcal{L}_{\text{CE}}^{(m)}.
$$
All losses are computed only at masked positions.

## 3.8 Baselines

We compare against two baselines.

**Vanilla Transformer** converts each integer into a token ID in a vocabulary of 20,003 entries: values from 0 to 19,999 plus `PAD`, `MASK`, and `UNK`. Out-of-vocabulary values are replaced by `UNK`. The same three prediction heads are applied to token embeddings. This baseline follows the plain tokenized-Transformer approach used in FACT at the architectural level, but with a smaller vocabulary chosen to fit within the same 8 GB VRAM constraint as IntSeqBERT. FACT used a much larger numeric range under larger computational assumptions.

**Magnitude-only ablation** uses the same architecture as IntSeqBERT but removes the Modulo stream and FiLM module, setting $\mathbf{e}_i=\mathbf{h}_i^{\text{mag}}$. This isolates the contribution of the Modulo stream.

## 3.9 Integer Reconstruction Solver

The pretrained model outputs magnitude $(\mu_i,\log\sigma_i^2)$, sign, and modulo distributions for each masked position. To recover concrete integers, we use **IntegerSolver**.

The solver first derives a 3-sigma interval $[n_{\min}, n_{\max}]$ from the magnitude prediction on the $v=1+\log_{10}(|x|)$ scale. It then chooses one of three modes based on the search width $\Delta n = |n_{\max}-n_{\min}|$:

| Mode | Applicable Range | Method |
|--------|---------|------|
| **Dense** | $\Delta n \leq 10^6$ | Enumerate and score all integers |
| **Sieve** | $10^6 < \Delta n \leq 10^{14}$ | CRT beam search using high-confidence moduli as anchors |
| **CRT** | $\Delta n > 10^{14}$ | Sparse CRT beam search for huge integers |

If the sign prediction is zero, the solver immediately returns 0. If no valid candidate exists in the search range, it records a no-candidate outcome.

Each candidate $n$ is scored by a weighted sum of a magnitude term and modulo log probabilities:
$$
\text{score}(n) =
-\frac{(v_n - \mu_i)^2}{2\sigma_i^2}
+ 0.3 \cdot \sum_{m=2}^{101} \log P(n \bmod m),
$$
where
$$
v_n =
\begin{cases}
0 & (n = 0), \\
1 + \log_{10}(|n|) & (n \neq 0).
\end{cases}
$$
The coefficient 0.3 prevents the modulo term from dominating when information overlaps across related moduli, such as a composite modulus and its prime factors. The solver returns the top-$k$ candidates, which are evaluated as Solver Top-$k$ accuracy in Section 5.4.
