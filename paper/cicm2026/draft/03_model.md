# 3. IntSeqBERT

<!-- 目標: ~3ページ (~1400語 + アーキテクチャ図1枚) -->

## 3.1 問題定式化

OEIS から取り出した整数数列の有限プレフィックスを $\mathbf{x} = (x_1, x_2, \ldots, x_L)$（$x_i \in \mathbb{Z}$、$L \leq 128$）とする。
本研究では**マスク付き系列モデリング**（masked sequence modelling）を採用する。位置の一部をランダムにマスクし、モデルはマスクされた値を予測するよう学習される。
具体的には、マスクされた各位置 $i$ において以下の 3 つの量を予測する：

1. **Magnitude**：$v_i =
\begin{cases}
0 & (x_i = 0), \\
1 + \log_{10}(|x_i|) & (x_i \neq 0)
\end{cases}
\in \mathbb{R}_{\geq 0}$（対数スケールの絶対値）
2. **符号（Sign）**：$s_i \in \{+, -, 0\}$（3 クラスラベル）
3. **剰余（Residues）**：各 $m \in \{2, 3, \ldots, 101\}$ に対して $r_i^{(m)} = x_i \bmod m$（100 個の独立した分類ターゲット）

この分解により、大きさ・正負・周期的算術構造が相補的な教師信号として分離される。

## 3.2 入力特徴量の抽出

各要素 $x_i$ について、学習可能な埋め込みの前段で 2 種類の特徴ベクトルを計算する。

**Magnitude 特徴量** $\mathbf{f}_i^{\text{mag}} \in \mathbb{R}^4$：
$$
\mathbf{f}_i^{\text{mag}} = \bigl[v_i,\; \mathbf{1}[x_i > 0],\; \mathbf{1}[x_i < 0],\; \mathbf{1}[x_i = 0]\bigr]
$$
後ろ 3 成分は符号（正・負・零）の one-hot 表現である。ただし
$
v_i =
\begin{cases}
0 & (x_i = 0), \\
1 + \log_{10}(|x_i|) & (x_i \neq 0)
\end{cases}
$。
float64 の表現範囲を超える天文学的な整数については、$|x_i|$ の十進桁数でフォールバックする。

**Modulo 特徴量** $\mathbf{f}_i^{\text{mod}} \in \mathbb{R}^{200}$：
各法 $m \in \{2, 3, \ldots, 101\}$ について $r = x_i \bmod m \in \{0, \ldots, m-1\}$ とし、剰余を単位円上の点として埋め込む：
$$
\phi_m(r) = \left[\sin\!\left(\frac{2\pi r}{m}\right),\; \cos\!\left(\frac{2\pi r}{m}\right)\right] \in \mathbb{R}^2.
$$
100 個の法すべてを連結することで $\mathbf{f}_i^{\text{mod}} \in \mathbb{R}^{200}$ を得る。
この Sin/Cos 埋め込みは $\mathbb{Z}/m\mathbb{Z}$ の群構造に対して同変であり、剰余 0 と $m$ が同じ点に写像されるため、折り返し境界での不連続性が生じない。

## 3.3 双ストリーム埋め込み

2 つの特徴ベクトルは独立した射影層でモデルの隠れ次元 $d$ に写像される。本研究では Magnitude 側に 2 層 MLP を用いる：
$$
\mathbf{h}_i^{\text{mag}} = \mathrm{MLP}_{\text{mag}}(\mathbf{f}_i^{\text{mag}}), \quad
\mathbf{h}_i^{\text{mod}} = W_{\text{mod}}\,\mathbf{f}_i^{\text{mod}} + \mathbf{b}_{\text{mod}}, \quad \mathbf{h}_i^{\text{mag}},\,\mathbf{h}_i^{\text{mod}} \in \mathbb{R}^d.
$$

## 3.4 FiLM 融合

2 つのストリームを Feature-wise Linear Modulation（FiLM）[cite:perez2018film] で融合する。
Modulo 埋め込みが要素ごとのスケール $\boldsymbol{\gamma}_i$ とシフト $\boldsymbol{\beta}_i$ を生成し、Magnitude 埋め込みを変調する：
$$
\boldsymbol{\gamma}_i = W_\gamma\,\mathbf{h}_i^{\text{mod}}, \quad \boldsymbol{\beta}_i = W_\beta\,\mathbf{h}_i^{\text{mod}},
$$
$$
\mathbf{e}_i = (1 + \boldsymbol{\gamma}_i) \odot \mathbf{h}_i^{\text{mag}} + \boldsymbol{\beta}_i.
$$
Modulo 側の射影後に ReLU を適用し、さらに FiLM 前にドロップアウトを入れる。
この定式化により、算術的周期性がパラメータ効率よく（$W_\gamma, W_\beta \in \mathbb{R}^{d \times d}$）連続値の Magnitude 表現を条件付けることができる。
エンコーダへの入力前に、$\mathbf{e}_i$ に標準的な Sin/Cos 位置エンコーディングを加算する。

アーキテクチャの全体像を図~\ref{fig:architecture} に示す。
入力系列の各要素は Magnitude ストリーム（青）と Modulo ストリーム（オレンジ）の 2 経路で独立に射影され、FiLM で融合されたのち Transformer エンコーダへ渡される。

<!-- 図1（experiment/cicm2026/fig1_architecture.{pdf,png}）
     \begin{figure}[t]
       \centering
       \includegraphics[width=\linewidth]{figures/fig1_architecture}
       \caption{IntSeqBERT のアーキテクチャ。左の Dual-Stream Embedding ブロックで
                Magnitude 特徴量（$\mathbf{f}^\mathrm{mag}\in\mathbb{R}^4$）と
                Modulo 特徴量（$\mathbf{f}^\mathrm{mod}\in\mathbb{R}^{200}$）を
                それぞれ $\mathbb{R}^d$ に射影し、FiLM で融合する。
                融合埋め込みに位置符号化を加算して Pre-LN Transformer エンコーダに入力し、
                3 つの予測ヘッド（Magnitude 回帰・符号分類・Modulo×100 分類）で出力を得る。}
       \label{fig:architecture}
     \end{figure} -->

## 3.5 Transformer エンコーダ

融合された系列 $(\mathbf{e}_1, \ldots, \mathbf{e}_L)$ を Pre-Layer Normalisation [cite:xiong2020layer] を採用した標準 Transformer エンコーダ [cite:vaswani2017attention] で処理する。
3 つのモデルサイズで実験を行う：

| 設定   | 層数 | $d$ | ヘッド数 | パラメータ数（概算） |
|--------|------|-----|---------|---------------------|
| Small  | 6    | 256 | 4       | 6.4M                |
| Middle | 8    | 512 | 8       | 29.0M               |
| Large  | 12   | 768 | 12      | 91.5M               |

## 3.6 予測ヘッド

マスク位置 $i$ のエンコーダ出力を $\mathbf{z}_i \in \mathbb{R}^d$ とする。

**Magnitude ヘッド**（回帰）：
$$
(\mu_i,\, \log \sigma_i^2) = \mathrm{MLP}_{\text{mag-head}}(\mathbf{z}_i),
$$
対数スケールの Magnitude 予測値は $\hat{v}_i = \mu_i$。
$\mathrm{MLP}_{\text{mag-head}}$ は隠れ次元 $d$・活性化 ReLU の 2 層 MLP（$d \to d \to 2$）である。
本研究では、損失計算で使用するのは主に $\mu_i$ であり、$\log \sigma_i^2$ は補助出力として保持する。

**符号ヘッド**（3 クラス分類）：
$$
\hat{s}_i = \operatorname{softmax}(W_{\text{sign}}\,\mathbf{z}_i), \quad W_{\text{sign}} \in \mathbb{R}^{3 \times d}.
$$

**Modulo ヘッド**（独立した $m$ 値分類器 × 100）：
各法 $m \in \{2, \ldots, 101\}$ について、$\{0, \ldots, m-1\}$ 上のロジットを出力する線形層を用意する。
100 個の分類器は同じ入力 $\mathbf{z}_i$ を共有するが、パラメータは独立する。
総出力次元は、 $\sum_{m=2}^{101} m = 5{,}150$ である。

## 3.7 学習目標

マルチタスク損失は次式で定義する：
$$
\mathcal{L} = w_{\text{mag}}\,\mathcal{L}_{\text{mag}} + w_{\text{sign}}\,\mathcal{L}_{\text{sign}} + w_{\text{mod}}\,\mathcal{L}_{\text{mod}},
$$
ただし $w_{\text{mag}} = 1.0$、$w_{\text{sign}} = 1.0$、$w_{\text{mod}} = 2.0$。
これらの重みに対して不確実性に基づく動的重み付け（uncertainty weighting）などの適応的な手法を試みたところ学習が不安定になることが観察されたため、上述の固定値を採用した。

$\mathcal{L}_{\text{mag}}$ は $\hat{v}_i$ と $v_i$ の間の Huber 損失（Smooth L1）とし、不確実性重み付けは行わない設定とした。
$\mathcal{L}_{\text{sign}}$ は 3 クラスのクロスエントロピー損失である。
$\mathcal{L}_{\text{mod}}$ は 100 個の Modulo ヘッドのクロスエントロピーの平均であり、クラス数の違いを補正するために $\log m$ で正規化する：
$$
\mathcal{L}_{\text{mod}} = \frac{1}{100}\sum_{m=2}^{101}\frac{1}{\log m}\,\mathcal{L}_{\text{CE}}^{(m)}.
$$
すべての損失はマスク位置のみで計算する。

## 3.8 ベースライン
以下2つのモデルを比較対象とした。

**Vanilla Transformer** は各整数を 20,003 エントリの語彙（値 $0$ から $19{,}999$ の 20,000 値に、`PAD`・`MASK`・`UNK` を加えたもの）のトークン ID に変換する。
語彙外の値は `UNK` で置換される。
同じ 3 つの予測ヘッドをトークン埋め込みの出力に適用する。
このベースラインは LLM における数値トークンの標準的な扱いに対応する。
語彙サイズ 20,003 は VRAM 8 GB という制約の下で IntSeqBERT と同等のメモリ消費となるよう設定した。先行研究 FACT [cite:zurich-fact] では 0 から数百万の値を扱っており、本実験より大規模な計算資源を前提としている。

**アブレーション（Magnitude-only）** は IntSeqBERT と同一だが Magnitude ストリームのみを使用し、FiLM モジュールを取り除いて $\mathbf{e}_i = \mathbf{h}_i^{\text{mag}}$ とする。
これにより Modulo ストリームの寄与を単独で定量化できる。

## 3.9 整数復元ソルバー（Solver）

事前学習済みモデルはマスク位置の Magnitude $(\mu_i, \log \sigma_i^2)$・符号・Modulo 確率分布を出力するが、これらから具体的な整数値を復元するために **IntegerSolver** を用いる。

Solver はまず、Magnitude 予測から 3σ 区間 $[n_{\min}, n_{\max}]$（$v = 1 + \log_{10}(|x|)$ スケール）を導出し、探索範囲の広さ $\Delta n = |n_{\max} - n_{\min}|$ に応じて以下の 3 モードを動的に選択する：

| モード | 適用範囲 | 手法 |
|--------|---------|------|
| **Dense** | $\Delta n \leq 10^6$ | 全整数を列挙して評価 |
| **Sieve** | $10^6 < \Delta n \leq 10^{14}$ | 確信度上位の法をアンカーとした CRT ビームサーチで候補を絞り込み |
| **CRT** | $\Delta n > 10^{14}$ | Sparse CRT ビームサーチで巨大整数を直接生成 |

なお、符号予測が零（$x_i = 0$）の場合は整数探索を省略して即時に 0 を返す。探索範囲内に有効な候補が存在しなかった場合は「候補なし」として別途集計する。

各候補 $n$ のスコアは、Magnitude 項と全法の Modulo 対数確率の重み付き和として計算される：
$$
\text{score}(n) =
-\frac{(v_n - \mu_i)^2}{2\sigma_i^2}
+ 0.3 \cdot \sum_{m=2}^{101} \log P\!\left(n \bmod m\right),
$$
ただし
$
v_n =
\begin{cases}
0 & (n = 0), \\
1 + \log_{10}(|n|) & (n \neq 0)
\end{cases}
$。
Modulo 項の係数 0.3 は、法間の情報重複（例：合成数モジュラスとその素因数モジュラスの相関）でスコアが過大になりやすいことを抑えるための経験的ハイパーパラメータとして採用した。

上位 $k$ 件の候補を返し、次項予測精度（Solver Top-$k$）として第 5.4 節で評価する。
