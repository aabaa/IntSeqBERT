### 1. ロス爆発の防止策 (Robust Loss)

`Val Loss` が `5.92` まで跳ね上がった原因は、二乗誤差 `(μ - y)²` が外れ値に対して過敏すぎるためです。
Heteroscedastic Loss の計算式に **Huber化（Smooth L1化）** または **Clamping** を導入し、数学的に上限を設ける仕様に変更しましょう。

**変更案 (Section 4.1. Magnitude Loss):**

通常のガウス分布尤度（MSEベース）ではなく、**Laplace分布（絶対値誤差ベース）** または **Huber化されたガウス分布** を仮定します。

```python
# 変更前 (MSEベース: 外れ値で爆発する)
# L_mag = (1/2) * log_var + (mu - y)**2 / (2 * var)

# 変更後 (Robust: ある程度以上離れたら線形ペナルティにする)
precision = torch.exp(-log_var)
diff = torch.abs(mu - y)

# Huber-style loss: 誤差が小さい時は二乗、大きい時は線形
# (簡易実装として SmoothL1Loss を使うか、以下のようなロジック)
is_small_error = diff < 1.0
squared_loss = 0.5 * diff**2
linear_loss = diff - 0.5
recon_loss = torch.where(is_small_error, squared_loss, linear_loss)

# 不確実性の項も合わせる
L_mag = 0.5 * log_var + (precision * recon_loss)

```

これにより、予測を大外ししてもLossが無限大に行かず、学習が安定します。

---

### 2. 不確実性パラメータの暴走防止 (Variance Clamping)

学習が進むと、モデルが自信過剰になり `log_var` が極端に小さく（負の無限大へ）なることがあります。これが `Loss` の不安定化を招きます。

**変更案 (Section 3.3. IntSeqForPreTraining):**

`forward` 内で `log_var` を計算する際、値をクリッピングします。

```python
# mag_log_var の出力直後
mag_log_var = self.mag_head_logvar(x)
# -10 (分散 e^-10 ≈ 0.000045) を下限とする
mag_log_var = torch.clamp(mag_log_var, min=-10.0, max=10.0)

```

---

### 3. Magnitude入力の表現力強化 (Input Projection)

**現状:** `mag_proj: Linear(5, 512)`
**課題:** 入力の次元数「5」は情報量が少なく、単純な線形変換だけでは 512次元の空間にリッチに展開できない可能性があります（ランク不足）。

**改善案 (Section 3.1. IntSeqEmbeddings):**

Magnitude入力に対して、**MLP (Multi-Layer Perceptron)** を通すことで、特徴量を非線形に膨らませてから FiLM に渡すようにします。

```python
# 変更前
self.mag_proj = nn.Linear(MAG_EXTENDED_DIM, d_model)

# 変更後
self.mag_proj = nn.Sequential(
    nn.Linear(MAG_EXTENDED_DIM, d_model),
    nn.GELU(),  # 非線形性
    nn.Linear(d_model, d_model)
)

```

これにより、単なる数値のスケーリングだけでなく、数値間の複雑な関係性を埋め込みベクトルに反映しやすくなります。

---

### 4. 正則化の強化 (Dropout)

過学習（Loss悪化・Acc向上）の傾向が見られるため、モデル定義レベルで Dropout の適用箇所を念入りに確認します。

**確認事項:**

* `IntSeqEmbeddings`: 最後の `Dropout` は必須。
* `IntSeqModel`: EncoderLayer 内の `dropout` 引数は必須。
* **追加提案:** `IntSeqEmbeddings` の `h_mag` と `h_mod` が融合する前（FiLM適用前）にも Dropout を入れると、片方のストリームへの過度な依存を防げる場合があります。

---

### 5. Vanilla Baseline のトークナイザ制限

**課題:**
仕様書 7.5 にて `VANILLA_VOCAB_SIZE = 10003` とありますが、これだと `10000` を超える数値はすべて `[UNK]` になり、Magnitude（大きさ）の比較ができなくなります。
これでは `IntSeqBERT` (連続値扱える) vs `Vanilla` (扱えない) の勝負になり、**「アーキテクチャの勝利」ではなく「入力形式の勝利」** になってしまいます。

**改善案 (Section 7.5. Vanilla Baseline):**

ベースラインを少し強く（公平に）するために、**Digit-level Tokenization**（数値を桁ごとのトークン列にする）を採用するか、あるいは制限を明記する必要があります。

* **案A (簡易):** 現状のままとし、論文では「Vanilla Transformer (Fixed Vocab)」として、語彙外への弱さを指摘する材料にする。（こちらのほうが実験は楽です）
* **案B (強力):** 数値を文字列として扱い BPE/Digit Tokenizer を使う。

今回は **案A** で進め、「IntSeqBERTなら未知の大きな数でも推論できる」ことを強みとして主張するのが良いでしょう。ただし、仕様書には「※大きな数は UNK になる制限あり」と注記を追加しておくと親切です。

---

### 修正後の仕様書への反映

上記の **1. (Robust Loss)** と **2. (Variance Clamping)** は、今の「Loss爆発」を抑えるための決定打になるため、**必須**で取り込むことをお勧めします。

> **Note for Implementation:**
> To prevent loss instability observed in previous experiments:
> 1. Use **Huber-like loss** or **L1 loss** component for Magnitude regression instead of pure MSE to handle outliers.
> 2. **Clamp** `mag_log_var` between `-10` and `10` to prevent variance collapse.
> 3. Use an **MLP** (Linear->GELU->Linear) for `mag_proj` to enhance feature extraction.
