### 1. Magnitude入力の表現力強化 (Input Projection)

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

### 2. 正則化の強化 (Dropout)

過学習（Loss悪化・Acc向上）の傾向が見られた場合、モデル定義レベルで Dropout の適用箇所を念入りに確認します。

**確認事項:**

* `IntSeqEmbeddings`: 最後の `Dropout` は必須。
* `IntSeqModel`: EncoderLayer 内の `dropout` 引数は必須。
* **追加提案:** `IntSeqEmbeddings` の `h_mag` と `h_mod` が融合する前（FiLM適用前）にも Dropout を入れると、片方のストリームへの過度な依存を防げる場合があります。

---

### 3. Vanilla Baseline のトークナイザ制限

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
