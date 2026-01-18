### 1. Vanilla Baseline のトークナイザ制限

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
