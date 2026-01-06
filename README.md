# IntSeqBERT

**IntSeqBERT** is a neuro-symbolic Transformer framework designed to learn deep mathematical representations of integer sequences.

Unlike standard language models that treat numbers as text tokens, IntSeqBERT utilizes a **Dual Stream Architecture** that simultaneously processes:

1. **Magnitude Stream:** Captures growth rates and approximate values (Continuous).
2. **Mod Spectrum Stream:** Captures cyclic patterns and divisibility properties across moduli 2 to 101 (Discrete).

Combined with a novel **Beam Search CRT (Chinese Remainder Theorem) Solver**, the system can reconstruct exact integers by solving systems of congruences predicted by the neural network.

## 🏗 Architecture

The system consists of two main stages:

### 1. IntSeqBERT (Encoder) - *Dual Stream Fusion*

A BERT-style encoder that fuses two distinct feature streams into a unified latent representation.

* **Inputs:**
  * **Magnitude Stream (5 dims):** Log-magnitude, velocity, acceleration, etc.
  * **Mod Spectrum Stream (200 dims):** Sin/Cos embeddings for $n \pmod m$ where $m \in [2, 101]$.
* **Mechanism:** Additive Fusion + Transformer Encoder.
* **Objective:** Masked Sequence Modeling (Dual Reconstruction Loss).

### 2. IntSeqDecoder (Solver) - *Neuro-Symbolic Reasoning*

A decoder that predicts number-theoretic properties and solves for the exact integer.

* **Heads:**
  * **Magnitude Head:** Heteroscedastic Regression predicting $\mu$ (mean) and $\sigma^2$ (uncertainty) of $\log_{10}|x|$.
  * **Mod Heads:** 100 separate classification heads predicting $x \pmod m$ for every $m$ from 2 to 101.
  * **Sign Head:** Classification (-1, 0, 1).



### Reconstruction Logic: Beam Search CRT

The decoder uses a symbolic solver to find the integer  that best satisfies the predicted constraints:

1. **Entropy Sorting:** Ranks modulo heads by confidence (entropy).
2. **Beam Search:** Incrementally solves the system of congruences using the **Extended Euclidean Algorithm**, keeping only consistent hypotheses.
3. **Magnitude Matching:** Selects the candidate that best fits the predicted magnitude distribution $\mathcal{N}(\mu, \sigma^2)$.

This allows the model to "rescue" predictions: even if the magnitude prediction is vague ($1000 \pm 500$), the modulo constraints (e.g., $x \equiv 3 \pmod{101}$) can pinpoint the exact value (e.g., 1215).

---

## 🚀 Quick Start

### 1. Prerequisites

This project uses `uv` for dependency management.

```bash
uv sync

```

### 2. Data Preparation

**Step 1: Download OEIS Data**

```bash
# 1. Create directory structure
mkdir -p data/oeis/raw
cd data/oeis/raw

# 2. Download basic sequence data
# Note: Use https to ensure connection
wget https://oeis.org/stripped.gz
wget https://oeis.org/names.gz

# 3. Unzip files (The preprocessor expects plain text for these)
gunzip -k stripped.gz
gunzip -k names.gz

# 4. Download extended metadata (keywords, authors, offsets)
# We need the 'seq' directory from the oeisdata repository.
# Go up to 'data/oeis' so 'seq' lands in 'data/oeis/seq'
cd ..

# Clone the repository (Warning: This is large, approx 1GB)
git clone https://github.com/oeis/oeisdata.git

# Move the 'seq' folder to the current directory
mv oeisdata/seq .

# Clean up the rest of the cloned repository to save space
rm -rf oeisdata

# Return to project root
cd ../..
```

**Step 2: Preprocess & Merge Metadata**

```bash
# 1. Convert raw sequences (stripped) to initial JSONL
# Input: Space-separated integers
# Output: Basic JSON records with OEIS ID and sequence
uv run python -m intseq_bert.preprocess stripped \
  --input data/oeis/raw/stripped \
  --output data/oeis/data_step1.jsonl

# 2. Merge sequence names
# Input: data_step1.jsonl + names file
# Output: Records with "name" field added
uv run python -m intseq_bert.preprocess merge-names \
  --input-jsonl data/oeis/data_step1.jsonl \
  --input-names data/oeis/raw/names \
  --output data/oeis/data_step2.jsonl

# 3. Merge extended metadata (Keywords, Offsets) from .seq files
# Input: data_step2.jsonl + seq/ directory
# Output: Final enriched dataset ready for feature extraction
uv run python -m intseq_bert.preprocess merge-metadata \
  --input-jsonl data/oeis/data_step2.jsonl \
  --seq-dir data/oeis/seq \
  --output data/oeis/data_final.jsonl
```

**Step 3: Extract Dual Stream Features**
This step generates individual `.pt` files for each sequence (Lazy Loading ready).

```bash
uv run python -m intseq_bert.preprocess features \
  --input data/oeis/data_final.jsonl \
  --output-dir data/oeis/features \
  --workers 8

```

* **Input**: `data/oeis/data_final.jsonl` (Created in Step 2)

* **Output**: A directory `data/oeis/features/` containing ~360,000 `.pt` files (e.g., `A000001.pt`, `A000042.pt`).

* **Storage**: Requires approx. 15-20 GB of disk space.

### 3. Train IntSeqBERT (Encoder)

Pre-train the encoder using Masked Modeling on both streams.

```bash
uv run python -m intseq_bert.train_bert \
  --features_dir data/oeis/features \
  --output_dir checkpoints/bert \
  --d_model 128 --nhead 4 --num_layers 6 \
  --epochs 10 --batch_size 32

```

### 4. Train Decoder (Solver)

Train the decoder to solve for integers using representations from the frozen encoder.

```bash
uv run python -m intseq_bert.train_decoder \
  --features_dir data/oeis/features \
  --encoder_checkpoint checkpoints/bert/best_model.pt \
  --output_dir checkpoints/decoder \
  --epochs 10 --lr 5e-4

```

---

## 📊 Evaluation Metrics

During decoder training, the system evaluates reconstruction capability:

```text
Evaluation Results:
  Mag Acc (Bin): 94.2%
  Mod Accuracies: 3:98% | 7:96% | ... | 100:92%
  Reconstruction (n=500):
    ✓ Perfect: 380 (76.0%)
    ✓ Rescued: 45 (9.0%)   ← CRT Logic Active
    ✗ Failed:  75 (15.0%)

```

* **Rescued:** Cases where the magnitude prediction was imprecise, but the Beam Search CRT solver successfully used modulo constraints to find the correct integer.

---

## 📂 Project Structure

```text
src/intseq_bert/
├── bert_model.py       # Dual Stream Encoder (Fusion)
├── decoder_model.py    # Decoder + Beam Search CRT Solver
├── features.py         # Feature Extraction (Mag + Mod Spectrum)
├── preprocess.py       # Data Pipeline Entry Point
├── train_bert.py       # Encoder Pre-training Script
├── train_decoder.py    # Decoder Training Script
├── loader.py           # Lazy Loading Dataset (DualStreamDataset)
├── collator.py         # Dual Stream Batching & Masking
├── converters.py       # OEIS Format Parsers
└── schemas.py          # Data Classes (OEISRecord)

tests/                  # Full Test Suite (148 tests)
├── test_features.py
├── test_bert_model.py
├── test_decoder_model.py
├── test_train_bert.py
├── test_train_decoder.py
└── ...

```

## 🧪 Testing

Run the comprehensive test suite to ensure mathematical correctness:

```bash
uv run pytest tests/

```

**Coverage:**

* CRT Solver logic (Extended GCD, Congruence solving)
* Dual Stream fusion and gradients
* Heteroscedastic regression loss
* Lazy loading and batching logic

## 📝 Advanced Configuration

### Dual Stream Config

The model's capacity to handle modular arithmetic is defined by `mod_dim` (default: 200). This covers sin/cos pairs for all moduli from 2 to 101.

### Decoder Config

* `--hidden_dim`: Size of the decoder's shared trunk layer (default: 512).
* `--lr`: Learning rate, default 5e-4 for decoder training.

## 📄 License

MIT License
