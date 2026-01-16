# IntSeqBERT

**IntSeqBERT** is a neuro-symbolic Transformer framework designed to learn deep mathematical representations of integer sequences.

Unlike standard language models that treat numbers as discrete text tokens, IntSeqBERT utilizes a **Dual Stream Architecture** that simultaneously processes both the "magnitude" (scale) and "periodicity" (modulo properties) of numbers.

## 🏗 Architecture

### Dual Stream Encoder

The model fuses two distinct feature streams into a unified latent representation using **FiLM (Feature-wise Linear Modulation)**.

*   **Inputs:**
    *   **Magnitude Stream (5 dims):** `[1 + log10(|x|), sign+, sign-, sign0, is_masked]`
    *   **Mod Spectrum Stream (200 dims):** Sin/Cos embeddings for $x \pmod m$ where $m \in [2, 101]$.
*   **Fusion Mechanism:** FiLM (Feature-wise Linear Modulation). The Mod spectrum stream modulates the Magnitude stream, allowing the model to "understand" how periodicity interacts with scale.
*   **Backbone:** Standard Transformer Encoder (BERT-style).

### Output Streams

The model performs multi-task learning with three simultaneous objectives:

1.  **Magnitude (Regression):** Predicts `log10(|x|)` using Heteroscedastic Regression (predicting both mean $\mu$ and variance $\sigma^2$).
2.  **Sign (Classification):** Predicts the sign of the number (+, -, or 0).
3.  **Modulo (Classification):** Predicts the residue $x \pmod m$ for all 100 moduli simultaneously.

---

## 📊 Dataset: OEIS

This project uses the **Online Encyclopedia of Integer Sequences (OEIS)**.

### Tag Filtering Strategy

We use [Official OEIS Keywords](https://oeis.org/wiki/Clear-cut_examples_of_keywords) to define dataset subsets.

| Subset | Type | Strategy | Tags |
|--------|------|----------|------|
| **`easy`** | PoC | **Include** | `core`, `easy`, `nice` |
| **`std`** | Main | **Exclude** | `cons`, `base`, `word`, `fini`, `dead`, `dumb`, `unkn`, `less`, `tabl`, `frac`, `cofr` |
| **`all`** | Test | **None** | (All sequences) |

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
uv sync
```

### 2. Data Preparation

**Step 1: Download OEIS Data**

```bash
mkdir -p data/oeis/raw
cd data/oeis/raw

# Download and decompress
wget https://oeis.org/stripped.gz
wget https://oeis.org/names.gz

# Optional: Clone metadata repository (large, ~1GB)
cd ..
git clone --depth 1 https://github.com/oeis/oeisdata.git
mv oeisdata/seq .
rm -rf oeisdata

cd ../..
```

**Step 2: Build JSONL**

```bash
uv run python -m intseq_bert.preprocess build-jsonl \
  --stripped data/oeis/raw/stripped.gz \
  --names data/oeis/raw/names.gz \
  --seq-dir data/oeis/seq \
  -o data/oeis/data.jsonl
```

**Step 3: Extract Features**

```bash
uv run python -m intseq_bert.preprocess extract-features \
  -i data/oeis/data.jsonl \
  -o data/oeis/features \
  --workers 8
```

**Step 4: Create Train/Val/Test Splits**

```bash
# Basic split (all sequences)
uv run python -m intseq_bert.preprocess split-dataset \
  -j data/oeis/data.jsonl \
  -f data/oeis/features \
  -o data/oeis/splits/all

# With tag filtering (recommended)
uv run python -m intseq_bert.preprocess split-dataset \
  -j data/oeis/data.jsonl \
  -f data/oeis/features \
  -o data/oeis/splits/std \
  --exclude-tags cons,base,word,fini,dead,dumb,unkn,less
```

### 3. Training

Train the IntSeqBERT model using the `train.py` script. This script handles the multi-task learning loop, automatically balancing losses for Magnitude, Sign, and Modulo tasks.

```bash
uv run python -m intseq_bert.train \
  --split_type std \
  --output_dir checkpoints/intseq_std \
  --epochs 20 \
  --batch_size 32 \
  --num_workers 4
```

**Loss Weighting:**
To prevent task collapse, we use fixed loss weights:
*   Magnitude: 1.0
*   Sign: 1.0
*   Modulo: 2.0 (Emphasized to encourage learning arithmetic structure)

---

## 📈 Analysis

We provide a suite of analysis tools to evaluate the model's mathematical understanding.

### Modulo Spectrum Analysis (`analyze_mod_spectrum`)

Evaluates how well the model understands different moduli (2 to 101). It produces a "Normalized Information Gain (NIG)" ranking, showing which arithmetic properties the model has learned best (e.g., parity, mod-10 patterns).

```bash
uv run python -m intseq_bert.analysis.analyze_mod_spectrum \
  --checkpoint checkpoints/intseq_std/best_model.pt \
  --split_type std \
  --output_dir results/analysis_mod \
  --model_type intseq
```

### Magnitude Analysis (`analyze_magnitude`)

Analyzes the regression performance across different scales (from small integers to astronomical numbers).

*   **Scale-wise Analysis:** Breaking down error by order of magnitude.
*   **Calibration:** Checking if the model's predicted uncertainty matches its actual error.

```bash
uv run python -m intseq_bert.analysis.analyze_magnitude \
  --checkpoint checkpoints/intseq_std/best_model.pt \
  --split_type std \
  --output_dir results/analysis_mag \
  --model_type intseq
```

Other tools include:
*   `analyze_attention`: Visualizes attention maps to see if the model attends to mathematically relevant positions.
*   `analyze_cases`: Deep dive into specific sequences or error cases.

---

## 📁 Project Structure

```text
src/intseq_bert/
├── config.py           # Centralized constants (paths, dimensions, seeds)
├── schemas.py          # Data classes (OEISRecord)
├── features.py         # Feature extraction (Mag + Mod Spectrum)
├── preprocess.py       # CLI: build-jsonl, extract-features, split-dataset
├── loader.py           # OEISDataset, load_dataset, create_splits
├── collator.py         # OEISCollator (dynamic masking, dimension extension)
├── models.py           # IntSeqEmbeddings (FiLM), IntSeqModel, Heads
├── train.py            # Training loop & Validation
└── analysis/           # Analysis tools
    ├── analyze_mod_spectrum.py
    ├── analyze_magnitude.py
    ├── analyze_attention.py
    └── analyze_cases.py

tests/                  # Unit tests
```

## 🧪 Testing

```bash
uv run pytest tests/ -v
```

## 📝 Configuration

Key constants in `config.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `MAG_RAW_DIM` | 4 | Magnitude input dimensions |
| `MAG_EXTENDED_DIM` | 5 | With `is_masked` flag |
| `MOD_FEATURE_DIM` | 200 | Sin/Cos pairs for 100 moduli |
| `NUM_MODULI` | 100 | Moduli range: 2 to 101 |
| `MAX_SEQUENCE_LENGTH` | 128 | Truncation limit |
| `MIN_SEQUENCE_LENGTH` | 10 | Minimum for feature extraction |
| `SEED` | 42 | Random seed for reproducibility |
| `MASK_PROB` | 0.15 | Masking probability for BERT training |

## 📄 License

MIT License
