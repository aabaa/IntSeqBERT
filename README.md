# IntSeqBERT

**IntSeqBERT** is a neuro-symbolic Transformer framework designed to learn deep mathematical representations of integer sequences.

Unlike standard language models that treat numbers as text tokens, IntSeqBERT utilizes a **Dual Stream Architecture** that simultaneously processes:

1. **Magnitude Stream:** Log10-scale absolute values with one-hot sign encoding (4 dims).
2. **Mod Spectrum Stream:** Sin/Cos embeddings for residues across moduli 2 to 101 (200 dims).

## 🏗 Architecture

### Dual Stream Encoder

A BERT-style encoder that fuses two distinct feature streams into a unified latent representation.

* **Inputs:**
  * **Magnitude Stream (4 dims):** `[1 + log10(|x|), sign+, sign-, sign0]`
  * **Mod Spectrum Stream (200 dims):** Sin/Cos pairs for $x \pmod m$ where $m \in [2, 101]$.
* **Mechanism:** Additive Fusion + Transformer Encoder.
* **Masking Strategy:** 
  * Magnitude: 5th channel `is_masked` flag distinguishes zeros from masked positions.
  * Modulo: Masked positions shifted to origin `(0, 0)` on unit circle.

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
  -o data/oeis/splits/strict \
  --include-tags core,easy,nice \
  --exclude-tags cons,base,word,fini,dead,dumb,unkn,less
```

---

## � Project Structure

```text
src/intseq_bert/
├── config.py           # Centralized constants (paths, dimensions, seeds)
├── schemas.py          # Data classes (OEISRecord)
├── features.py         # Feature extraction (Mag + Mod Spectrum)
├── preprocess.py       # CLI: build-jsonl, extract-features, split-dataset
├── loader.py           # OEISDataset, load_dataset, create_splits
└── collator.py         # OEISCollator (dynamic masking, dimension extension)

tests/                  # Unit tests
├── test_schemas.py
├── test_features.py
├── test_loader.py
├── test_collator.py
└── test_preprocess.py
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

## 📄 License

MIT License
