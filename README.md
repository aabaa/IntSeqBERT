# IntSeqBERT

**IntSeqBERT** is a Transformer-based framework designed to learn mathematical representations of integer sequences. Unlike standard language models that treat numbers as text tokens, IntSeqBERT utilizes a **27-dimensional number-theoretic feature vector** and a novel **Probabilistic CRT (Chinese Remainder Theorem) Decoder** to understand the deep structure of sequences.

## 🏗 Architecture

The system consists of two main components:

1. **IntSeqBERT (Encoder):**
   - Compresses integer sequences into dense vector representations.
   - Input: Sequence of 27-dim feature vectors (Log-magnitude, Prime gaps, Valuation, etc.).
   - Objective: Masked Sequence Modeling.
   - Output: Context-aware feature vectors.

2. **NumberTheoreticDecoder (Decoder):**
   - Reconstructs integers from the latent vectors produced by IntSeqBERT.
   - Mechanism: **Multi-Task Learning** + **Probabilistic CRT Search**.
   - Heads:
     - **Sign:** Positive / Negative / Zero.
     - **Magnitude:** Log-scale regression ($\log_{10}|x|$).
     - **Modulo:** Classification of residues for mod 3, 5, 8, 10.

### Reconstruction Logic: CRT Search

The decoder doesn't just guess the number. It uses a **"Lattice Search"** approach:

1. Estimates the rough range using the **Magnitude** head.
2. Filters candidates using **Modulo** constraints (Chinese Remainder Theorem).
3. Even if the magnitude prediction is slightly off, the modulo constraints can "rescue" the prediction and pinpoint the exact integer.

**Example:** If the magnitude prediction suggests the number is around ±50, but the modulo heads correctly predict (mod 3 = 0, mod 5 = 2, mod 10 = 7), the CRT lattice search can identify the exact value (like 42 or 57) even with imprecise magnitude.

## 🚀 Quick Start

### 1. Prerequisites

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync
```

**Requirements:**
- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy ≥ 2.4
- tqdm (progress bars)

### 2. Data Preparation

**Step 1: Download OEIS Data**

```bash
# Download stripped file (integer sequences only)
wget http://oeis.org/stripped.gz
gunzip stripped.gz
mv stripped data/oeis/stripped

# Download names file (optional, for metadata)
wget http://oeis.org/names.gz
gunzip names.gz
mv names data/oeis/names
```

**Step 2: Preprocess Raw Data**

Convert raw OEIS format to JSONL:

```bash
uv run python -m intseq_bert.preprocess \
  process-stripped \
  --input data/oeis/stripped \
  --output data/oeis/data_step1.jsonl
```

Optional: Merge with sequence names

```bash
uv run python -m intseq_bert.preprocess \
  merge-names \
  --sequences data/oeis/data_step1.jsonl \
  --names data/oeis/names \
  --output data/oeis/data_step3.jsonl
```

**Step 3: Extract Features**

Convert JSONL to 27-dimensional feature tensors:

```bash
uv run python -m intseq_bert.encoder \
  --input data/oeis/data_step3.jsonl \
  --output data/oeis/features.pt \
  --batch_size 1000
```

This creates a `.pt` file with format: `{oeis_id: Tensor(seq_len, 27)}`

### 3. Train IntSeqBERT (Encoder)

Train the backbone model to understand sequence contexts.

```bash
uv run python -m intseq_bert.train_bert \
  --features_path data/oeis/features.pt \
  --output_dir checkpoints/bert \
  --epochs 20 \
  --batch_size 128 \
  --lr 1e-4 \
  --d_model 128 \
  --nhead 4 \
  --num_layers 6
```

**Output:**
- `checkpoints/bert/best_model.pt` - Best model checkpoint
- `checkpoints/bert/config.json` - Model configuration
- `checkpoints/bert/train.log` - Training logs

### 4. Train Decoder

Train the decoder to map latent vectors back to integers.

> **Important:** Requires both `features.pt` AND original `jsonl` data for ground truth labels.

```bash
uv run python -m intseq_bert.train_decoder \
  --bert_checkpoint checkpoints/bert/best_model.pt \
  --features_path data/oeis/features.pt \
  --jsonl_path data/oeis/data_step3.jsonl \
  --output_dir checkpoints/decoder \
  --epochs 10 \
  --batch_size 64 \
  --lr 1e-3
```

**Why two data sources?**
- `features.pt`: Provides input to frozen BERT (27-dim vectors)
- `jsonl`: Provides ground truth integers for decoder targets (sign, magnitude, modulo)

## 📊 Evaluation Metrics

When training the decoder, you will see a **Reconstruction Report**:

```
Evaluation Results:
  Mag MAE: 0.082
  Sign Acc: 94.2% | Mod3: 91.8% | Mod10: 87.3%
  Reconstruction (n=500):
    ✓ Perfect: 356 (71.2%)
    ✓ Rescued: 78 (15.6%)  ← CRT Success!
    ✗ Failed: 66 (13.2%)
```

**Metrics Explained:**

- **Perfect:** The model predicted the exact integer correctly from the start.
- **Rescued (CRT Success):** The magnitude prediction was wrong (error > 0.5), but the Modulo constraints successfully corrected it to the right integer. **This demonstrates the power of the number-theoretic approach.**
- **Failed:** The model could not recover the integer.

A high "Rescued" count indicates that discrete modulo information is effectively compensating for continuous magnitude errors.

## 📂 Project Structure

```
IntSeqBERT/
├── src/intseq_bert/
│   ├── bert_model.py         # IntSeqBERT (Encoder) definition
│   ├── decoder_model.py      # NumberTheoreticDecoder + CRT search
│   ├── features.py           # 27-dim Feature Extraction logic
│   ├── encoder.py            # CLI for batch feature encoding
│   ├── train_bert.py         # Training script for Encoder
│   ├── train_decoder.py      # Training script for Decoder
│   ├── loader.py             # Data loading utilities
│   ├── collator.py           # Batching and masking logic
│   ├── preprocess.py         # OEIS data preprocessing
│   └── schemas.py            # Data structures
├── tests/                    # Comprehensive test suite
│   ├── test_bert_model.py
│   ├── test_decoder_model.py
│   ├── test_train_bert.py
│   ├── test_train_decoder.py
│   └── ...
├── data/                     # Data directory (user-created)
└── checkpoints/              # Model checkpoints (generated)
```

## 🧪 Testing

Run the full test suite to ensure system integrity:

```bash
uv run pytest tests/
```

**Test Coverage:**
- Feature extraction logic (27 dimensions)
- BERT model architecture
- Decoder model with CRT search
- Data loading and preprocessing
- Training pipelines (BERT + Decoder)
- Integration tests

Currently **79 tests**, all passing.

## 🎯 Key Features

### 27-Dimensional Feature Vector

Each integer is represented by:
- **Log Magnitude** (1 dim): Continuous scale representation
- **Sign & Direction** (3 dims): Positive/negative/zero classification
- **Modular Arithmetic** (4 dims): Residues mod 3, 5, 8, 10
- **Logarithmic Differences** (4 dims): Growth rate indicators
- **Number-Theoretic Properties** (15 dims): Prime, square, cube, square-free, power-of-2, digit sum, etc.

### Multi-Task Decoder Architecture

```
Input (27 dims) → Feature vector from IntSeqBERT
    ↓
Shared Encoder
    Linear(27 → 256) + ReLU + Dropout
    Linear(256 → 256) + ReLU
    ↓
Multi-Task Heads
    ├─ sign_head: Softmax(256 → 3)
    ├─ mag_head: Regression(256 → 1)
    ├─ mod3_head: Softmax(256 → 3)
    ├─ mod5_head: Softmax(256 → 5)
    ├─ mod8_head: Softmax(256 → 8)
    └─ mod10_head: Softmax(256 → 10)
```

### CRT-Based Integer Reconstruction

The `reconstruct_value` method implements a probabilistic lattice search:

1. **Base Estimate**: Combine sign and magnitude predictions to get approximate value
2. **Search Window**: Generate candidates in range `[base - 150, base + 150]`
3. **Scoring**: For each candidate, compute:
   ```
   score = Σ log P(c mod k | predictions) - λ * (mag_error)²
   ```
4. **Select Best**: Choose candidate with highest score

This approach allows the model to leverage both continuous (magnitude) and discrete (modulo) information.

## 📝 Model Checkpoints

### Loading Models

IntSeqBERT models can be loaded using the convenience class method:

```python
from intseq_bert.bert_model import IntSeqBERT

# Load model with automatic config restoration
model, checkpoint = IntSeqBERT.load_from_checkpoint(
    'checkpoints/bert/best_model.pt',
    device='cuda'
)

print(f"Loaded model from epoch {checkpoint['epoch']}")
```

## 🔬 Advanced Usage

### Custom Training Parameters

**BERT Training:**
```bash
uv run python -m intseq_bert.train_bert \
  --features_path data/oeis/features.pt \
  --metadata_path data/oeis/data_step3.jsonl \
  --include_tags nonn core \
  --exclude_tags dead \
  --d_model 256 \
  --nhead 8 \
  --num_layers 12 \
  --epochs 30 \
  --batch_size 256 \
  --lr 5e-5 \
  --warmup_steps 2000
```

**Decoder Training:**
```bash
uv run python -m intseq_bert.train_decoder \
  --bert_checkpoint checkpoints/large_bert/best_model.pt \
  --features_path data/oeis/features.pt \
  --jsonl_path data/oeis/data_step3.jsonl \
  --epochs 20 \
  --batch_size 128 \
  --lr 5e-4 \
  --weight_decay 0.01
```

## 🤝 Contributing

This is a research project exploring neuro-symbolic approaches to integer sequence modeling. Contributions, suggestions, and discussions are welcome!

## 📄 License

MIT License
