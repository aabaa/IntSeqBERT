# IntSeqBERT

A Neuro-Symbolic Bidirectional Transformer for Representation Learning of Integer Sequences (OEIS).

## Overview

**IntSeqBERT** learns mathematical structures in integer sequences by combining:
- **Symbolic Feature Extraction**: Converts sequences into 27-dimensional continuous feature vectors capturing number-theoretic properties
- **BERT-style Pretraining**: Uses masked reconstruction (regression task) to learn sequence representations

**Key Innovation**: Instead of treating integers as discrete tokens, we extract rich mathematical features (primality, divisibility, digital properties, etc.) and train a Transformer to predict masked features.

## Architecture

```
Integer Sequence → Feature Extraction → Masked Modeling → Learned Representations
     [1,2,3,5,8]        (27 dims)        Transformer        (contextual embeddings)
```

**Components:**
- **Feature Extractor** (27 dimensions): Number-theoretic, analytic, and digital features
- **Model**: Transformer Encoder with positional encoding
- **Training**: Masked feature reconstruction (15% masking rate)

## Project Structure

```
src/intseq_bert/
├── features.py      # Mathematical feature extraction (27 features per integer)
├── encoder.py       # JSONL → Tensor conversion (.pt file creation)
├── loader.py        # Efficient data loading with metadata filtering
├── collator.py      # Dynamic masking for BERT-style training
├── bert_model.py    # Transformer Encoder (IntSeqBERT architecture)
├── train.py         # Training loop, checkpointing, validation
├── preprocess.py    # OEIS raw data → JSONL conversion
├── schemas.py       # Data schemas (OEISRecord)
├── converters.py    # Format converters (stripped, names, metadata)
└── utils.py         # Number-theoretic utilities (SymPy wrappers)

tests/
├── test_features.py      # Feature extraction tests
├── test_encoder.py       # Encoding pipeline tests
├── test_loader.py        # Data loading tests
├── test_collator.py      # Collator tests
├── test_bert_model.py    # Model architecture tests
├── test_train.py         # Training integration tests
└── ...
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/IntSeqBERT.git
cd IntSeqBERT

# Install dependencies with uv
uv sync
```

### Complete Pipeline

#### Step 1: Download OEIS Data

```bash
# Create directory structure
mkdir -p data/oeis/raw
mkdir -p data/oeis/seq

# Download core data
cd data/oeis/raw
wget https://oeis.org/stripped.gz
wget https://oeis.org/names.gz

# Clone metadata repository (optional, for rich features)
git clone https://github.com/oeis/oeisdata.git oeisdata_repo
cp -r oeisdata_repo/seq/* ../seq/
cd ../../..
```

#### Step 2: Preprocess Raw Data

Convert OEIS data to JSONL format:

```bash
# Basic conversion (stripped.gz → JSONL)
uv run python -m intseq_bert.preprocess stripped \
  --input data/oeis/raw/stripped.gz \
  --output data/oeis/data_step1.jsonl \
  --min_len 10

# Optional: Merge names
uv run python -m intseq_bert.preprocess merge-names \
  --input-jsonl data/oeis/data_step1.jsonl \
  --input-names data/oeis/raw/names.gz \
  --output data/oeis/data_step2.jsonl

# Optional: Merge metadata
uv run python -m intseq_bert.preprocess merge-metadata \
  --input-jsonl data/oeis/data_step2.jsonl \
  --seq-dir data/oeis/seq \
  --output data/oeis/data_step3.jsonl
```

> **Note**: For basic usage, only Step 1 is required. Rename output:
> ```bash
> mv data/oeis/data_step1.jsonl data/oeis/data_step3.jsonl
> ```

#### Step 3: Feature Encoding

Extract mathematical features from sequences (~350k sequences, takes several hours):

```bash
uv run python -m intseq_bert.encoder \
  --input data/oeis/data_step3.jsonl \
  --output data/oeis/features.pt \
  --min_len 10
```

**Output**: `features.pt` containing `Dict[str, torch.Tensor]` where each tensor has shape `(SeqLen, 27)`.

#### Step 4: Train Model

Train IntSeqBERT with masked reconstruction:

```bash
uv run python -m intseq_bert.train \
  --features_path data/oeis/features.pt \
  --output_dir checkpoints/v1 \
  --epochs 20 \
  --batch_size 64 \
  --lr 1e-4 \
  --d_model 128 \
  --nhead 4 \
  --num_layers 6
```

**Advanced Options:**

```bash
# Large model with metadata filtering
uv run python -m intseq_bert.train \
  --features_path data/oeis/features.pt \
  --metadata_path data/oeis/data_step3.jsonl \
  --output_dir checkpoints/large \
  --epochs 30 \
  --batch_size 128 \
  --lr 5e-5 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 12 \
  --warmup_steps 2000
```

## Model Configuration

### Default Architecture

- **Input**: 27-dimensional feature vectors
- **Embedding**: Linear projection to `d_model=128`
- **Transformer**: 6 layers, 4 heads, Pre-LN
- **FFN**: 512 hidden units
- **Output**: Regression head (→ 27 dims)

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate (AdamW) |
| `--weight_decay` | 0.01 | Weight decay |
| `--mask_prob` | 0.15 | Masking probability |
| `--warmup_steps` | auto (10%) | LR warmup steps |

### Data Filtering

```bash
# Filter by keywords
uv run python -m intseq_bert.train \
  --features_path data/oeis/features.pt \
  --metadata_path data/oeis/data_step3.jsonl \
  --include_tags nonn core \
  --exclude_tags dead
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_bert_model.py -v

# Run with coverage
uv run pytest tests/ --cov=src/intseq_bert
```

**Test Coverage:** 62 tests covering all modules

### Code Organization

**Feature Extraction (`features.py`):**
- 27 features per integer (analytic, algebraic, numeric, digital)
- Number-theoretic functions (primality, divisibility, etc.)
- Log-scale transformations for numerical stability

**Data Pipeline:**
- `preprocess.py`: Raw OEIS → JSONL
- `encoder.py`: JSONL → PyTorch tensors
- `loader.py`: Tensor loading + train/val/test splitting

**Model & Training:**
- `bert_model.py`: Transformer architecture
- `collator.py`: Dynamic masking (BERT-style)
- `train.py`: Training loop with validation

## Output Files

### Training Checkpoints

```
checkpoints/
├── best_model.pt     # Best model by validation loss
├── last_model.pt     # Latest epoch model
├── config.json       # Training configuration
└── train.log         # Detailed logs
```

### Checkpoint Format

```python
import torch
checkpoint = torch.load("checkpoints/best_model.pt")

# Contains:
# - model_state_dict: Model weights
# - optimizer_state_dict: Optimizer state
# - scheduler_state_dict: LR scheduler state
# - epoch: Training epoch
# - train_loss, val_loss: Metrics
# - config: Full training config
```

## Data Format Specifications

### JSONL Format (Input to encoder)

```json
{"oeis_id": "A000045", "sequence": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]}
{"oeis_id": "A000027", "sequence": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
```

**Required Fields:**
- `oeis_id` (string): Unique sequence identifier
- `sequence` (array of integers): Sequence values

**Optional Fields:**
- `name`, `keywords`, `offset_a`, `related`

### Feature Tensor Format

Each sequence → `(SeqLen, 27)` tensor:

**Feature Groups:**
1. **Analytic** (8 features): Log magnitude, sign, direction, differences
2. **Algebraic** (4 features): Modular patterns (sin/cos)
3. **Numeric** (9 features): Primality, divisibility, square-free
4. **Digital** (6 features): Binary/decimal properties

## Requirements

- Python ≥ 3.13
- PyTorch ≥ 2.9
- SymPy ≥ 1.14 (for number theory)
- NumPy ≥ 2.4
- tqdm (progress bars)

## License

MIT License

## Acknowledgments

- [OEIS](https://oeis.org/): The Online Encyclopedia of Integer Sequences
- Data from [oeis/oeisdata](https://github.com/oeis/oeisdata) repository
