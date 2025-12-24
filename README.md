# IntSeqBERT
A Neuro-Symbolic Bidirectional Transformer for Representation Learning of Integer Sequences (OEIS).

## Data Preparation

### 1. Data Pipeline Overview

The data processing pipeline consists of the following stages:

* **Raw OEIS Data** → Download from OEIS database (`stripped.gz`, optionally `names.gz`)
* **Preprocessing** → Parse and filter raw data into JSONL format (`data_step3.jsonl`)
* **Feature Extraction** → Convert sequences into multi-dimensional feature tensors (`features.pt`)

### 2. Download Raw Data

Download the OEIS database files. For the full dataset (including metadata), we recommend cloning the `oeisdata` repository and organizing the directory structure.

**Required Files:**
* **stripped.gz**: Core sequence data (OEIS IDs and sequence values)
* **names.gz**: Sequence names and descriptions (Optional)
* **oeisdata (GitHub repo)**: Contains metadata and `.seq` files (Optional, but recommended for rich features)

**Setup Commands:**
```bash
# 1. Create directory structure
mkdir -p data/oeis/raw
mkdir -p data/oeis/seq

# 2. Download minimal files
cd data/oeis/raw
wget https://oeis.org/stripped.gz
wget https://oeis.org/names.gz

# 3. Clone oeisdata and extract sequence files
# We clone into a temporary directory and move the 'seq' folder to data/oeis/seq
git clone https://github.com/oeis/oeisdata.git oeisdata_repo
cp -r oeisdata_repo/seq/* ../seq/
# (Optional) Clean up the repo to save space
# rm -rf oeisdata_repo
cd ../../..
```

**Final Directory Structure:**

```text
data/oeis/
├── raw/
│   ├── stripped.gz
│   └── names.gz
└── seq/             <-- Contains A000001.seq, etc.
    ├── A000...
    └── ...
```

### 3. Preprocessing (Raw → JSONL)

Convert the raw OEIS data into JSONL format using the `preprocess.py` module.

**Step 1: Convert stripped.gz to JSONL**
```bash
uv run python -m intseq_bert.preprocess stripped \
  --input data/oeis/raw/stripped.gz \
  --output data/oeis/data_step1.jsonl \
  --min_len 10
```

**Step 2 (Optional): Merge sequence names**
```bash
uv run python -m intseq_bert.preprocess merge-names \
  --input-jsonl data/oeis/data_step1.jsonl \
  --input-names data/oeis/raw/names.gz \
  --output data/oeis/data_step2.jsonl
```

**Step 3 (Optional): Merge additional metadata**

If you have set up the `data/oeis/seq` directory:
```bash
uv run python -m intseq_bert.preprocess merge-metadata \
  --input-jsonl data/oeis/data_step2.jsonl \
  --seq-dir data/oeis/seq \
  --output data/oeis/data_step3.jsonl
```

> [!NOTE]
> For basic usage, you only need Step 1. The output file can be renamed to `data_step3.jsonl` for consistency:
> ```bash
> mv data/oeis/data_step1.jsonl data/oeis/data_step3.jsonl
> ```

### 4. Input Data Specification (The JSONL Format)

The encoder expects input data in JSON Lines format. **Users must prepare their data according to the following specification:**

* **Format:** JSON Lines (`.jsonl`) - one JSON object per line
* **Required Fields:**
    * `oeis_id` (string): Unique identifier for the sequence (e.g., "A000045")
    * `sequence` (array of integers): The integer sequence values

* **Optional Fields:**
    * `name` (string): Human-readable sequence name
    * `keywords` (array of strings): Classification keywords  
    * `offset_a` (integer): Sequence offset
    * `related` (array of strings): Related sequence IDs

* **Example:**
    ```json
    {"oeis_id": "A000045", "sequence": [0, 1, 1, 2, 3, 5, 8]}
    {"oeis_id": "A000027", "sequence": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    {"oeis_id": "A000290", "sequence": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]}
    ```

### 5. Feature Encoding

To convert your JSONL data into feature tensors, use the `encoder.py` module:

**Command:**
```bash
uv run python -m intseq_bert.encoder \
  --input data/oeis/data_step3.jsonl \
  --output data/oeis/features.pt
```

**Optional Parameters:**
* `--min_len`: Minimum sequence length to process (default: 10)

**Output Format:**

The generated `.pt` file contains a `Dict[str, torch.Tensor]` where:
* **Keys:** OEIS IDs (strings)
* **Values:** Feature tensors with shape `(SeqLen, 27)` and dtype `float32`
* Each row represents one integer in the sequence with 27 extracted features
