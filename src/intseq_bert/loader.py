"""
loader.py:
Handles loading of OEIS feature files (.pt) based on STATIC split files.
Ensures deterministic data loading by decoupling split creation from data loading.
"""

import torch
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional
from torch.utils.data import Dataset

# Use centralized config and strict schemas
from . import config
from . import schemas

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OEISDataset(Dataset):
    """
    Dataset that loads specific OEIS IDs from individual .pt files.
    Strictly follows config for paths and keys.
    """
    def __init__(self, oeis_ids: List[str], features_dir: Path):
        self.oeis_ids = oeis_ids
        self.features_dir = features_dir

    def __len__(self) -> int:
        return len(self.oeis_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        oeis_id = self.oeis_ids[idx]
        # Construct path using config-defined structure: e.g. data/oeis/features/A000001.pt
        file_path = self.features_dir / f"{oeis_id}.pt"

        if not file_path.exists():
            # Fail fast if data is missing during training/eval
            raise FileNotFoundError(f"Feature file missing for ID {oeis_id}: {file_path}")

        try:
            # Load data to CPU to avoid GPU memory fragmentation during loading
            data = torch.load(file_path, map_location='cpu')

            # Strict Validation using Config Keys
            required_keys = [config.KEY_MAG_FEATURES, config.KEY_MOD_FEATURES]
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key '{key}' in {file_path}")
            
            # Inject OEIS ID into the data dict for tracking
            data[config.KEY_OEIS_ID] = oeis_id
            
            return data

        except Exception as e:
            logger.error(f"Failed to load ID {oeis_id}: {e}")
            raise e


# ==========================================
# 1. Dataset Loading (Runtime: For Training/Eval)
# ==========================================

def load_dataset(split_type: str, split_name: str, *, data_root: Optional[str] = None) -> OEISDataset:
    """
    Loads a dataset based on a pre-existing split file.
    Does NOT perform any shuffling or splitting logic.
    
    Args:
        split_type: e.g., config.SPLIT_STRICT ('strict')
        split_name: 'train', 'val', or 'test'
        data_root: Optional override for config.DATA_ROOT (for testing)
        
    Returns:
        OEISDataset instance
    """
    # Path construction: data/oeis/splits/{split_type}/{split_name}.txt
    root = data_root or config.DATA_ROOT
    split_dir = Path(root) / config.SPLIT_DIR_NAME / split_type
    split_file = split_dir / f"{split_name}.txt"

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}.\n"
            f"Run 'create_splits' first to generate static split lists."
        )

    # Load IDs from file (Physical Isolation)
    with open(split_file, 'r', encoding='utf-8') as f:
        oeis_ids = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(oeis_ids)} IDs from {split_file} (Type: {split_type})")
    
    features_dir = Path(root) / config.FEATURES_DIR_NAME
    return OEISDataset(oeis_ids, features_dir)


# ==========================================
# 2. Split Creation (Admin: One-time setup)
# ==========================================

def create_splits(
    source_jsonl: str,
    output_split_type: str,
    include_tags: Optional[List[str]] = None,
    exclude_tags: Optional[List[str]] = None,
    *,
    data_root: Optional[str] = None
):
    """
    Scans the source JSONL, filters by tags, shuffles ONCE, and writes static ID lists to disk.
    This replaces the old 'load_and_split_data' and ensures physical separation.
    
    Args:
        source_jsonl: Path to data.jsonl
        output_split_type: Directory name to save splits (e.g. 'strict')
        data_root: Optional override for config.DATA_ROOT (for testing)
    """
    root = data_root or config.DATA_ROOT
    logger.info(f"Generating splits for type: {output_split_type}...")
    
    # 1. Collect Valid IDs from JSONL (using schemas for safety)
    valid_ids: List[str] = []
    jsonl_path = Path(source_jsonl)
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Source file not found: {jsonl_path}")

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                # Use schemas to parse strictly
                record = schemas.OEISRecord.from_json_line(line)
                
                # Tag Filtering
                keywords = record.keywords
                if exclude_tags and any(t in keywords for t in exclude_tags):
                    continue
                if include_tags and not any(t in keywords for t in include_tags):
                    continue
                
                valid_ids.append(record.oeis_id)
            except (ValueError, TypeError):
                continue # Skip invalid records

    logger.info(f"Found {len(valid_ids)} valid IDs in JSONL after tag filtering.")

    if not valid_ids:
        raise ValueError("No IDs found matching criteria.")

    # 2. Verify Feature Existence
    # Only keep IDs that actually have a corresponding .pt file in the features directory
    features_dir = Path(root) / config.FEATURES_DIR_NAME
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    existing_ids = []
    # Note: Checking file existence one by one is robust but can be slow for millions of files.
    # For now, we prioritize robustness.
    for oid in valid_ids:
        if (features_dir / f"{oid}.pt").exists():
            existing_ids.append(oid)
    
    logger.info(f"{len(existing_ids)} IDs have corresponding feature files and will be used.")

    if not existing_ids:
        raise ValueError("No matching feature files found.")

    # 3. Deterministic Shuffle
    # Reset seed to ensure the same input always produces the same split
    random.seed(config.SEED)
    random.shuffle(existing_ids)

    # 4. Split logic
    n_total = len(existing_ids)
    n_val = int(n_total * config.VAL_RATIO)
    n_test = int(n_total * config.TEST_RATIO)
    # Train gets the rest to ensure no data loss due to rounding
    
    test_ids = existing_ids[:n_test]
    val_ids = existing_ids[n_test : n_test + n_val]
    train_ids = existing_ids[n_test + n_val :]

    # 5. Write to Disk
    # Path: data/oeis/splits/{output_split_type}/
    save_dir = Path(root) / config.SPLIT_DIR_NAME / output_split_type
    save_dir.mkdir(parents=True, exist_ok=True)

    def _write_list(filename, ids):
        path = save_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            for oid in ids:
                f.write(oid + '\n')
        logger.info(f"Saved {len(ids)} IDs to {path}")

    _write_list("test.txt", test_ids)
    _write_list("val.txt", val_ids)
    _write_list("train.txt", train_ids)

    logger.info("Split generation complete. Static files are ready.")
