"""
Data loading utilities for the Dual Stream architecture.
Handles directory-based dataset of individual .pt files.
"""

import torch
import json
import logging
import random
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from torch.utils.data import Dataset

# スキーマはタグフィルタリングのために必要
from . import schemas

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DualStreamDataset(Dataset):
    """
    Dataset for loading preprocessed Dual Stream features from individual .pt files.
    Implements lazy loading to save memory.
    """
    def __init__(self, feature_files: List[Path]):
        self.feature_files = feature_files

    def __len__(self) -> int:
        return len(self.feature_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Loads a single .pt file on demand.
        Returns dict with keys: 'oeis_id', 'mag_features', 'mod_features', 'targets'
        """
        path = self.feature_files[idx]
        try:
            # map_location='cpu' is crucial to avoid GPU memory leaks in dataloader workers
            data = torch.load(path, map_location='cpu')
            
            # Basic validation
            if 'mag_features' not in data or 'mod_features' not in data:
                 raise ValueError(f"Invalid data format in {path}")
            
            return data
            
        except Exception as e:
            # In a real training loop, you might want to return None and collate it out,
            # but raising error helps debug preprocessing issues early.
            logger.error(f"Error loading {path}: {e}")
            raise e

def load_and_split_data(
    features_dir: str,
    metadata_path: Optional[str] = None,
    include_tags: Optional[List[str]] = None,
    exclude_tags: Optional[List[str]] = None,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    max_samples: Optional[int] = None
) -> Tuple[DualStreamDataset, DualStreamDataset, DualStreamDataset]:
    """
    Scans a directory for .pt files, optionally filters by tags, and splits into Train/Val/Test.
    """
    features_path = Path(features_dir)
    if not features_path.exists():
        raise FileNotFoundError(f"Directory not found: {features_dir}")

    # 1. Tag filtering (Optional)
    valid_ids: Optional[Set[str]] = None
    if metadata_path and (include_tags or exclude_tags):
        logger.info(f"Loading metadata from {metadata_path} for tag filtering...")
        valid_ids = _filter_by_tags(metadata_path, include_tags, exclude_tags)
        logger.info(f"Filtered to {len(valid_ids)} valid IDs based on tags")

    # 2. Collect all .pt files
    logger.info(f"Scanning for .pt files in {features_dir}...")
    # Using glob is faster than os.walk for flat directories
    all_files = sorted(list(features_path.glob("*.pt")))
    
    if len(all_files) == 0:
        raise ValueError("No .pt files found.")
        
    # 3. Apply Filtering
    final_files = []
    if valid_ids is not None:
        for f in all_files:
            # Assumes filename is "A000001.pt"
            oeis_id = f.stem 
            if oeis_id in valid_ids:
                final_files.append(f)
        logger.info(f"Retained {len(final_files)} files after tag filtering.")
    else:
        final_files = all_files

    if max_samples:
        final_files = final_files[:max_samples]
        logger.info(f"Truncating to {max_samples} samples.")

    if len(final_files) == 0:
        raise ValueError("No files remain after filtering.")

    # 4. Shuffle and Split
    random.seed(seed)
    random.shuffle(final_files)
    
    n_total = len(final_files)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test
    
    train_files = final_files[:n_train]
    val_files = final_files[n_train:n_train+n_val]
    test_files = final_files[n_train+n_val:]
    
    logger.info(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    return (
        DualStreamDataset(train_files),
        DualStreamDataset(val_files),
        DualStreamDataset(test_files)
    )

def _filter_by_tags(
    metadata_path: str,
    include_tags: Optional[List[str]],
    exclude_tags: Optional[List[str]]
) -> Set[str]:
    """
    Filter OEIS IDs based on keyword tags. Same logic as before.
    """
    valid_ids: Set[str] = set()
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return valid_ids
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                record = schemas.OEISRecord.from_json_line(line)
                keywords = record.keywords or []
                
                # Exclude filter
                if exclude_tags and any(tag in keywords for tag in exclude_tags):
                    continue
                
                # Include filter
                if include_tags:
                    if not any(tag in keywords for tag in include_tags):
                        continue
                
                valid_ids.add(record.oeis_id)
            except Exception:
                continue
                
    return valid_ids
