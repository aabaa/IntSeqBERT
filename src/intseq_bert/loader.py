"""
Data loader for IntSeqBERT feature tensors with metadata-based filtering.
"""

import gc
import logging
import random
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Set
from torch.utils.data import Dataset

from . import schemas

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntSeqDataset(Dataset):
    """
    Simple PyTorch Dataset wrapper for pre-loaded feature tensors.
    
    Args:
        tensors: List of feature tensors, each with shape (SeqLen, 27)
    """
    
    def __init__(self, tensors: List[torch.Tensor]):
        self.tensors = tensors
    
    def __len__(self) -> int:
        return len(self.tensors)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensors[idx]


def load_and_split_data(
    features_path: str,
    metadata_path: Optional[str] = None,
    include_tags: Optional[List[str]] = None,
    exclude_tags: Optional[List[str]] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    min_len: int = 0
) -> Tuple[IntSeqDataset, IntSeqDataset, IntSeqDataset]:
    """
    Load feature tensors with optional metadata-based filtering and split into train/val/test.
    
    Args:
        features_path: Path to .pt file containing Dict[oeis_id, Tensor]
        metadata_path: Optional path to JSONL file for tag filtering
        include_tags: Keep sequences with ANY of these keywords (OR logic). None = no filtering
        exclude_tags: Remove sequences with ANY of these keywords
        val_ratio: Validation set ratio (0.0 to 1.0)
        test_ratio: Test set ratio (0.0 to 1.0)
        seed: Random seed for reproducible splitting
        min_len: Minimum sequence length to include
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    
    # Step 1: Tag filtering (Metadata Loading)
    valid_ids: Optional[Set[str]] = None
    
    if metadata_path and (include_tags or exclude_tags):
        logger.info(f"Loading metadata from {metadata_path} for tag filtering...")
        valid_ids = _filter_by_tags(metadata_path, include_tags, exclude_tags)
        logger.info(f"Filtered to {len(valid_ids)} valid IDs based on tags")
    
    # Step 2: Feature loading
    logger.info(f"Loading features from {features_path}...")
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    loaded_dict = torch.load(features_path)
    logger.info(f"Loaded {len(loaded_dict)} sequences from features file")
    
    # Step 3: Intersection (filter by valid_ids and min_len)
    filtered_tensors: List[torch.Tensor] = []
    
    for oeis_id, tensor in loaded_dict.items():
        # Check if ID is valid (if filtering is enabled)
        if valid_ids is not None and oeis_id not in valid_ids:
            continue
        
        # Check minimum length
        if tensor.shape[0] < min_len:
            continue
        
        filtered_tensors.append(tensor)
    
    logger.info(f"After filtering: {len(filtered_tensors)} sequences remain")
    
    # Step 4: Memory cleanup - delete large dictionary
    del loaded_dict
    gc.collect()
    
    # Step 5: Splitting
    random.seed(seed)
    random.shuffle(filtered_tensors)
    
    total = len(filtered_tensors)
    if total == 0:
        logger.warning("No sequences remain after filtering!")
        return IntSeqDataset([]), IntSeqDataset([]), IntSeqDataset([])
    
    # Calculate split indices
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    train_size = total - test_size - val_size
    
    train_tensors = filtered_tensors[:train_size]
    val_tensors = filtered_tensors[train_size:train_size + val_size]
    test_tensors = filtered_tensors[train_size + val_size:]
    
    logger.info(f"Split: Train={len(train_tensors)}, Val={len(val_tensors)}, Test={len(test_tensors)}")
    
    # Step 6: Dataset creation
    train_ds = IntSeqDataset(train_tensors)
    val_ds = IntSeqDataset(val_tensors)
    test_ds = IntSeqDataset(test_tensors)
    
    return train_ds, val_ds, test_ds


def _filter_by_tags(
    metadata_path: str,
    include_tags: Optional[List[str]],
    exclude_tags: Optional[List[str]]
) -> Set[str]:
    """
    Filter OEIS IDs based on keyword tags.
    
    Args:
        metadata_path: Path to JSONL file
        include_tags: Keep sequences with ANY of these keywords (OR logic)
        exclude_tags: Remove sequences with ANY of these keywords
    
    Returns:
        Set of valid OEIS IDs that pass the filter
    """
    valid_ids: Set[str] = set()
    
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return valid_ids
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                record = schemas.OEISRecord.from_json_line(line)
                keywords = record.keywords or []
                
                # Exclude filter (takes precedence)
                if exclude_tags:
                    if any(tag in keywords for tag in exclude_tags):
                        continue
                
                # Include filter (if specified, must match at least one)
                if include_tags:
                    if not any(tag in keywords for tag in include_tags):
                        continue
                
                # Passed all filters
                valid_ids.add(record.oeis_id)
                
            except Exception as e:
                logger.warning(f"Error parsing metadata line: {e}")
                continue
    
    return valid_ids
