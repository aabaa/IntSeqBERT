"""
preprocess.py:
The main entry point for the OEIS data pipeline.
Handles raw data parsing, structure validation, feature extraction, and dataset splitting.
Strictly separates parsing logic, file I/O, and command execution.
"""

import argparse
import gzip
import logging
import multiprocessing
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from functools import partial

import torch
from tqdm import tqdm

# Internal modules
from . import config
from . import schemas
from . import features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================
# Layer 1: Pure Logic Functions (Parsing)
# ==========================================

def _parse_stripped_line(line: str) -> Optional[Tuple[str, List[int]]]:
    """
    Parses a single line from stripped.gz.
    Format: "A000001 ,1,2,3,5,8"
    """
    if not line:
        return None
    
    parts = line.strip().split(" ,")
    if len(parts) != 2:
        return None
    
    oeis_id = parts[0]
    seq_str = parts[1]
    
    # Basic validation
    if not oeis_id.startswith("A"):
        return None
        
    try:
        # Parse sequence
        sequence = [int(x) for x in seq_str.split(",")]
        return oeis_id, sequence
    except ValueError:
        # Non-integer values in sequence
        return None

def _parse_names_line(line: str) -> Optional[Tuple[str, str]]:
    """
    Parses a single line from names.gz.
    Format: "A000001 Name of the sequence"
    """
    if not line or line.startswith("#"):
        return None
    
    # Split by first space only
    parts = line.strip().split(" ", 1)
    if len(parts) != 2:
        return None
        
    oeis_id = parts[0]
    name = parts[1]
    
    if not oeis_id.startswith("A"):
        return None
        
    return oeis_id, name

def _parse_seq_content(lines: List[str]) -> Dict[str, Any]:
    """
    Parses content of an OEIS internal format (.seq) file.
    Extracts Keywords (%K) and Offsets (%O).
    """
    meta = {
        "keywords": [],
        "offset_a": 0
    }
    
    for line in lines:
        if line.startswith("%K"):
            # Format: %K A000001 nonn,easy
            content = line[3:].strip().split(maxsplit=1)
            if len(content) > 1:
                # content[0] is ID, content[1] is keywords
                # But sometimes ID is skipped or implicit? 
                # Usually: "%K A000045 nonn,easy"
                # We just want the keywords part.
                kw_str = content[-1]
                meta["keywords"] = [k.strip() for k in kw_str.split(",")]
                
        elif line.startswith("%O"):
            # Format: %O A000001 0,2
            content = line[3:].strip().split(maxsplit=1)
            if len(content) > 1:
                offsets = content[-1].split(",")
                if len(offsets) >= 1:
                    try:
                        meta["offset_a"] = int(offsets[0])
                    except ValueError:
                        pass
    return meta


# ==========================================
# Layer 2: Worker & Helper Functions
# ==========================================

def _load_names_map(names_path: Path) -> Dict[str, str]:
    """Loads all names into memory for fast lookup."""
    logger.info(f"Loading names from {names_path}...")
    names_map = {}
    with gzip.open(names_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Reading names"):
            res = _parse_names_line(line)
            if res:
                names_map[res[0]] = res[1]
    logger.info(f"Loaded {len(names_map)} names.")
    return names_map

def _scan_seq_files(seq_dir: Path) -> Dict[str, Path]:
    """Maps OEIS IDs to their .seq file paths."""
    logger.info(f"Scanning .seq files in {seq_dir}...")
    seq_map = {}
    # Walk directory
    for root, _, files in os.walk(seq_dir): # Need to import os
        for filename in files:
            if filename.endswith(".seq"):
                oeis_id = filename.replace(".seq", "")
                if oeis_id.startswith("A"):
                    seq_map[oeis_id] = Path(root) / filename
    logger.info(f"Found {len(seq_map)} metadata files.")
    return seq_map

def _worker_extract_features(chunk: List[str], output_dir: Path) -> int:
    """
    Worker process for extracting features from a chunk of JSONL lines.
    Saves .pt files directly.
    """
    count = 0
    for line in chunk:
        try:
            # 1. Deserialize
            record = schemas.OEISRecord.from_json_line(line)
            
            # 2. Filter by length (Config-driven)
            if len(record.sequence) < config.MIN_SEQUENCE_LENGTH:
                continue

            # 3. Feature Extraction
            # features.process_sequence handles truncation and logic
            features_dict = features.process_sequence(record.sequence)
            
            # 4. Save
            # Add ID to the dict for safety/verification
            features_dict[config.KEY_OEIS_ID] = record.oeis_id
            
            save_path = output_dir / f"{record.oeis_id}.pt"
            torch.save(features_dict, save_path)
            count += 1
            
        except Exception as e:
            # Log at DEBUG level to avoid flooding stdout
            logger.debug(f"Failed to process record: {e}")
            continue
            
    return count


# ==========================================
# Layer 3: Command Handlers
# ==========================================

def cmd_build_jsonl(args):
    """
    Command: build-jsonl
    Combines stripped, names, and metadata into a single data.jsonl file.
    """
    stripped_path = Path(args.stripped)
    output_path = Path(args.output)
    
    # Optional Loaders
    names_map = {}
    if args.names:
        names_map = _load_names_map(Path(args.names))
        
    seq_map = {}
    if args.seq_dir:
        seq_map = _scan_seq_files(Path(args.seq_dir))
        
    logger.info(f"Processing {stripped_path} -> {output_path}")
    
    count = 0
    with gzip.open(stripped_path, 'rt', encoding='utf-8', errors='ignore') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc="Building JSONL"):
            # 1. Parse Stripped
            parsed = _parse_stripped_line(line)
            if not parsed:
                continue
                
            oeis_id, sequence = parsed
            
            # 2. Merge Name
            name = names_map.get(oeis_id, "")
            
            # 3. Merge Metadata (On-demand read)
            keywords = []
            offset_a = 0
            if oeis_id in seq_map:
                try:
                    with open(seq_map[oeis_id], 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        meta = _parse_seq_content(lines)
                        keywords = meta["keywords"]
                        offset_a = meta["offset_a"]
                except Exception:
                    pass

            # 4. Create Record & Write
            try:
                record = schemas.OEISRecord(
                    oeis_id=oeis_id,
                    sequence=sequence,
                    name=name,
                    offset_a=offset_a,
                    keywords=keywords
                )
                fout.write(record.to_json_line() + '\n')
                count += 1
            except Exception as e:
                logger.debug(f"Record creation failed for {oeis_id}: {e}")
                
    logger.info(f"Completed. Built {count} records.")


def cmd_extract_features(args):
    """
    Command: extract-features
    Converts JSONL to .pt files using multiprocessing.
    """
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Read JSONL into chunks
    logger.info(f"Reading {input_path}...")
    chunks = []
    current_chunk = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            current_chunk.append(line)
            if len(current_chunk) >= args.chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
    
    if current_chunk:
        chunks.append(current_chunk)
        
    logger.info(f"Prepared {len(chunks)} chunks. Starting {args.workers} workers.")
    
    # 2. Run Parallel Processing
    # Use partial to pass fixed arguments
    worker = partial(_worker_extract_features, output_dir=output_dir)
    
    total_processed = 0
    with multiprocessing.Pool(args.workers) as pool:
        # tqdm is in the main process
        results = list(tqdm(pool.imap_unordered(worker, chunks), total=len(chunks), desc="Extracting"))
        total_processed = sum(results)
        
    logger.info(f"Done. Extracted features for {total_processed} sequences.")


def cmd_split_dataset(args):
    """
    Command: split-dataset
    Splits data into train/val/test lists with optional tag filtering.
    Uses JSONL for tag info and verifies .pt file existence.
    """
    jsonl_path = Path(args.jsonl)
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    
    # Parse tag arguments
    include_tags = None
    exclude_tags = None
    if args.include_tags:
        include_tags = [t.strip() for t in args.include_tags.split(",")]
    if args.exclude_tags:
        exclude_tags = [t.strip() for t in args.exclude_tags.split(",")]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Reading JSONL: {jsonl_path}")
    logger.info(f"Features dir: {features_dir}")
    if include_tags:
        logger.info(f"Include tags: {include_tags}")
    if exclude_tags:
        logger.info(f"Exclude tags: {exclude_tags}")
    
    # 1. Collect IDs from JSONL with tag filtering
    valid_ids = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Filtering JSONL"):
            if not line.strip():
                continue
            try:
                record = schemas.OEISRecord.from_json_line(line)
                keywords = record.keywords or []
                
                # Tag filtering
                if exclude_tags and any(t in keywords for t in exclude_tags):
                    continue
                if include_tags and not any(t in keywords for t in include_tags):
                    continue
                
                valid_ids.append(record.oeis_id)
            except Exception:
                continue
    
    logger.info(f"Found {len(valid_ids)} IDs after tag filtering.")
    
    if not valid_ids:
        logger.warning("No IDs found matching criteria. Exiting.")
        return
    
    # 2. Verify feature file existence
    existing_ids = []
    for oid in valid_ids:
        if (features_dir / f"{oid}.pt").exists():
            existing_ids.append(oid)
    
    logger.info(f"{len(existing_ids)} IDs have corresponding .pt files.")
    
    if not existing_ids:
        logger.warning("No matching feature files found. Exiting.")
        return
    
    # 3. Deterministic shuffle
    random.seed(config.SEED)
    random.shuffle(existing_ids)
    
    # 4. Split
    n_total = len(existing_ids)
    n_test = int(n_total * config.TEST_RATIO)
    n_val = int(n_total * config.VAL_RATIO)
    
    test_ids = existing_ids[:n_test]
    val_ids = existing_ids[n_test : n_test + n_val]
    train_ids = existing_ids[n_test + n_val:]
    
    # 5. Save
    def _save_list(name, ids):
        path = output_dir / name
        with open(path, 'w') as f:
            for oid in ids:
                f.write(oid + '\n')
        logger.info(f"Saved {name}: {len(ids)} IDs")
    
    _save_list("test.txt", test_ids)
    _save_list("val.txt", val_ids)
    _save_list("train.txt", train_ids)
    
    logger.info("Split complete.")


# ==========================================
# Main Entry Point
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="IntSeqBERT Data Preprocessing")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # 1. build-jsonl
    p_build = subparsers.add_parser("build-jsonl", help="Convert raw OEIS data to JSONL")
    p_build.add_argument("--stripped", required=True, help="Path to stripped.gz")
    p_build.add_argument("--names", help="Path to names.gz")
    p_build.add_argument("--seq-dir", help="Directory containing .seq files")
    p_build.add_argument("-o", "--output", required=True, help="Output .jsonl file")
    p_build.set_defaults(func=cmd_build_jsonl)
    
    # 2. extract-features
    p_feat = subparsers.add_parser("extract-features", help="Generate .pt files from JSONL")
    p_feat.add_argument("-i", "--input", required=True, help="Input .jsonl file")
    p_feat.add_argument("-o", "--output-dir", required=True, help="Output directory for .pt files")
    p_feat.add_argument("--workers", type=int, default=4)
    p_feat.add_argument("--chunk-size", type=int, default=1000)
    p_feat.set_defaults(func=cmd_extract_features)
    
    # 3. split-dataset
    p_split = subparsers.add_parser("split-dataset", help="Split data into train/val/test with tag filtering")
    p_split.add_argument("-j", "--jsonl", required=True, help="Path to data.jsonl (for tag info)")
    p_split.add_argument("-f", "--features-dir", required=True, help="Directory with .pt files")
    p_split.add_argument("-o", "--output-dir", required=True, help="Output directory for split lists")
    p_split.add_argument("--include-tags", help="Comma-separated tags to include (OR logic)")
    p_split.add_argument("--exclude-tags", help="Comma-separated tags to exclude (OR logic)")
    p_split.set_defaults(func=cmd_split_dataset)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
