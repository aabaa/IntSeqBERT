import argparse
import gzip
import logging
import sys
import json
import torch
import multiprocessing
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import os
from functools import partial

# Import modules
from . import converters
from . import schemas
# New feature extraction logic
from intseq_bert.features import extract_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _open_text(path: str):
    """Helper to open plain or gzipped text files."""
    p = Path(path)
    if p.suffix == '.gz':
        return gzip.open(p, 'rt', encoding='utf-8', errors='ignore')
    return open(p, 'rt', encoding='utf-8', errors='ignore')

# ==========================================
# Existing Pipeline Steps
# ==========================================

def process_stripped(args):
    """Handler for converting stripped.gz to jsonl."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    logger.info(f"Converting stripped data: {input_path} -> {output_path}")
    
    converter = converters.StrippedConverter(
        min_len=args.min_len,
        max_val_threshold=10**100
    )
    
    count = 0
    with _open_text(input_path) as fin, open(output_path, 'w', encoding='utf-8') as fout:
        iterator = converter.parse(fin)
        for record in tqdm(iterator, desc="Converting"):
            fout.write(record.to_json_line() + '\n')
            count += 1
            
    logger.info(f"Finished. Converted {count} records.")

def process_merge_names(args):
    """Handler for merging names.gz into existing jsonl."""
    jsonl_path = Path(args.input_jsonl)
    names_path = Path(args.input_names)
    output_path = Path(args.output)
    
    logger.info("Step 1: Loading names into memory map...")
    names_map: Dict[str, str] = {}
    name_parser = converters.NamesConverter()
    
    with _open_text(names_path) as f:
        for oid, name in tqdm(name_parser.parse(f), desc="Loading Names"):
            names_map[oid] = name
            
    logger.info(f"Loaded {len(names_map)} names.")
    
    logger.info("Step 2: Merging into JSONL records...")
    updated_count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc="Merging"):
            if not line.strip(): continue
            
            record = schemas.OEISRecord.from_json_line(line)
            
            if record.oeis_id in names_map:
                record.name = names_map[record.oeis_id]
                updated_count += 1
            
            fout.write(record.to_json_line() + '\n')
            
    logger.info(f"Finished. Updated names for {updated_count} records.")

def process_merge_metadata(args):
    """Handler for merging metadata from .seq files into existing jsonl."""
    jsonl_path = Path(args.input_jsonl)
    seq_dir = Path(args.seq_dir)
    output_path = Path(args.output)
    
    if not jsonl_path.exists():
        logger.error(f"Input JSONL not found: {jsonl_path}")
        return
    if not seq_dir.exists():
        logger.error(f"Sequence directory not found: {seq_dir}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1: Loading existing JSONL records into memory map...")
    records_map: Dict[str, schemas.OEISRecord] = {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading JSONL"):
            if not line.strip(): continue
            rec = schemas.OEISRecord.from_json_line(line)
            records_map[rec.oeis_id] = rec
            
    logger.info(f"Loaded {len(records_map)} records into memory.")

    logger.info(f"Step 2: Scanning .seq files in {seq_dir}...")
    seq_parser = converters.SeqMetadataConverter()
    updated_count = 0
    
    for root, _, files in os.walk(seq_dir):
        for filename in files:
            if not filename.endswith(".seq"):
                continue
            
            oeis_id = filename.replace(".seq", "")
            
            if oeis_id in records_map:
                file_path = Path(root) / filename
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        meta = seq_parser.parse(f, expected_id=oeis_id)
                    
                    record = records_map[oeis_id]
                    if meta["keywords"]: record.keywords = meta["keywords"]
                    record.offset_a = meta["offset_a"]
                    if meta["related"]: record.related = meta["related"]
                    
                    updated_count += 1
                    if updated_count % 10000 == 0:
                        logger.info(f"Processed metadata for {updated_count} sequences...")

                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")

    logger.info(f"Metadata merge complete. Updated {updated_count} records.")

    logger.info(f"Step 3: Saving fully merged data to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as fout:
        sorted_ids = sorted(records_map.keys())
        for oid in tqdm(sorted_ids, desc="Saving"):
            fout.write(records_map[oid].to_json_line() + '\n')
            
    logger.info("Done.")

# ==========================================
# New Step: Feature Extraction
# ==========================================

def _process_feature_chunk(chunk: List[Dict], output_dir: Path) -> int:
    """Helper for multiprocessing feature extraction."""
    count = 0
    for record in chunk:
        oeis_id = record.get('oeis_id')
        seq = record.get('sequence')
        
        # Skip invalid or too short sequences
        if not oeis_id or not seq or len(seq) < 5:
            continue
            
        try:
            seq_ints = [int(x) for x in seq]
            
            # Extract features (New Dual Stream Logic)
            features_dict = extract_features(seq_ints)
            
            # Save structure
            save_data = {
                'oeis_id': oeis_id,
                'mag_features': features_dict['mag_features'], # (Seq, 5)
                'mod_features': features_dict['mod_features'], # (Seq, 200)
                'targets': features_dict['targets']            # Dict[str, Tensor]
            }
            
            torch.save(save_data, output_dir / f"{oeis_id}.pt")
            count += 1
            
        except Exception:
            continue
            
    return count

def process_features(args):
    """Handler for extracting features from jsonl to .pt files."""
    jsonl_path = Path(args.input)
    output_dir = Path(args.output_dir)
    num_workers = args.workers
    chunk_size = args.chunk_size
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from {jsonl_path}")
    chunks = []
    current_chunk = []
    
    # Read JSONL and chunk it
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                if not line.strip(): continue
                record = json.loads(line)
                current_chunk.append(record)
                if len(current_chunk) >= chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = []
            except json.JSONDecodeError:
                continue
    
    if current_chunk:
        chunks.append(current_chunk)
        
    logger.info(f"Split data into {len(chunks)} chunks. Starting processing with {num_workers} workers...")
    
    # Run multiprocessing
    process_func = partial(_process_feature_chunk, output_dir=output_dir)
    
    total_processed = 0
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_func, chunks), total=len(chunks), desc="Extracting Features"))
        total_processed = sum(results)
        
    logger.info(f"Successfully processed {total_processed} sequences.")
    logger.info(f"Saved feature files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="OEIS Data Preprocessing Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Subcommand: stripped
    p_stripped = subparsers.add_parser("stripped", help="Convert stripped.gz to jsonl")
    p_stripped.add_argument("-i", "--input", required=True, help="Path to stripped.gz")
    p_stripped.add_argument("-o", "--output", required=True, help="Output .jsonl path")
    p_stripped.add_argument("--min_len", type=int, default=10, help="Min sequence length")
    p_stripped.set_defaults(func=process_stripped)
    
    # Subcommand: merge-names
    p_names = subparsers.add_parser("merge-names", help="Merge names.gz into existing jsonl")
    p_names.add_argument("--input-jsonl", required=True, help="Path to existing .jsonl")
    p_names.add_argument("--input-names", required=True, help="Path to names.gz")
    p_names.add_argument("-o", "--output", required=True, help="Output path for merged .jsonl")
    p_names.set_defaults(func=process_merge_names)
    
    # Subcommand: merge-metadata
    p_meta = subparsers.add_parser("merge-metadata", help="Merge .seq metadata")
    p_meta.add_argument("--input-jsonl", required=True, help="Path to existing .jsonl")
    p_meta.add_argument("--seq-dir", required=True, help="Path to oeisdata/seq directory root")
    p_meta.add_argument("-o", "--output", required=True, help="Output path for fully merged .jsonl")
    p_meta.set_defaults(func=process_merge_metadata)
    
    # Subcommand: features (NEW)
    p_feat = subparsers.add_parser("features", help="Extract Dual Stream features to .pt files")
    p_feat.add_argument("-i", "--input", required=True, help="Path to input .jsonl")
    p_feat.add_argument("-o", "--output-dir", required=True, help="Output directory for .pt files")
    p_feat.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    p_feat.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for processing")
    p_feat.set_defaults(func=process_features)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()