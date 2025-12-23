import argparse
import gzip
import logging
import sys
from pathlib import Path
from typing import Dict
from tqdm import tqdm

# Import modules
from . import converters
from . import schemas

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _open_text(path: str):
    """Helper to open plain or gzipped text files."""
    p = Path(path)
    if p.suffix == '.gz':
        return gzip.open(p, 'rt', encoding='utf-8', errors='ignore')
    return open(p, 'rt', encoding='utf-8', errors='ignore')

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
        # Use tqdm to wrap the iterator
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
            
            # Update name if exists
            if record.oeis_id in names_map:
                record.name = names_map[record.oeis_id]
                updated_count += 1
            
            fout.write(record.to_json_line() + '\n')
            
    logger.info(f"Finished. Updated names for {updated_count} records.")

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
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()