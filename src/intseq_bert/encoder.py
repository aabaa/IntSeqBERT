import argparse
import logging
import torch
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict

# Import internal modules
from . import schemas
from . import features

# Logging configuration
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def process_encode(args):
    """
    Reads a JSONL file, extracts features, and saves them to a single .pt file.
    """
    input_path = Path(args.input)
    output_path = Path(args.output)
    min_len = args.min_len

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dictionary to store results: Dict[oeis_id, Tensor]
    data_map: Dict[str, torch.Tensor] = {}
    
    logger.info(f"Reading from {input_path} and extracting features...")
    
    success_count = 0
    skip_count = 0
    error_count = 0

    # Read and process file
    with open(input_path, 'r', encoding='utf-8') as f:
        # Display progress with tqdm
        for line in tqdm(f, desc="Encoding"):
            if not line.strip():
                continue
                
            try:
                # 1. Parse
                record = schemas.OEISRecord.from_json_line(line)
                seq = record.sequence
                
                # 2. Length check (raw data)
                if len(seq) < min_len:
                    skip_count += 1
                    continue
                
                # 3. Feature extraction (depends on features.py)
                # Return value: numpy.ndarray (Shape: [SeqLen, 35])
                feat_array = features.extract_features(seq)
                
                # If extraction result is empty or invalid
                if feat_array is None or feat_array.shape[0] == 0:
                    skip_count += 1
                    continue

                # 4. Tensor conversion (float32)
                # Use clone() to ensure memory layout
                tensor = torch.tensor(feat_array, dtype=torch.float32).clone()
                
                # 5. Store
                data_map[record.oeis_id] = tensor
                success_count += 1
                
            except Exception as e:
                # Skip and log calculation or parsing errors
                error_count += 1
                # Show error details only for the first 10 cases
                if error_count <= 10:
                    logger.warning(f"Error processing sequence: {e}")
                continue

    # Display result summary
    logger.info("-" * 30)
    logger.info(f"Encoding complete.")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Skipped (too short/empty): {skip_count}")
    logger.info(f"  Errors : {error_count}")
    logger.info("-" * 30)

    if success_count == 0:
        logger.warning("No sequences were successfully encoded. Output file will be empty.")

    # Save process
    logger.info(f"Saving tensor dictionary to {output_path}...")
    try:
        torch.save(data_map, output_path)
        logger.info("Save successful.")
    except Exception as e:
        logger.error(f"Failed to save output file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Encode OEIS sequences to feature tensors.")
    
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Path to input JSONL file (e.g., data/oeis/data_step3.jsonl)")
    parser.add_argument("--output", "-o", type=str, required=True, 
                        help="Path to output .pt file (e.g., data/oeis/features.pt)")
    parser.add_argument("--min_len", type=int, default=10, 
                        help="Minimum sequence length to process (default: 10)")
    
    args = parser.parse_args()
    process_encode(args)

if __name__ == "__main__":
    main()