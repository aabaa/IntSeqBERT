"""
evaluate_final.py:
Evaluation script for the Encoder-Decoder architecture.
Replaces 'solver.py' by directly orchestrating IntSeqBERT (Encoder) and IntSeqDecoder.
"""

import json
import math
import argparse
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple
from tqdm import tqdm

# Import models directly
from intseq_bert.bert_model import IntSeqBERT
from intseq_bert.decoder_model import IntSeqDecoder
from intseq_bert import loader
from intseq_bert.features import extract_features


def setup_args():
    parser = argparse.ArgumentParser(description="Final Evaluation for IntSeqBERT (Enc-Dec)")
    
    # Model Paths
    parser.add_argument("--model_path", type=str, required=True, help="Path to ENCODER checkpoint (IntSeqBERT)")
    parser.add_argument("--decoder_path", type=str, default=None, help="Path to DECODER checkpoint (IntSeqDecoder). If None, tries to load from model_path.")
    
    # Data paths
    parser.add_argument("--features_dir", type=str, required=True, help="Path to features directory")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to raw data_clean_strict.jsonl")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json")
    
    # Data Split
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    
    # Inference Parameters
    parser.add_argument("--beam_width", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Debug limit")
    
    return parser.parse_args()


def load_models(model_path: str, decoder_path: Optional[str], device: str) -> Tuple[nn.Module, nn.Module]:
    """
    Load Encoder and Decoder.
    If decoder_path is provided, loads decoder from there.
    Otherwise, tries to load both from model_path.
    """
    # 1. Initialize Models
    # Ensure d_model matches your training config (usually 512 or 128 depending on your experiments)
    # If your best_model was trained with d_model=512, keep it. If 128, change it here.
    encoder = IntSeqBERT(vocab_size=1000, d_model=512)
    decoder = IntSeqDecoder(d_model=512, hidden_dim=512)
    
    # 2. Load Encoder
    print(f"📦 Loading Encoder from {model_path}...")
    enc_checkpoint = torch.load(model_path, map_location=device)
    enc_state = enc_checkpoint['state_dict'] if 'state_dict' in enc_checkpoint else enc_checkpoint
    
    # Filter/Clean keys for encoder
    clean_enc_state = {}
    for k, v in enc_state.items():
        if k.startswith("encoder."):
            clean_enc_state[k.replace("encoder.", "")] = v
        elif not k.startswith("decoder."): # Assume other keys belong to encoder
            clean_enc_state[k] = v
            
    encoder.load_state_dict(clean_enc_state, strict=False)
    
    # 3. Load Decoder
    dec_path_to_use = decoder_path if decoder_path else model_path
    print(f"📦 Loading Decoder from {dec_path_to_use}...")
    
    dec_checkpoint = torch.load(dec_path_to_use, map_location=device)
    dec_state = dec_checkpoint['state_dict'] if 'state_dict' in dec_checkpoint else dec_checkpoint
    
    clean_dec_state = {}
    for k, v in dec_state.items():
        if k.startswith("decoder."):
            clean_dec_state[k.replace("decoder.", "")] = v
        elif decoder_path: # If explicit path given, assume all keys are for decoder
            clean_dec_state[k] = v

    if clean_dec_state:
        decoder.load_state_dict(clean_dec_state, strict=False)
    else:
        print("⚠️ WARNING: No weights found for Decoder! It will use random initialization.")

    encoder.to(device).eval()
    decoder.to(device).eval()
    
    return encoder, decoder


def get_test_ids_from_loader(features_dir: str, val_ratio: float, test_ratio: float, seed: int) -> Set[str]:
    """Reproduce test split."""
    _, _, test_ds = loader.load_and_split_data(
        features_dir=features_dir, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    return {p.stem for p in test_ds.feature_files}


def load_test_sequences(args) -> List[Dict]:
    """Load raw test data from JSONL."""
    print(f"🔍 Identifying test set...")
    test_ids = get_test_ids_from_loader(args.features_dir, args.val_ratio, args.test_ratio, args.seed)
    
    data = []
    print(f"📖 Scanning {args.jsonl_path}...")
    with open(args.jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                rec = json.loads(line)
                if rec.get('oeis_id') in test_ids:
                    data.append(rec)
            except: continue
            
    return data


def run_inference(
    encoder: nn.Module,
    decoder: nn.Module,
    input_seq: List[int],
    device: str,
    beam_width: int,
    top_k: int
) -> Dict[str, Any]:
    """
    Run the Encoder-Decoder pipeline.
    """
    # 1. Preprocess
    feats = extract_features(input_seq)
    mag_f = feats['mag_features']
    mod_f = feats['mod_features']
    
    # Pad to fixed length for batch processing (Batch=1)
    max_len = 128
    curr_len = mag_f.size(0)
    if curr_len < max_len:
        pad = max_len - curr_len
        mag_in = torch.cat([mag_f, torch.zeros(pad, 5)], dim=0).unsqueeze(0)
        mod_in = torch.cat([mod_f, torch.zeros(pad, 200)], dim=0).unsqueeze(0)
        mask = torch.cat([torch.ones(curr_len), torch.zeros(pad)], dim=0).unsqueeze(0)
    else:
        mag_in = mag_f[-max_len:].unsqueeze(0)
        mod_in = mod_f[-max_len:].unsqueeze(0)
        mask = torch.ones(max_len).unsqueeze(0)

    mag_in, mod_in, mask = mag_in.to(device), mod_in.to(device), mask.to(device)

    # 2. Forward
    with torch.no_grad():
        # Encoder
        enc_out = encoder(mag_in, mod_in, mask)
        
        # Get Latent Vector
        if isinstance(enc_out, dict) and 'last_hidden_state' in enc_out:
            last_hidden = enc_out['last_hidden_state']
        else:
            # Fallback
            last_hidden = enc_out[0] if isinstance(enc_out, tuple) else enc_out
            
        # Extract embedding of the last valid token
        idx = min(curr_len, max_len) - 1
        latent = last_hidden[:, idx, :] 

        # Decoder
        predictions = decoder(latent)

    # 3. Beam Search Solve (Native Decoder Method)
    candidates = decoder.beam_search_solve(
        predictions,
        beam_width=beam_width,
        max_candidates=top_k
    )
    
    # Predictions
    pred_log_mag = predictions["mag_mu"].item()
    
    return {
        "candidates": candidates,
        "predicted_magnitude": 10**pred_log_mag
    }


def main():
    args = setup_args()
    
    # 1. Load Data
    test_data = load_test_sequences(args)
    if args.limit:
        test_data = test_data[:args.limit]
        print(f"⚠️ Limiting to {args.limit} samples")

    # 2. Load Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 【修正箇所】ここで args.decoder_path を渡すように修正しました
    encoder, decoder = load_models(args.model_path, args.decoder_path, device)
    
    # 3. Evaluation Loop
    results = {
        "config": vars(args),
        "summary": {"total": 0, "correct_top1": 0, "correct_top5": 0, "mag_error": 0.0},
        "logs": []
    }
    
    print("🚀 Starting Evaluation (Encoder-Decoder)...")
    
    for i, record in enumerate(tqdm(test_data)):
        try:
            seq = record.get('sequence')
            if not seq or len(seq) < 5: continue
            
            input_seq = seq[:-1]
            target = seq[-1]
            
            # Inference
            output = run_inference(encoder, decoder, input_seq, device, args.beam_width, args.top_k)
            
            # Metrics
            cands = [c[0] for c in output['candidates']]
            top1 = (target == cands[0]) if cands else False
            top5 = (target in cands[:5])
            
            tgt_log = math.log10(abs(target)) if target != 0 else -1.0
            pred_log = math.log10(output['predicted_magnitude']) if output['predicted_magnitude'] > 0 else -1.0
            mag_err = abs(tgt_log - pred_log)
            
            # Update Stats
            results["summary"]["total"] += 1
            if top1: results["summary"]["correct_top1"] += 1
            if top5: results["summary"]["correct_top5"] += 1
            results["summary"]["mag_error"] += mag_err
            
            # Logging
            if not top1 or (i % 50 == 0):
                results["logs"].append({
                    "id": record.get('oeis_id'),
                    "target": target,
                    "candidates": cands[:3],
                    "mag_err": round(mag_err, 3)
                })

            # Save periodically
            if i % 100 == 0:
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
        except Exception as e:
            # print(f"Error: {e}")
            continue

    # Final Report
    total = results["summary"]["total"]
    if total > 0:
        print(f"\n🏆 Results ({total} samples)")
        print(f"Top-1: {results['summary']['correct_top1']/total*100:.2f}%")
        print(f"Top-5: {results['summary']['correct_top5']/total*100:.2f}%")
        print(f"Mag Err: {results['summary']['mag_error']/total:.4f}")
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
