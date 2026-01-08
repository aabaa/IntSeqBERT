"""
evaluate_final.py:
Evaluation script for the Encoder-Decoder architecture.
Correctly handles IntSeqBERT's 'encoded_state' output and custom load_from_checkpoint method.
"""

import json
import math
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

# Import models
from intseq_bert.bert_model import IntSeqBERT
from intseq_bert.decoder_model import IntSeqDecoder
from intseq_bert.features import extract_features


def setup_args():
    parser = argparse.ArgumentParser(description="Final Evaluation for IntSeqBERT (Enc-Dec)")
    
    # Model Paths
    parser.add_argument("--model_path", type=str, required=True, help="Path to ENCODER checkpoint")
    parser.add_argument("--decoder_path", type=str, default=None, help="Path to DECODER checkpoint")
    
    # Data paths
    parser.add_argument("--features_dir", type=str, required=True, help="Path to features directory")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to raw data_clean_strict.jsonl")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json")
    
    # Eval params
    parser.add_argument("--beam_width", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples")
    parser.add_argument("--device", type=str, default=None)
    
    # Ignored args
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    
    return parser.parse_args()


def normalize_id(oid: Any) -> str:
    """Normalize ID to 'Axxxxxx' format."""
    s = str(oid).strip()
    if s.startswith('A'):
        s = s[1:]
    return f"A{int(s):06d}"


def load_models(model_path: str, decoder_path: Optional[str], device: str) -> Tuple[nn.Module, nn.Module]:
    """Load Encoder and Decoder safely using model's own methods."""
    print(f"📦 Loading Encoder from {model_path}...")
    
    # 1. Load Encoder using its custom classmethod
    # Returns (model, checkpoint_dict)
    # Argument is 'device', NOT 'map_location'
    try:
        encoder, _ = IntSeqBERT.load_from_checkpoint(model_path, device=device)
    except Exception as e:
        print(f"⚠️ Standard load failed: {e}. Falling back to manual...")
        # Fallback if config is missing in checkpoint
        encoder = IntSeqBERT(d_model=512)
        ckpt = torch.load(model_path, map_location=device)
        state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        # Filter keys
        clean_state = {k.replace('encoder.', ''): v for k, v in state.items() if 'encoder.' in k}
        if not clean_state: clean_state = state # Try direct
        encoder.load_state_dict(clean_state, strict=False)
        encoder.to(device)

    encoder.eval()

    # 2. Initialize Decoder
    # Infer d_model from encoder
    d_model = getattr(encoder, "d_model", 512)
    print(f"ℹ️ Inferred d_model: {d_model}")
    decoder = IntSeqDecoder(d_model=d_model, hidden_dim=512)
    
    # 3. Load Decoder Weights
    dec_path_to_use = decoder_path if decoder_path else model_path
    print(f"📦 Loading Decoder from {dec_path_to_use}...")
    
    dec_checkpoint = torch.load(dec_path_to_use, map_location=device)
    dec_state = dec_checkpoint['state_dict'] if 'state_dict' in dec_checkpoint else dec_checkpoint
    
    # Clean keys (remove 'decoder.' prefix if present)
    clean_dec_state = {}
    for k, v in dec_state.items():
        if k.startswith("decoder."):
            clean_dec_state[k.replace("decoder.", "")] = v
        elif decoder_path: # If explicit path, assume all keys are valid
            clean_dec_state[k] = v

    if clean_dec_state:
        decoder.load_state_dict(clean_dec_state, strict=False)
    else:
        print("⚠️ WARNING: No weights found for Decoder! Using random init.")

    decoder.to(device).eval()
    return encoder, decoder


def load_test_sequences_direct(args) -> List[Dict]:
    """Load sequences by checking feature file existence."""
    data = []
    features_dir = Path(args.features_dir)
    
    print(f"📖 Scanning {args.jsonl_path}...")
    
    with open(args.jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                rec = json.loads(line)
                raw_id = rec.get('oeis_id') or rec.get('id')
                if not raw_id: continue
                
                norm_id = normalize_id(raw_id)
                pt_path = features_dir / f"{norm_id}.pt"
                
                if pt_path.exists():
                    rec['oeis_id'] = norm_id
                    data.append(rec)
                    
                    if args.limit and len(data) >= args.limit:
                        break
            except: continue
            
    print(f"✅ Loaded {len(data)} available sequences.")
    return data


def run_inference(
    encoder: nn.Module,
    decoder: nn.Module,
    input_seq: List[int],
    device: str,
    beam_width: int,
    top_k: int
) -> Dict[str, Any]:
    """Encoder-Decoder Inference."""
    # Preprocess
    feats = extract_features(input_seq)
    mag_f = feats['mag_features']
    mod_f = feats['mod_features']
    
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

    # Forward
    with torch.no_grad():
        # Encoder Output is a DICT with 'encoded_state'
        enc_out = encoder(mag_in, mod_in, mask)
        
        # Correctly extract the tensor
        # bert_model.py: "encoded_state": encoded
        encoded_state = enc_out["encoded_state"]
        
        # Extract embedding of the last valid token
        idx = min(curr_len, max_len) - 1
        latent = encoded_state[:, idx, :] 

        # Decoder
        predictions = decoder(latent)

    # Solve
    candidates = decoder.beam_search_solve(
        predictions,
        beam_width=beam_width,
        max_candidates=top_k
    )
    
    pred_log_mag = predictions["mag_mu"].item()
    
    return {
        "candidates": candidates,
        "predicted_magnitude": 10**pred_log_mag
    }


def main():
    args = setup_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    test_data = load_test_sequences_direct(args)
    if not test_data:
        print("❌ ERROR: No matches found.")
        return

    # 2. Load Models
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
            if not seq and 'seq' in record:
                seq = [int(x) for x in record['seq'].split(',')]
            
            if not seq or len(seq) < 5: continue
            
            input_seq = seq[:-1]
            target = seq[-1]
            
            output = run_inference(encoder, decoder, input_seq, device, args.beam_width, args.top_k)
            
            cands = [c[0] for c in output['candidates']]
            top1 = (target == cands[0]) if cands else False
            top5 = (target in cands[:5])
            
            tgt_log = math.log10(abs(target)) if target != 0 else -1.0
            pred_log = math.log10(output['predicted_magnitude']) if output['predicted_magnitude'] > 0 else -1.0
            mag_err = abs(tgt_log - pred_log)
            
            results["summary"]["total"] += 1
            if top1: results["summary"]["correct_top1"] += 1
            if top5: results["summary"]["correct_top5"] += 1
            results["summary"]["mag_error"] += mag_err
            
            if not top1 or (i % 50 == 0):
                results["logs"].append({
                    "id": record.get('oeis_id'),
                    "target": target,
                    "candidates": cands[:3],
                    "mag_err": round(mag_err, 3)
                })

            if i % 100 == 0:
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
        except Exception as e:
            print(f"🚨 Error in sample {i}: {e}")
            import traceback
            traceback.print_exc()
            break

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
