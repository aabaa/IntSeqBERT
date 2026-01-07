import torch
import glob
import os
import numpy as np
from collections import Counter
from intseq_bert.bert_model import IntSeqBERT

def calculate_entropy(predictions, mod_size):
    """シャノンエントロピーを計算し、理論最大値(ln(mod_size))で正規化する"""
    if not predictions:
        return 0.0
    
    counts = Counter(predictions)
    total = len(predictions)
    probs = np.array([count / total for count in counts.values()])
    
    # H = -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # Normalize by max entropy (uniform distribution)
    max_entropy = np.log(mod_size)
    if max_entropy == 0:
        return 1.0
        
    return entropy / max_entropy

def inspect_full_spectrum():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "checkpoints/bert_multitask_easy/best_model.pt"
    feature_dir = "data/oeis/features_easy"
    
    print(f"Loading Encoder from {ckpt_path} ...")
    if not os.path.exists(ckpt_path):
        return

    model, _ = IntSeqBERT.load_from_checkpoint(ckpt_path, device=device)
    model.eval()

    files = glob.glob(os.path.join(feature_dir, "*.pt"))
    sample_files = files[:200] # 200サンプルで検証
    print(f"Scanning 200 sequences across ALL Mod heads (2-101)...")

    # 全Modの予測結果を格納
    all_preds = {m: [] for m in range(2, 102)}

    with torch.no_grad():
        for i, fpath in enumerate(sample_files):
            data = torch.load(fpath)
            mag = data['mag_features'].unsqueeze(0).to(device)
            mod = data['mod_features'].unsqueeze(0).to(device)
            mask = torch.ones(1, mag.size(1)).to(device)

            outputs = model(mag, mod, mask)
            
            # 全ヘッドのargmaxを取得
            for m in range(2, 102):
                key = f"mod{m}"
                pred = torch.argmax(outputs[key][0, -1]).item()
                all_preds[m].append(pred)

    # --- レポート出力 ---
    print("\n" + "="*80)
    print(f"{'Mod':<5} | {'Unique':<8} | {'Top-1 Val (Rate)':<20} | {'Entropy':<8} | {'Status'}")
    print("-" * 80)

    suspicious_mods = []

    for m in range(2, 102):
        preds = all_preds[m]
        counts = Counter(preds)
        unique_count = len(counts)
        top_val, top_count = counts.most_common(1)[0]
        top_ratio = top_count / len(sample_files)
        
        # 正規化エントロピー (0~1)
        norm_entropy = calculate_entropy(preds, m)
        
        # ステータス判定
        status = ""
        if top_ratio > 0.9:
            status = "🚨 COLLAPSED" # 90%以上が同じ値
            suspicious_mods.append(m)
        elif top_ratio > 0.5:
            status = "⚠️ BIASED"    # 50%以上が同じ値
        elif norm_entropy > 0.7:
            status = "✅ HEALTHY"   # 健全に分散
        else:
            status = "✓ OK"

        # 視覚化用のバー
        bar = "█" * int(norm_entropy * 10)
        entropy_str = f"{norm_entropy:.2f} {bar}"

        print(f"{m:<5} | {unique_count:>3}/{m:<4} | Val {top_val:<3} ({top_ratio*100:>3.0f}%)    | {entropy_str:<8} | {status}")

    print("="*80)
    if suspicious_mods:
        print(f"🚨 WARNING: The following Mods show signs of collapse: {suspicious_mods}")
    else:
        print("🎉 EXCELLENT: No complete collapse detected across the spectrum!")

if __name__ == "__main__":
    inspect_full_spectrum()