import torch
import glob
import os
import collections
from intseq_bert.bert_model import IntSeqBERT

def inspect_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 疑わしいEncoderのチェックポイントをロード
    # (Decoderの学習ログに出ていたスコアの元凶はこれです)
    ckpt_path = "checkpoints/bert_multitask_easy/best_model.pt"
    
    print(f"Loading Encoder from {ckpt_path} ...")
    if not os.path.exists(ckpt_path):
        print("Error: Checkpoint not found.")
        return

    # Encoderのロード
    model, checkpoint = IntSeqBERT.load_from_checkpoint(ckpt_path, device=device)
    model.eval()

    # 2. データのロード (Datasetクラスを使わず直接読み込む)
    feature_dir = "data/oeis/features_easy"
    files = glob.glob(os.path.join(feature_dir, "*.pt"))
    
    if not files:
        print("Error: No feature files found in data/oeis/features_easy")
        return

    # ランダムにではなく、先頭から100個ほどのファイルを見てみる
    sample_files = files[:200]
    print(f"Inspecting {len(sample_files)} sequences...")

    # 統計用リスト
    preds_mod7 = []
    preds_mod100 = []
    
    # ループ処理
    with torch.no_grad():
        for fpath in sample_files:
            data = torch.load(fpath)
            
            # テンソルを取り出してバッチ化 (Batch Size = 1)
            mag = data['mag_features'].unsqueeze(0).to(device) # (1, Seq, 5)
            mod = data['mod_features'].unsqueeze(0).to(device) # (1, Seq, 200)
            
            # マスク作成 (すべてValidとする)
            seq_len = mag.size(1)
            mask = torch.ones(1, seq_len).to(device)

            # 推論
            outputs = model(mag, mod, mask)
            
            # --- Mod 7 の予測 (最後の項) ---
            logits_7 = outputs['mod7'] # (1, Seq, 7)
            last_pred_7 = torch.argmax(logits_7[0, -1]).item()
            preds_mod7.append(last_pred_7)
            
            # --- Mod 100 の予測 (最後の項) ---
            logits_100 = outputs['mod100'] # (1, Seq, 100)
            last_pred_100 = torch.argmax(logits_100[0, -1]).item()
            preds_mod100.append(last_pred_100)

    # 3. 結果発表
    print("\n" + "="*60)
    print("📊 DIAGNOSIS REPORT: Are predictions collapsed?")
    print("="*60)

    # --- Mod 7 Analysis ---
    print(f"\n[Mod 7 Predictions] (Expected Range: 0-6)")
    c7 = collections.Counter(preds_mod7)
    print(f"Unique values predicted: {len(c7)} / 7")
    print("Top 5 frequent predictions:")
    for val, count in c7.most_common(5):
        ratio = count / len(sample_files) * 100
        bar = "#" * int(ratio // 2)
        print(f"  Val {val}: {count:>3} times ({ratio:>5.1f}%) {bar}")

    if c7.most_common(1)[0][1] > len(sample_files) * 0.9:
        print("🚨 ALERT: Mod 7 is COLLAPSED (predicting same value always)!")
    else:
        print("✅ Mod 7 looks distributed.")

    # --- Mod 100 Analysis ---
    print(f"\n[Mod 100 Predictions] (Expected Range: 0-99)")
    c100 = collections.Counter(preds_mod100)
    print(f"Unique values predicted: {len(c100)}")
    print("Top 10 frequent predictions:")
    for val, count in c100.most_common(10):
        ratio = count / len(sample_files) * 100
        bar = "#" * int(ratio // 2)
        print(f"  Val {val}: {count:>3} times ({ratio:>5.1f}%) {bar}")

    top_val, top_count = c100.most_common(1)[0]
    if top_count > len(sample_files) * 0.5:
        print(f"🚨 ALERT: Mod 100 shows heavy bias towards {top_val}!")
        if top_val < 3:
            print("   -> Likely 'Small Number Bias' (model bets on 0, 1, or 2).")
    else:
        print("✅ Mod 100 looks distributed.")

if __name__ == "__main__":
    inspect_encoder()