# `src/intseq_bert/analysis/analyze_solver.py` 実装仕様書

## 目次

1. [概要](#1-概要)
2. [依存関係](#2-依存関係)
3. [コマンドライン引数](#3-コマンドライン引数)
4. [処理フロー](#4-処理フロー)
5. [分析指標](#5-分析指標)
6. [出力ファイル](#6-出力ファイル)
7. [実装上の工夫](#7-実装上の工夫)
8. [エラーハンドリング](#8-エラーハンドリング)

---

## 1. 概要

学習済みモデルと `solver.py` を組み合わせ、テストデータに対して「次の項」の数値を復元（推論）し、その **完全一致正解率（Exact Match Accuracy）** を計測する。

### 評価のポイント

1. **Exact Match:** 数値が完全に一致したか（GPT-4等との最大の差別化ポイント）
2. **Top-K Accuracy:** 正解が候補の Top-5 に含まれていたか
3. **Solver Mode別性能:** Dense / Sieve / CRT の各モードでどれくらい解けているか
4. **桁数別性能:** 小さい数から巨大数まで、どの範囲が得意か

---

## 2. 依存関係

### ライブラリ

```python
import torch
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
```

### 内部モジュール

```python
from intseq_bert import config
from intseq_bert.solver import IntegerSolver
from intseq_bert.models import IntSeqForPreTraining
from intseq_bert.features import process_sequence
from intseq_bert.collator import OEISCollator
```

### 設定 (`config.py`)

| 定数 | 値 | 用途 |
|------|------|------|
| `MOD_RANGE` | `list(range(2, 102))` | 法のリスト |
| `MAGNITUDE_BUCKETS` | リスト | 桁数別分類の閾値 |
| `SOLVER_TOP_K_DEFAULT` | 5 | デフォルトの候補数 |

---

## 3. コマンドライン引数

```bash
python -m intseq_bert.analysis.analyze_solver \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --split_type std \
    --split_name test \
    --output_dir results/solver_analysis \
    --max_samples 1000 \
    --top_k 5
```

### 引数一覧

| 引数 | 型 | 必須 | デフォルト | 説明 |
|------|------|------|-----------|------|
| `--checkpoint` | str | ✅ | - | モデルチェックポイントパス |
| `--split_type` | str | ✅ | - | 分割タイプ (例: `std`, `easy`) |
| `--split_name` | str | | `test` | 分割名 (`train`, `val`, `test`) |
| `--output_dir` | str | ✅ | - | 出力ディレクトリ |
| `--data_root` | str | | `config.DATA_ROOT` | データルートディレクトリ |
| `--max_samples` | int | | `1000` | 評価する最大サンプル数 |
| `--top_k` | int | | `5` | Solver が返す候補数 |
| `--filter_magnitude` | str | | `None` | 特定桁数範囲のみテスト (`small`, `medium`, `large`, `huge`) |
| `--batch_size` | int | | `1` | バッチサイズ（現在は1件ずつ処理） |
| `--device` | str | | `auto` | デバイス指定 (`cuda`, `cpu`, `auto`) |

---

## 4. 処理フロー

### 4.1. メインフロー

```
1. 引数パース & ロギング設定
2. モデル読み込み (チェックポイントから設定復元)
3. JSONL からテストデータ読み込み
4. 推論ループ (1件ずつ処理)
5. 結果の集計
6. 出力ファイル生成
```

### 4.2. Step 1: データの準備

通常の `DataLoader` は数値を Tensor（Magnitude/Mod）に変換してしまうが、**Exact Match 判定には「変換前の生の整数」が必要**。

そのため、JSONL を直接読み込み、以下を行う：

1. 系列から最後の項（ターゲット）を分離
2. 入力系列に対して `features.process_sequence()` でテンソル化
3. ターゲットは Python int として保持

```python
def load_test_samples(
    jsonl_path: Path,
    split_ids: List[str],
    max_samples: int
) -> List[Dict]:
    """
    Returns:
        List of dicts with keys:
          - oeis_id: str
          - input_seq: List[int] (ターゲット除く)
          - target: int (正解の整数)
          - target_str: str (正解の文字列表現)
    """
```

### 4.3. Step 2: 推論ループ

各サンプルに対して以下を実行：

```python
for sample in tqdm(samples):
    # 1. 特徴量作成
    features_dict = process_sequence(sample["input_seq"])
    
    # 2. バッチ形式に変換 (B=1)
    batch = collator([features_dict])
    
    # 3. Forward Pass
    with torch.no_grad():
        outputs = model(
            mag_features=batch["mag_inputs"].to(device),
            mod_features=batch["mod_inputs"].to(device),
            src_key_padding_mask=(batch["attention_mask"] == 0).to(device)
        )
    
    # 4. Solver 用パラメータ抽出 (最後の位置 = 次の項予測)
    last_pos = batch["attention_mask"].sum(dim=1).item() - 1
    args = IntegerSolver.from_model_output(
        outputs["predictions"], position=last_pos, model=model
    )
    
    # 5. Solve
    candidates = solver.solve(*args, top_k=top_k)
    
    # 6. 判定
    match_rank = -1
    for rank, cand in enumerate(candidates, 1):
        if cand["value"] == sample["target"]:
            match_rank = rank
            break
    
    # 7. 結果記録
    results.append({
        "oeis_id": sample["oeis_id"],
        "target": sample["target"],
        "target_str": sample["target_str"],
        "pred_top1": candidates[0]["value"] if candidates else None,
        "match_rank": match_rank,
        "solver_mode": candidates[0]["method"] if candidates else "none",
        "mag_log10": math.log10(abs(sample["target"])) if sample["target"] != 0 else 0,
        "score_top1": candidates[0]["score"] if candidates else None,
        "sign_pred": args[2],  # sign_idx
        "sign_true": 2 if sample["target"] == 0 else (0 if sample["target"] > 0 else 1)
    })
```

> **Note:** マスク位置について
> 
> IntSeqBERT は「マスクされた位置の次の項」を予測する設計。
> 評価時は、**入力系列の最後の位置**が「次の項を予測する位置」として扱われる。
> Collator のマスク処理は使わず、最後の位置の予測を直接使用する。

### 4.4. Step 3: 結果の集計

全サンプルの結果を集計し、統計量を算出する。

---

## 5. 分析指標

### 5.1. Overall Metrics

| 指標 | 計算方法 | 説明 |
|------|----------|------|
| `top1_acc` | `(match_rank == 1).mean() × 100` | Top-1 完全一致率 (%) |
| `top5_acc` | `(match_rank > 0).mean() × 100` | Top-5 内に正解が含まれる率 (%) |
| `sign_acc` | `(sign_pred == sign_true).mean() × 100` | 符号予測正解率 (%) |
| `valid_rate` | `(match_rank != -1 or candidates not empty)` | Solver が解を返した率 |

### 5.2. By Magnitude (桁数別)

`config.MAGNITUDE_BUCKETS` に基づく分類：

| バケット | 範囲 (log10) | 数値範囲 |
|----------|-------------|---------|
| Small | 0 ~ 2 | 1 ~ 100 |
| Medium | 2 ~ 5 | 100 ~ 100,000 |
| Large | 5 ~ 20 | 10^5 ~ 10^20 |
| Huge | 20 ~ 50 | 10^20 ~ 10^50 |
| Astronomical | 50+ | 10^50+ |

各バケットについて Top-1 Acc, Top-5 Acc を算出。

### 5.3. By Solver Mode

| モード | 説明 |
|--------|------|
| `dense` | Mode A: 全探索 |
| `sieve` | Mode AB: アンカー・シーブ |
| `crt` | Mode B: Sparse CRT |
| `zero` | ゼロ即時返却 |
| `none` | 解なし |

各モードについて使用率と正解率を算出。

---

## 6. 出力ファイル

### 6.1. ディレクトリ構成

```text
results/solver_analysis/
├── solver_results.csv        # 全サンプルの詳細結果
├── summary.json              # 集計サマリー
├── magnitude_breakdown.csv   # 桁数別集計
├── mode_breakdown.csv        # モード別集計
└── analysis_config.json      # 実行設定
```

### 6.2. `solver_results.csv`

個別サンプルの詳細結果。

| カラム | 型 | 説明 |
|--------|------|------|
| `oeis_id` | str | OEIS ID |
| `target` | int | 正解の整数 |
| `target_str` | str | 正解の文字列表現（大きな数用） |
| `pred_top1` | int | Top-1 予測値 |
| `match_rank` | int | 正解の順位 (1-5, -1=不正解) |
| `solver_mode` | str | 使用したモード |
| `mag_log10` | float | ターゲットの桁数 (log10) |
| `score_top1` | float | Top-1 のスコア |
| `sign_pred` | int | 予測した符号 (0/1/2) |
| `sign_true` | int | 正解の符号 (0/1/2) |

```csv
oeis_id,target,target_str,pred_top1,match_rank,solver_mode,mag_log10,score_top1,sign_pred,sign_true
A000045,13,13,13,1,dense,1.114,-0.05,0,0
A000040,101,101,99,-1,dense,2.004,-1.20,0,0
A123456,12345678901234567890,12345678901234567890,12345678901234567890,1,crt,19.091,-0.01,0,0
```

### 6.3. `summary.json`

```json
{
  "overall": {
    "total_samples": 1000,
    "top1_acc": 45.2,
    "top5_acc": 62.8,
    "sign_acc": 98.5,
    "valid_rate": 99.1
  },
  "by_magnitude": {
    "Small": {"count": 450, "top1_acc": 72.3, "top5_acc": 85.1},
    "Medium": {"count": 320, "top1_acc": 48.5, "top5_acc": 65.2},
    "Large": {"count": 180, "top1_acc": 22.1, "top5_acc": 38.5},
    "Huge": {"count": 45, "top1_acc": 8.9, "top5_acc": 15.6},
    "Astronomical": {"count": 5, "top1_acc": 0.0, "top5_acc": 0.0}
  },
  "by_mode": {
    "dense": {"count": 850, "top1_acc": 52.1},
    "sieve": {"count": 100, "top1_acc": 28.0},
    "crt": {"count": 45, "top1_acc": 8.9},
    "zero": {"count": 5, "top1_acc": 100.0}
  },
  "execution": {
    "total_time_sec": 245.3,
    "avg_time_per_sample_sec": 0.245
  }
}
```

### 6.4. `magnitude_breakdown.csv`

```csv
bucket,count,top1_acc,top5_acc,top1_count,top5_count
Small,450,72.3,85.1,325,383
Medium,320,48.5,65.2,155,209
Large,180,22.1,38.5,40,69
Huge,45,8.9,15.6,4,7
Astronomical,5,0.0,0.0,0,0
```

### 6.5. `mode_breakdown.csv`

```csv
mode,count,usage_rate,top1_acc,top5_acc
dense,850,85.0,52.1,68.5
sieve,100,10.0,28.0,45.0
crt,45,4.5,8.9,15.6
zero,5,0.5,100.0,100.0
```

### 6.6. `analysis_config.json`

```json
{
  "checkpoint": "checkpoints/intseq_std/best_model.pt",
  "split_type": "std",
  "split_name": "test",
  "max_samples": 1000,
  "top_k": 5,
  "filter_magnitude": null,
  "device": "cuda",
  "timestamp": "2026-01-19 16:00:00"
}
```

---

## 7. 実装上の工夫

### 7.1. 正解データの取得方法

JSONL ファイルを直接読み込み、`process_sequence()` 相当の処理でテンソル化しつつ、**「最後の項（ターゲット）」を Python int として保持**する。

> **重要:** 既存の DataLoader 経由だと、巨大整数が float 変換でロスする可能性があるため、JSONL 直接読み込みを採用。

```python
def load_test_samples(jsonl_path: Path, split_ids: Set[str], max_samples: int):
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["oeis_id"] not in split_ids:
                continue
            
            seq = record["sequence"]
            if len(seq) < 2:
                continue
            
            target = seq[-1]
            input_seq = seq[:-1]
            
            samples.append({
                "oeis_id": record["oeis_id"],
                "input_seq": input_seq,
                "target": target,
                "target_str": str(target)
            })
            
            if len(samples) >= max_samples:
                break
    
    return samples
```

### 7.2. モデルパラメータの復元

チェックポイントから `d_model`, `nhead`, `num_layers` を復元する。

```python
def load_model_from_checkpoint(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 設定復元 (train.py と同じロジック)
    ckpt_config = checkpoint.get("config", {})
    model = IntSeqForPreTraining(
        d_model=ckpt_config.get("d_model", config.D_MODEL),
        nhead=ckpt_config.get("nhead", config.NHEAD),
        num_layers=ckpt_config.get("num_layers", config.NUM_LAYERS)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model
```

### 7.3. パフォーマンス目安

| サンプル数 | 推定時間 (GPU) | 推定時間 (CPU) |
|-----------|---------------|---------------|
| 100 | ~30 秒 | ~2 分 |
| 1000 | ~5 分 | ~20 分 |
| 全件 (2500) | ~12 分 | ~50 分 |

---

## 8. エラーハンドリング

| 状況 | 対応 |
|------|------|
| チェックポイント不存在 | `FileNotFoundError` を raise |
| JSONL 不存在 | `FileNotFoundError` を raise |
| Split ファイル不存在 | `FileNotFoundError` を raise |
| Solver が空リストを返す | `match_rank = -1`, `solver_mode = "none"` として記録 |
| 系列長が短すぎる (< 2) | スキップ |
| 巨大数で log10 オーバーフロー | `math.log10` のガード（文字列長で代用） |

---

## 9. 使用例

### 基本的な使用

```bash
python -m intseq_bert.analysis.analyze_solver \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --split_type std \
    --output_dir results/solver_analysis
```

### 小さい数だけテスト

```bash
python -m intseq_bert.analysis.analyze_solver \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --split_type std \
    --output_dir results/solver_small \
    --filter_magnitude small
```

### 全件評価

```bash
python -m intseq_bert.analysis.analyze_solver \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --split_type std \
    --output_dir results/solver_full \
    --max_samples 999999
```
