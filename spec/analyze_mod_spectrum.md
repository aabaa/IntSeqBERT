# `src/intseq_bert/analysis/analyze_mod_spectrum.py` 実装仕様書

## 1. 概要

本スクリプトは、学習済みモデルの **Modulo Spectrum Analysis** を実行する。
全ての法 m (2〜101) を対等な条件で比較し、モデルが得意とする構造をランキング化する。
**IntSeqBERT**, **Vanilla Transformer**, **Ablation (No-Mod)** の全てのモデルタイプに対応する。

### 主要機能

1. **Global Ranking**: 法 m ごとの NIG (Normalized Information Gain) を算出し、ランキング化
2. **Tag-Stratified Analysis**: OEISタグごとの層別分析
3. **Bootstrap CI**: 統計的有意性のための 95% 信頼区間推定

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
from typing import Dict, Optional, List, Tuple
from torch.utils.data import DataLoader
```

### 内部モジュール

```python
from intseq_bert import config
from intseq_bert.loader import load_dataset, OEISDataset
from intseq_bert.collator import OEISCollator
from intseq_bert.analysis.common import create_model_wrapper, ModelWrapper
```

---

## 3. コマンドライン引数 (CLI)

```bash
python -m intseq_bert.analysis.analyze_mod_spectrum \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --split_type std \
    --split_name test \
    --output_dir results/analysis \
    --model_type intseq \
    --jsonl_path data/oeis/data.jsonl \
    --batch_size 64 \
    --bootstrap_samples 1000
```

### 引数一覧

| 引数 | 型 | 必須 | デフォルト | 説明 |
|------|------|------|-----------|------|
| `--checkpoint` | str | ✅ | - | チェックポイントパス |
| `--split_type` | str | ✅ | - | 分割タイプ (例: `std`, `easy`) |
| `--split_name` | str | | `test` | 分割名 (`train`, `val`, `test`) |
| `--output_dir` | str | ✅ | - | 出力ディレクトリ |
| `--model_type` | str | | `intseq` | モデル種別 (`intseq`, `vanilla`, `ablation`) |
| `--jsonl_path` | str | | `data/oeis/data.jsonl` | OEIS JSONL パス (タグ情報取得用) |
| `--batch_size` | int | | `64` | バッチサイズ |
| `--bootstrap_samples` | int | | `1000` | Bootstrap サンプル数 |
| `--data_root` | str | | `config.DATA_ROOT` | データルート |
| `--device` | str | | `auto` | デバイス指定 (`cuda`, `cpu`, `auto`) |

---

## 4. モデル抽象化

異なるモデルタイプを統一的に扱うため、`ModelWrapper` クラスを導入する。

### 4.1. `ModelWrapper` (Abstract Base)

```python
class ModelWrapper(ABC):
    """抽象モデルラッパー"""
    
    @abstractmethod
    def predict(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                "mag_mu": (B, L),           # Magnitude 予測平均
                "mag_log_var": (B, L),      # Magnitude 不確実性
                "sign_logits": (B, L, 3),   # Sign ロジット
                "mod_logits": (B, L, ~5150) # Modulo 結合ロジット
            }
        """
        pass
    
    @abstractmethod
    def get_mod_log_probs(self, mod_logits: torch.Tensor) -> List[torch.Tensor]:
        """各法 m の log-probability を返す"""
        pass
```

### 4.2. `IntSeqWrapper`

```python
class IntSeqWrapper(ModelWrapper):
    """IntSeqForPreTraining のラッパー"""
    
    def __init__(self, checkpoint_path: str, device: str):
        self.model = IntSeqForPreTraining.from_checkpoint(checkpoint_path)
        self.model.to(device).eval()
        self.device = device
    
    def predict(self, batch: Dict) -> Dict:
        with torch.no_grad():
            outputs = self.model(
                mag_features=batch["mag_inputs"].to(self.device),
                mod_features=batch["mod_inputs"].to(self.device),
                src_key_padding_mask=(batch["attention_mask"] == 0).to(self.device)
            )
        return outputs["predictions"]
    
    def get_mod_log_probs(self, mod_logits: torch.Tensor) -> List[torch.Tensor]:
        split_logits = self.model._split_mod_logits(mod_logits)
        return [F.log_softmax(logits, dim=-1) for logits in split_logits]
```

### 4.3. `VanillaWrapper`

```python
class VanillaWrapper(ModelWrapper):
    """VanillaTransformerForPreTraining のラッパー"""
    
    def __init__(self, checkpoint_path: str, device: str):
        self.model = VanillaTransformerForPreTraining.from_checkpoint(checkpoint_path)
        self.model.to(device).eval()
        self.device = device
    
    def predict(self, batch: Dict) -> Dict:
        with torch.no_grad():
            outputs = self.model(
                token_ids=batch["token_ids"].to(self.device),
                src_key_padding_mask=(batch["attention_mask"] == 0).to(self.device)
            )
        return outputs["predictions"]
```

### 4.4. ファクトリ関数

```python
def create_model_wrapper(
    model_type: str,
    checkpoint_path: str,
    device: str
) -> ModelWrapper:
    """モデルタイプに応じたラッパーを生成"""
    if model_type == "intseq":
        return IntSeqWrapper(checkpoint_path, device)
    elif model_type == "vanilla":
        return VanillaWrapper(checkpoint_path, device)
    elif model_type == "ablation":
        return AblationWrapper(checkpoint_path, device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
```

---

## 5. 評価指標

### 5.1. Normalized Information Gain (NIG)

法によってクラス数（難易度）が異なるため、最大エントロピーで正規化した指標を用いる。

```python
def compute_nig(ce_loss: float, modulus: int) -> float:
    """
    計算式: R(m) = 1.0 - (Loss / log(m))
    
    Args:
        ce_loss: 平均 CrossEntropy 損失
        modulus: 法 m
    
    Returns:
        NIG スコア (1.0 = 完全, 0.0 = ランダム, < 0 = ランダム以下)
    """
    max_entropy = np.log(modulus)
    return 1.0 - (ce_loss / max_entropy)
```

### 5.2. Per-Modulus Metrics

各法について以下を算出:

| 指標 | 計算 | 説明 |
|------|------|------|
| `accuracy` | `(pred == target).mean()` | 分類正解率 (%) |
| `ce_loss` | `CrossEntropy(logits, targets).mean()` | 平均損失 |
| `nig_score` | `1 - ce_loss / log(m)` | 正規化情報利得 |

---

## 6. 処理フロー

### 6.1. メインフロー

```
1. 引数パース & ロギング設定
2. モデルラッパー作成 (create_model_wrapper)
3. データセット & DataLoader 準備
4. 推論ループ (collect_predictions)
5. Per-Modulus 指標計算 (compute_mod_metrics)
6. Bootstrap 信頼区間推定 (bootstrap_ci)
7. タグ別層別分析 (tag_stratified_analysis)
8. 出力ファイル生成
```

### 6.2. Streaming Evaluation (Memory Efficiency)

大規模データセット（例: 30k sequences）において、全ての予測結果（logits: 30000 x 128 x 5150）をメモリに保持すると 70GB+ のメモリを消費し OOM が発生するため、**Streaming Evaluation** 方式を採用する。

1. **バッチ毎の集計:**
   - 予測 (`mod_logits`) と正解データ (`mod_labels`) をバッチ単位で処理する。
   - 各バッチで「損失の合計 (`loss_sum`)」「正解数の合計 (`acc_sum`)」「有効サンプル数 (`counts`)」のみを計算・蓄積する。
   - 巨大な logits テンソルはバッチ処理後に即座に破棄する。

2. **遅延計算:**
   - 全バッチ処理完了後、蓄積された統計量 (Sufficient Statistics) から全体の平均 Loss / Accuracy / NIG を算出する。
   - これにより、データセットサイズに依存せず一定のメモリ消費量で解析が可能となる。

```python
class StreamingEvaluator:
    """
    Evaluates model metrics batch-by-batch to avoid OOM.
    Stores per-sample statistics instead of full logits.
    """
    def process_batch(self, preds: Dict, batch: Dict):
        # バッチ内の統計量を計算し、self.results に追記
        pass

    def finalize(self) -> Dict[str, torch.Tensor]:
        # 全バッチの統計量を結合して返す
        pass
```

> **重要: Collator の分析モード互換性**
>
> 学習時 (`train.py`) と分析時でデータの読み込み方が同じであることを保証する必要がある。
> 特に `OEISCollator` が `mod_labels` (全100法) を返すことを確認する。
>
> **学習時に「ランダムに法を選んで計算」している場合、分析用には「全法計算」モードが必要。**
>
> ```python
> # Collator 初期化時に analysis_mode=True を渡す
> collator = OEISCollator(analysis_mode=True)  # 全法出力モード
> dataloader = DataLoader(dataset, collate_fn=collator, ...)
> ```

### 6.3. `compute_mod_metrics` 関数

各法についてメトリクスを計算。

```python
def compute_mod_metrics(
    mod_logits: torch.Tensor,
    mod_targets: torch.Tensor,
    mask_map: torch.Tensor
) -> pd.DataFrame:
    """
    Returns:
        DataFrame with columns: [modulus, accuracy, ce_loss, nig_score]
    """
    results = []
    split_logits = _split_mod_logits(mod_logits)  # List of (N, L, m)
    
    for i, m in enumerate(config.MOD_RANGE):
        logits_m = split_logits[i]  # (N, L, m)
        targets_m = mod_targets[:, :, i]  # (N, L)
        
        # マスク位置のみ
        valid = mask_map & (targets_m != config.IGNORE_INDEX)
        logits_flat = logits_m[valid]  # (num_valid, m)
        targets_flat = targets_m[valid]  # (num_valid,)
        
        # Accuracy
        preds = logits_flat.argmax(dim=-1)
        accuracy = (preds == targets_flat).float().mean().item() * 100
        
        # CE Loss
        ce_loss = F.cross_entropy(logits_flat, targets_flat).item()
        
        # NIG
        nig = compute_nig(ce_loss, m)
        
        results.append({
            "modulus": m,
            "accuracy": accuracy,
            "ce_loss": ce_loss,
            "nig_score": nig
        })
    
    return pd.DataFrame(results)
```

### 6.4. `bootstrap_ci` 関数

NIG スコアの 95% 信頼区間を推定。

```python
def bootstrap_ci(
    mod_logits: torch.Tensor,
    mod_targets: torch.Tensor,
    mask_map: torch.Tensor,
    n_samples: int = 1000,
    ci_level: float = 0.95
) -> pd.DataFrame:
    """
    Returns:
        DataFrame with columns: [modulus, nig_mean, nig_lower, nig_upper]
    """
    n_sequences = mod_logits.size(0)
    results = {m: [] for m in config.MOD_RANGE}
    
    for _ in tqdm(range(n_samples), desc="Bootstrap"):
        # リサンプリング
        indices = np.random.choice(n_sequences, n_sequences, replace=True)
        sample_logits = mod_logits[indices]
        sample_targets = mod_targets[indices]
        sample_mask = mask_map[indices]
        
        # Per-modulus NIG 計算
        metrics = compute_mod_metrics(sample_logits, sample_targets, sample_mask)
        for _, row in metrics.iterrows():
            results[row["modulus"]].append(row["nig_score"])
    
    # CI 計算
    alpha = (1 - ci_level) / 2
    ci_data = []
    for m in config.MOD_RANGE:
        nig_values = np.array(results[m])
        ci_data.append({
            "modulus": m,
            "nig_mean": nig_values.mean(),
            "nig_lower": np.percentile(nig_values, alpha * 100),
            "nig_upper": np.percentile(nig_values, (1 - alpha) * 100)
        })
    
    return pd.DataFrame(ci_data)
```

---

## 7. タグ別層別分析

### 7.1. OEIS タグ読み込み

```python
def load_oeis_tags(jsonl_path: str) -> Dict[str, List[str]]:
    """
    Returns:
        {oeis_id: [tag1, tag2, ...], ...}
    """
    id_to_tags = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            id_to_tags[record["oeis_id"]] = record.get("keywords", [])
    return id_to_tags
```

### 7.2. `tag_stratified_analysis` 関数

```python
def tag_stratified_analysis(
    mod_logits: torch.Tensor,
    mod_targets: torch.Tensor,
    mask_map: torch.Tensor,
    oeis_ids: List[str],
    id_to_tags: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Returns:
        DataFrame with columns: [tag, count, overall_acc, non_base10_acc, nig_score, top_modulus]
    """
    # タグ → インデックスリスト
    tag_to_indices = defaultdict(list)
    for i, oeis_id in enumerate(oeis_ids):
        for tag in id_to_tags.get(oeis_id, []):
            tag_to_indices[tag].append(i)
    
    results = []
    for tag, indices in tag_to_indices.items():
        if len(indices) < 10:  # 最低10サンプル
            continue
        
        indices_t = torch.tensor(indices)
        tag_logits = mod_logits[indices_t]
        tag_targets = mod_targets[indices_t]
        tag_mask = mask_map[indices_t]
        
        # Per-tag metrics
        metrics = compute_mod_metrics(tag_logits, tag_targets, tag_mask)
        
        # Top modulus
        top_row = metrics.loc[metrics["nig_score"].idxmax()]
        
        results.append({
            "tag": tag,
            "count": len(indices),
            "overall_acc": metrics["accuracy"].mean(),
            "non_base10_acc": _compute_non_base10_acc(metrics),
            "nig_score": metrics["nig_score"].mean(),
            "top_modulus": int(top_row["modulus"])
        })
    
    return pd.DataFrame(results).sort_values("nig_score", ascending=False)


def _compute_non_base10_acc(metrics: pd.DataFrame) -> float:
    """
    Base-10 関連の法 (10, 20, 50, 100) を除外した正解率の平均
    
    **用語整理:**
    - 「自明解 (Trivial Solution)」: |y| < m で剰余計算が不要なケース
    - 「基数依存 (Base-Dependent)」: Mod 10, 100 等の 10進表記に由来する法
    
    この関数は後者（基数依存）を除外する。
    「数論的構造理解」を測定するための指標として使用。
    """
    base10_related_mods = {10, 20, 50, 100}  # Base-10 関連の法
    non_base10 = metrics[~metrics["modulus"].isin(base10_related_mods)]
    return non_base10["accuracy"].mean()
```

---

## 8. 出力ファイル

### 8.1. ディレクトリ構成

```text
results/analysis/mod/
├── mod_spectrum_ranking.csv      # 法別 NIG ランキング
├── mod_spectrum_with_ci.csv      # Bootstrap 信頼区間付き
├── tag_performance.csv           # タグ別層別分析
├── analysis_config.json          # 実行設定
└── figures/
    └── mod_spectrum_bar.png      # (オプション) バーチャート
```

### 8.2. `mod_spectrum_ranking.csv`

```csv
rank,modulus,accuracy,ce_loss,nig_score,interpretation
1,2,92.5,0.104,0.85,Parity (Odd/Even)
2,3,78.2,0.243,0.78,Ternary Pattern
3,10,42.1,0.645,0.72,Base-10 Pattern
...
```

### 8.3. `mod_spectrum_with_ci.csv`

```csv
modulus,nig_mean,nig_lower,nig_upper,accuracy_mean
2,0.85,0.83,0.87,92.5
3,0.78,0.75,0.81,78.2
...
```

### 8.4. `tag_performance.csv`

拡張されたタグ別メトリクス。

| カラム | 説明 |
|--------|------|
| `tag` | タグ名 |
| `count` | サンプル数 |
| `overall_acc` | 全法平均正解率 |
| `non_base10_acc` | Base-10除外平均正解率 |
| `nig_score` | 平均NIGスコア |
| `top_modulus` | 最高スコアの法 |
| `acc_mod_2`, `acc_mod_3`, `acc_mod_5`, `acc_mod_10`, `acc_mod_100` | 主要法の個別正解率 |
| `base10_bias` | 十進法バイアス (`acc_mod_10` - `non_base10_acc`) |
| `top_5_mods_nig` | NIG上位5法 (例: "2(0.85); 3(0.78); ...") |
| `worst_5_mods_nig` | NIG下位5法 |
| `mag_mse` | Magnitude MSE (将来拡張用、現在は N/A) |
| `mag_acc` | Magnitude 正解率 (将来拡張用、現在は N/A) |

```csv
tag,count,overall_acc,non_base10_acc,nig_score,top_modulus,acc_mod_2,acc_mod_3,acc_mod_5,acc_mod_10,acc_mod_100,base10_bias,top_5_mods_nig,worst_5_mods_nig,mag_mse,mag_acc
mult,850,65.2,60.5,0.68,2,92.5,78.2,65.0,42.1,35.0,18.4,"2(0.85); 3(0.78); 4(0.72); 6(0.70); 8(0.68)","98(0.10); 99(0.12); 97(0.15); 101(0.18); 95(0.20)",,
prime,400,55.0,52.0,0.60,2,88.0,70.5,55.0,38.0,28.0,14.0,"2(0.80); 3(0.68); ...",...,,
...
```

### 8.5. `analysis_config.json`

```json
{
  "checkpoint": "checkpoints/intseq_std/best_model.pt",
  "model_type": "intseq",
  "split_type": "std",
  "split_name": "test",
  "n_sequences": 2500,
  "bootstrap_samples": 1000,
  "timestamp": "2026-01-15 12:00:00"
}
```

---

## 9. 実装メモ

### 9.1. 解釈マッピング

高 NIG の法に対して自動的に解釈を付与する。

```python
INTERPRETATION_MAP = {
    # 基本的な周期性
    2: "Parity (Odd/Even)",
    3: "Mod-3 (桁和の剰余)",
    4: "Last 2 Bits",
    5: "Last Digit (Base-5)",
    6: "LCM(2,3) - 2 & 3 Combined",
    7: "Prime",
    8: "Last 3 Bits",
    9: "Mod-9 (Digital Root)",
    
    # Base-10 関連 (表記依存性の指標)
    10: "Base-10 (Last Digit)",
    20: "Base-10 Multiple",
    50: "Base-10 Multiple",
    100: "Base-10 (Last 2 Digits)",
    
    # Highly Composite Numbers
    12: "Highly Composite (LCM(3,4))",
    24: "Highly Composite",
    60: "Sexagesimal Base",
    
    # 大きな素数
    101: "Large Prime (Near 100)",
    97: "Large Prime",
}

def get_interpretation(modulus: int) -> str:
    if modulus in INTERPRETATION_MAP:
        return INTERPRETATION_MAP[modulus]
    elif is_prime(modulus):
        return f"Prime ({modulus})"
    elif modulus % 10 == 0:
        return "Base-10 Multiple"
    elif modulus % 2 == 0:
        return f"Even ({modulus})"
    else:
        return ""
```

### 9.2. エラーハンドリング

| 状況 | 対応 |
|------|------|
| チェックポイント不存在 | `FileNotFoundError` |
| モデルタイプ不明 | `ValueError` |
| データセット空 | `ValueError` |
| CUDA OOM | バッチサイズ自動縮小 or エラー終了 |

---

## 10. 使用例

### IntSeqBERT の分析

```bash
python -m intseq_bert.analysis.analyze_mod_spectrum \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --split_type std \
    --output_dir results/intseq_analysis \
    --model_type intseq
```

### Vanilla Transformer との比較

```bash
# IntSeqBERT
python -m intseq_bert.analysis.analyze_mod_spectrum \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --output_dir results/comparison/intseq \
    --model_type intseq

# Vanilla Transformer
python -m intseq_bert.analysis.analyze_mod_spectrum \
    --checkpoint checkpoints/vanilla_std/best_model.pt \
    --output_dir results/comparison/vanilla \
    --model_type vanilla
```
