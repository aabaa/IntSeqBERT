# `src/intseq_bert/analysis/analyze_attention.py` 実装仕様書

## 1. 概要

本スクリプトは、Transformer モデルの **Attention パターン可視化** を専門的に行う。
学習済みモデルが「どの位置に注目して予測を行っているか」を解析し、数列の構造理解を検証する。

> **Note:** このスクリプトは Optional であり、`analyze_cases.py` でも簡易版の Attention 可視化が提供される。
> 本スクリプトはより詳細な層別・ヘッド別分析を行う。

### 主要機能

1. **Layer-wise Attention**: 各 Encoder 層の Attention パターン可視化
2. **Head-wise Analysis**: 個別ヘッドの専門性分析
3. **Aggregated View**: 全層・全ヘッド平均のサマリビュー
4. **Recurrence Detection**: 隣接項への注目パターンの検出

---

## 2. 依存関係

### ライブラリ

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
```

### 内部モジュール

```python
from intseq_bert import config
from intseq_bert.analysis.analyze_mod_spectrum import ModelWrapper, create_model_wrapper
from intseq_bert.analysis.analyze_cases import load_single_sequence
```

---

## 3. コマンドライン引数 (CLI)

```bash
python -m intseq_bert.analysis.analyze_attention \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --oeis_ids A000045,A000142 \
    --output_dir results/attention_analysis \
    --model_type intseq \
    --layer_ids all \
    --head_ids all
```

### 引数一覧

| 引数 | 型 | 必須 | デフォルト | 説明 |
|------|------|------|-----------|------|
| `--checkpoint` | str | ✅ | - | チェックポイントパス |
| `--oeis_ids` | str | ✅ | - | カンマ区切りの OEIS ID リスト |
| `--output_dir` | str | ✅ | - | 出力ディレクトリ |
| `--model_type` | str | | `intseq` | モデル種別 (`intseq`, `vanilla`, `ablation`) |
| `--features_dir` | str | | `data/oeis/features` | 特徴量ディレクトリ |
| `--layer_ids` | str | | `all` | 可視化する層 (`all` or カンマ区切り) |
| `--head_ids` | str | | `all` | 可視化するヘッド (`all` or カンマ区切り) |
| `--device` | str | | `auto` | デバイス指定 |
| `--figsize` | str | | `16,12` | 図のサイズ |
| `--dpi` | int | | `150` | 出力解像度 |

---

## 4. Attention 抽出

### 4.1. `AttentionExtractor` クラス

Forward Hook を用いて全層の Attention Weight を収集する。

```python
class AttentionExtractor:
    """Transformer Encoder の Attention Weight を抽出"""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.attention_weights = []
        self.hooks = []
    
    def register_hooks(self):
        """全 EncoderLayer の self_attn に hook を登録"""
        for layer in self._get_encoder_layers():
            hook = layer.self_attn.register_forward_hook(self._hook_fn)
            self.hooks.append(hook)
    
    def _hook_fn(self, module, input, output):
        """Attention weight を保存 (output[1] に格納されている)"""
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]  # (B, num_heads, L, L)
            if attn_weights is not None:
                self.attention_weights.append(attn_weights.detach().cpu())
    
    def _get_encoder_layers(self):
        """モデルタイプに応じて EncoderLayer を取得"""
        if hasattr(self.model, 'bert'):
            return self.model.bert.encoder.layers
        elif hasattr(self.model, 'encoder'):
            return self.model.encoder.encoder.layers
        else:
            raise ValueError("Cannot find encoder layers")
    
    def remove_hooks(self):
        """登録した hook を削除"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear(self):
        """収集した weight をクリア"""
        self.attention_weights = []
    
    def get_attention_tensor(self) -> torch.Tensor:
        """
        Returns:
            (num_layers, B, num_heads, L, L)
        """
        return torch.stack(self.attention_weights, dim=0)
```

### 4.2. 使用例 (パディングトリミング付き)

OEIS 数列は実際には 30〜50 程度の長さが多く、`MAX_LEN` (512) までパディングされる。
そのままヒートマップを描画すると、左上のごく一部に意味のある模様があり、残り90%が真っ白になる。

**有効長だけを切り取って描画するため、`valid_len` を取得して可視化関数に渡す。**

```python
extractor = AttentionExtractor(model)
extractor.register_hooks()

batch = load_single_sequence(oeis_id, features_dir)

# 有効長を取得 (パディングを除く実際の数列長)
valid_len = batch["attention_mask"].sum().item()

with torch.no_grad():
    outputs = model(**batch)

attention = extractor.get_attention_tensor()  # (num_layers, 1, num_heads, L, L)
extractor.remove_hooks()

# 可視化時に valid_len を渡す
plot_layerwise_attention(attention[:, 0], output_path, oeis_id, valid_len=valid_len)
```

---

## 5. 可視化

### 5.1. Layer-wise Grid (パディングトリミング対応)

```python
def plot_layerwise_attention(
    attention: torch.Tensor,      # (num_layers, num_heads, L, L)
    output_path: Path,
    oeis_id: str,
    valid_len: Optional[int] = None,  # 有効長 (指定時はトリミング)
    layer_ids: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    全層の平均 Attention を Grid 表示
    
    Layout: 2行 x (num_layers/2)列
    
    Note:
        valid_len を指定すると、パディング領域を除外して有効部分のみ描画。
        これにより短い数列 (30〜50項) でも詳細なパターンが確認できる。
    """
    num_layers = attention.size(0)
    if layer_ids is None:
        layer_ids = list(range(num_layers))
    
    # 各層のヘッド平均
    layer_avg = attention.mean(dim=1)  # (num_layers, L, L)
    
    # パディングトリミング: 有効長が指定されていれば切り取る
    if valid_len is not None:
        layer_avg = layer_avg[:, :valid_len, :valid_len]
    
    ncols = min(4, len(layer_ids))
    nrows = (len(layer_ids) + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    for idx, layer_id in enumerate(layer_ids):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        
        im = ax.imshow(layer_avg[layer_id].numpy(), cmap='Blues', vmin=0)
        ax.set_title(f'Layer {layer_id}')
        ax.set_xlabel("Key Pos")
        ax.set_ylabel("Query Pos")
    
    # 未使用の axes を非表示
    for idx in range(len(layer_ids), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')
    
    title = f'Layer-wise Attention: {oeis_id}'
    if valid_len is not None:
        title += f' (L={valid_len})'
    fig.suptitle(title, fontsize=14)
    plt.colorbar(im, ax=axes, shrink=0.6, label='Attention Weight')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

### 5.2. Head-wise Analysis

各ヘッドの「専門性」を分析。

```python
def plot_headwise_attention(
    attention: torch.Tensor,      # (num_layers, num_heads, L, L)
    layer_id: int,
    output_path: Path,
    oeis_id: str,
    valid_len: Optional[int] = None  # パディングトリミング用
):
    """
    指定層の全ヘッドを Grid 表示
    """
    num_heads = attention.size(1)
    layer_attn = attention[layer_id]  # (num_heads, L, L)
    
    # パディングトリミング
    if valid_len is not None:
        layer_attn = layer_attn[:, :valid_len, :valid_len]
    
    ncols = min(4, num_heads)
    nrows = (num_heads + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.atleast_2d(axes)
    
    for head_id in range(num_heads):
        row, col = divmod(head_id, ncols)
        ax = axes[row, col]
        
        im = ax.imshow(layer_attn[head_id].numpy(), cmap='Blues', vmin=0)
        ax.set_title(f'Head {head_id}')
    
    fig.suptitle(f'{oeis_id} - Layer {layer_id} Heads', fontsize=14)
    plt.colorbar(im, ax=axes, shrink=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

### 5.3. Aggregated Summary

```python
def plot_aggregated_attention(
    attention: torch.Tensor,      # (num_layers, num_heads, L, L)
    output_path: Path,
    oeis_id: str,
    valid_len: Optional[int] = None  # パディングトリミング用
):
    """
    全層・全ヘッド平均の Attention + 横プロファイル
    """
    # 全層・全ヘッド平均
    avg_attn = attention.mean(dim=(0, 1)).numpy()  # (L, L)
    
    # パディングトリミング
    if valid_len is not None:
        avg_attn = avg_attn[:valid_len, :valid_len]
    
    L = avg_attn.shape[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Heatmap
    im = axes[0].imshow(avg_attn, cmap='Blues', vmin=0)
    axes[0].set_title(f'Aggregated Attention (L={L})')
    axes[0].set_xlabel("Key Position")
    axes[0].set_ylabel("Query Position")
    plt.colorbar(im, ax=axes[0])
    
    # Panel 2: Horizontal Profile (各 query の最大 attention key)
    max_key_pos = avg_attn.argmax(axis=1)
    relative_pos = max_key_pos - np.arange(L)  # 相対位置
    
    axes[1].bar(range(L), relative_pos, color='steelblue', alpha=0.7)
    axes[1].axhline(y=-1, color='red', linestyle='--', label='n-1 (prev)')
    axes[1].axhline(y=-2, color='orange', linestyle='--', label='n-2')
    axes[1].set_xlabel('Query Position n')
    axes[1].set_ylabel('Relative Key Position (max attn)')
    axes[1].set_title('Attention Focus Offset')
    axes[1].legend()
    
    fig.suptitle(f'Attention Analysis: {oeis_id}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

---

## 6. 再帰パターン検出

### 6.1. 隣接項注目の定量化

Fibonacci のような再帰数列では、`a_n = f(a_{n-1}, a_{n-2})` の関係から、
n-1 および n-2 への Attention が強くなることが期待される。

```python
def analyze_recurrence_pattern(
    attention: torch.Tensor      # (num_layers, num_heads, L, L)
) -> Dict[str, float]:
    """
    再帰パターンへの注目度を定量化
    
    Returns:
        {
            "prev_1_ratio": float,  # n-1 への平均 attention 比率
            "prev_2_ratio": float,  # n-2 への平均 attention 比率
            "diagonal_ratio": float,  # 対角線 (自己注目) 比率
            "total_local_ratio": float,  # |offset| <= 2 の総和比率
        }
    """
    avg_attn = attention.mean(dim=(0, 1)).numpy()  # (L, L)
    L = avg_attn.shape[0]
    
    prev_1_sum = 0
    prev_2_sum = 0
    diag_sum = 0
    total = 0
    
    for q in range(L):
        row_sum = avg_attn[q].sum()
        total += row_sum
        
        # 対角線 (自己注目)
        diag_sum += avg_attn[q, q]
        
        # n-1
        if q >= 1:
            prev_1_sum += avg_attn[q, q - 1]
        
        # n-2
        if q >= 2:
            prev_2_sum += avg_attn[q, q - 2]
    
    # Local ratio (|offset| <= 2)
    local_sum = 0
    for q in range(L):
        for offset in range(-2, 3):
            k = q + offset
            if 0 <= k < L:
                local_sum += avg_attn[q, k]
    
    return {
        "prev_1_ratio": prev_1_sum / total if total > 0 else 0,
        "prev_2_ratio": prev_2_sum / total if total > 0 else 0,
        "diagonal_ratio": diag_sum / total if total > 0 else 0,
        "total_local_ratio": local_sum / total if total > 0 else 0,
    }
```

### 6.2. 期待パターンとの照合

```python
EXPECTED_PATTERNS = {
    "A000045": {"type": "linear_recurrence", "recurrence_depth": 2},  # Fibonacci: a_n = a_{n-1} + a_{n-2}
    "A000142": {"type": "linear_recurrence", "recurrence_depth": 1},  # Factorial: a_n = n * a_{n-1}
    "A000040": {"type": "non_local", "recurrence_depth": None},        # Primes: no local pattern
}

def check_pattern_alignment(
    oeis_id: str,
    recurrence_stats: Dict[str, float]
) -> str:
    """
    期待パターンとの整合性を判定
    
    Returns:
        "ALIGNED" | "MISALIGNED" | "UNKNOWN"
    """
    if oeis_id not in EXPECTED_PATTERNS:
        return "UNKNOWN"
    
    expected = EXPECTED_PATTERNS[oeis_id]
    
    if expected["type"] == "linear_recurrence":
        # 隣接項への注目が強いはず
        if recurrence_stats["total_local_ratio"] > 0.5:
            return "ALIGNED"
        else:
            return "MISALIGNED"
    
    elif expected["type"] == "non_local":
        # 特定パターンがないはず
        if recurrence_stats["total_local_ratio"] < 0.3:
            return "ALIGNED"
        else:
            return "MISALIGNED"
    
    return "UNKNOWN"
```

---

## 7. 処理フロー

### 7.1. メインフロー

```
1. 引数パース & ロギング設定
2. モデルラッパー作成
3. AttentionExtractor 初期化
4. 各 OEIS ID について:
   a. 特徴量読み込み
   b. Attention 抽出
   c. 可視化生成
      - Layer-wise Grid
      - Head-wise (最終層)
      - Aggregated Summary
   d. 再帰パターン分析
5. サマリ CSV 出力
```

### 7.2. Main 関数

```python
def main(args):
    # Setup
    model = create_model_wrapper(args.model_type, args.checkpoint, args.device)
    extractor = AttentionExtractor(model.model)
    extractor.register_hooks()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for oeis_id in args.oeis_ids.split(","):
        logging.info(f"Processing: {oeis_id}")
        
        try:
            batch = load_single_sequence(oeis_id, Path(args.features_dir))
            
            # 有効長を取得 (パディングを除く実際の数列長)
            valid_len = int(batch["attention_mask"].sum().item())
            
            # Forward pass (attention collected via hook)
            extractor.clear()
            _ = model.predict(batch)
            
            attention = extractor.get_attention_tensor()[:, 0]  # Remove batch dim
            # attention: (num_layers, num_heads, L, L)
            
            # Visualizations (有効長でトリミング)
            plot_layerwise_attention(
                attention,
                output_dir / f"{oeis_id}_layerwise.png",
                oeis_id,
                valid_len=valid_len
            )
            
            plot_headwise_attention(
                attention,
                layer_id=-1,  # Last layer
                output_path=output_dir / f"{oeis_id}_heads_last.png",
                oeis_id=oeis_id,
                valid_len=valid_len
            )
            
            plot_aggregated_attention(
                attention,
                output_dir / f"{oeis_id}_aggregated.png",
                oeis_id,
                valid_len=valid_len
            )
            
            # Recurrence analysis
            stats = analyze_recurrence_pattern(attention)
            alignment = check_pattern_alignment(oeis_id, stats)
            
            results.append({
                "oeis_id": oeis_id,
                **stats,
                "pattern_alignment": alignment
            })
            
        except Exception as e:
            logging.error(f"Error processing {oeis_id}: {e}")
            continue
    
    extractor.remove_hooks()
    
    # Save summary
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "attention_summary.csv", index=False)
    logging.info(f"Saved summary to {output_dir / 'attention_summary.csv'}")
```

---

## 8. 出力ファイル

### 8.1. ディレクトリ構成

```text
results/analysis/attention/
├── A000045_layerwise.png        # 全層の Attention Grid
├── A000045_heads_last.png       # 最終層の各ヘッド
├── A000045_aggregated.png       # 集約ビュー + 再帰分析
├── A000142_layerwise.png
├── A000142_heads_last.png
├── A000142_aggregated.png
└── attention_summary.csv        # 再帰パターン統計
```

### 8.2. `attention_summary.csv`

```csv
oeis_id,prev_1_ratio,prev_2_ratio,diagonal_ratio,total_local_ratio,pattern_alignment
A000045,0.25,0.18,0.12,0.65,ALIGNED
A000142,0.35,0.08,0.15,0.62,ALIGNED
A000040,0.10,0.08,0.20,0.42,ALIGNED
```

---

## 9. 制約事項

| 制約 | 説明 |
|------|------|
| PyTorch TransformerEncoder | `need_weights=True` が必要（デフォルトで有効） |
| Vanilla Transformer | 同じ Hook 方式で対応可能 |
| メモリ | 長系列では Attention (L×L) が大きくなる |

---

## 10. 使用例

### 基本使用

```bash
python -m intseq_bert.analysis.analyze_attention \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --oeis_ids A000045,A000142,A000040 \
    --output_dir results/attention
```

### 特定層のみ

```bash
python -m intseq_bert.analysis.analyze_attention \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --oeis_ids A000045 \
    --output_dir results/attention \
    --layer_ids 0,3,5
```
