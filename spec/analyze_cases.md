# `src/intseq_bert/analysis/analyze_cases.py` 実装仕様書

## 1. 概要

本スクリプトは、代表的な数列に対する **Case Study Visualization** を実行する。
モデルの内部挙動（不確実性や周期性パターン）を可視化し、カンニング（過学習）ではなく構造理解が行われているかを確認する。
**IntSeqBERT**, **Vanilla Transformer**, **Ablation (No-Mod)** の全てのモデルタイプに対応する。

### 主要機能

1. **Magnitude & Uncertainty Plot**: 増大軌道と不確実性の可視化
2. **Sign Probability Plot**: 符号クラス確率遷移の可視化
3. **Modulo Spectrum Heatmap**: 周期性指紋の可視化
4. **Multi-Model Comparison**: 複数モデルの同一数列比較

---

## 2. 依存関係

### ライブラリ

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
```

### 内部モジュール

```python
from intseq_bert import config
from intseq_bert.loader import load_dataset
from intseq_bert.analysis.analyze_mod_spectrum import ModelWrapper, create_model_wrapper
```

---

## 3. コマンドライン引数 (CLI)

```bash
python -m intseq_bert.analysis.analyze_cases \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --oeis_ids A000045,A000040,A000290 \
    --output_dir results/case_studies \
    --model_type intseq \
    --features_dir data/oeis/features
```

### 引数一覧

| 引数 | 型 | 必須 | デフォルト | 説明 |
|------|------|------|-----------|------|
| `--checkpoint` | str | ✅ | - | チェックポイントパス |
| `--oeis_ids` | str | ✅ | - | カンマ区切りの OEIS ID リスト |
| `--output_dir` | str | ✅ | - | 出力ディレクトリ |
| `--model_type` | str | | `intseq` | モデル種別 (`intseq`, `vanilla`, `ablation`) |
| `--features_dir` | str | | `data/oeis/features` | 特徴量ディレクトリ |
| `--compare_checkpoints` | str | | - | 比較用チェックポイント (カンマ区切り) |
| `--compare_labels` | str | | - | 比較モデルのラベル (カンマ区切り) |
| `--device` | str | | `auto` | デバイス指定 |
| `--figsize` | str | | `12,10` | 図のサイズ (width,height) |
| `--dpi` | int | | `150` | 出力解像度 |

---

## 4. 対象数列 (Archetypes)

デフォルトの代表数列セット。コマンドライン引数で上書き可能。

```python
DEFAULT_ARCHETYPES = {
    "linear_recurrence": "A000045",   # Fibonacci
    "polynomial": "A000290",          # Squares (n²)
    "sign_oscillation": "A033999",    # Alternating (-1)^n
    "number_theory": "A000040",       # Primes
    "super_growth": "A000142",        # Factorial (n!)
}
```

| カテゴリ | OEIS ID | 数列名 | 検証の狙い |
|---------|---------|--------|-----------|
| Linear Recurrence | A000045 | Fibonacci | 周期性ストライプの確認 |
| Polynomial | A000290 | Squares | 増大カーブへの追従性 |
| Sign Oscillation | A033999 | Alternating | 符号振動パターンの分離 |
| Number Theory | A000040 | Primes | 不確実性の正直さ |
| Super Growth | A000142 | Factorial | 急激な増大への追従 |

---

## 5. 可視化パネル構成

1枚の画像に4つのパネルを配置する 2x2 レイアウト。

### 5.1. Panel 1: Magnitude & Uncertainty

```python
def plot_magnitude_uncertainty(
    ax: plt.Axes,
    positions: np.ndarray,      # (L,)
    ground_truth: np.ndarray,   # (L,) log10(|x|)
    pred_mu: np.ndarray,        # (L,)
    pred_sigma: np.ndarray,     # (L,) = sqrt(exp(log_var))
    mask: np.ndarray            # (L,) 予測対象位置
):
    """
    増大軌道と不確実性の可視化
    
    - 青実線: Ground Truth
    - 赤破線: 予測平均 μ
    - 赤帯: ±2σ の不確実性帯
    """
    ax.plot(positions, ground_truth, 'b-', label='Ground Truth', linewidth=2)
    ax.plot(positions[mask], pred_mu[mask], 'r--', label='Predicted μ', linewidth=1.5)
    
    # Uncertainty band (masked positions only)
    ax.fill_between(
        positions[mask],
        pred_mu[mask] - 2 * pred_sigma[mask],
        pred_mu[mask] + 2 * pred_sigma[mask],
        color='red', alpha=0.2, label='±2σ'
    )
    
    ax.set_xlabel('Position n')
    ax.set_ylabel('log₁₀(|x|)')
    ax.set_title('Magnitude & Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)
```

### 5.2. Panel 2: Sign Probability

```python
def plot_sign_probability(
    ax: plt.Axes,
    positions: np.ndarray,      # (L,)
    sign_probs: np.ndarray,     # (L, 3) [P(+), P(-), P(0)]
    ground_truth_sign: np.ndarray  # (L,) 0=+, 1=-, 2=0
):
    """
    符号クラス確率の積み上げ面グラフ
    
    - 青: Positive
    - 赤: Negative
    - 灰: Zero
    """
    ax.stackplot(
        positions,
        sign_probs[:, 0],  # Positive
        sign_probs[:, 1],  # Negative
        sign_probs[:, 2],  # Zero
        labels=['Positive', 'Negative', 'Zero'],
        colors=['#2196F3', '#F44336', '#9E9E9E'],
        alpha=0.8
    )
    
    # Ground truth markers
    for i, sign in enumerate(ground_truth_sign):
        marker_y = 0.95 if sign == 0 else (0.5 if sign == 1 else 0.05)
        ax.plot(positions[i], marker_y, 'ko', markersize=3)
    
    ax.set_xlabel('Position n')
    ax.set_ylabel('Probability')
    ax.set_title('Sign Probability')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
```

### 5.3. Panel 3: Modulo Spectrum Heatmap

```python
def plot_modulo_heatmap(
    ax: plt.Axes,
    positions: np.ndarray,          # (L,)
    mod_confidences: np.ndarray,    # (L, num_display_mods)
    display_mods: List[int],        # 表示する法のリスト
    ground_truth_mod: np.ndarray    # (L, num_display_mods) 正解剰余
):
    """
    周期性指紋のヒートマップ
    
    - X軸: Position n
    - Y軸: Modulus m
    - 色: 正解クラスに対する予測確率 (Confidence)
    """
    im = ax.imshow(
        mod_confidences.T,
        aspect='auto',
        cmap='RdYlGn',
        vmin=0, vmax=1,
        origin='lower'
    )
    
    ax.set_xlabel('Position n')
    ax.set_ylabel('Modulus m')
    ax.set_title('Modulo Spectrum (Confidence on GT)')
    
    # Y-axis labels
    ax.set_yticks(range(len(display_mods)))
    ax.set_yticklabels(display_mods)
    
    plt.colorbar(im, ax=ax, label='P(correct)')
```

### 5.4. Panel 4: Attention Heatmap (Optional)

```python
def plot_attention_heatmap(
    ax: plt.Axes,
    attention_weights: np.ndarray,  # (L, L) 平均アテンション重み
    positions: np.ndarray
):
    """
    Attention パターンのヒートマップ
    
    - X軸: Key position n'
    - Y軸: Query position n
    - 色: Attention weight
    """
    im = ax.imshow(
        attention_weights,
        aspect='auto',
        cmap='Blues',
        vmin=0
    )
    
    ax.set_xlabel("Key Position n'")
    ax.set_ylabel('Query Position n')
    ax.set_title('Attention Pattern (Avg over heads)')
    plt.colorbar(im, ax=ax, label='Weight')
```

---

## 6. モデル抽象化 (拡張)

`analyze_mod_spectrum.py` で定義した `ModelWrapper` を拡張し、ケーススタディ用の追加機能を提供する。

### 6.1. 追加インターフェース

```python
class ModelWrapper(ABC):
    # ... (既存メソッド)
    
    @abstractmethod
    def predict_with_details(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        ケーススタディ用の詳細予測
        
        Returns:
            {
                "mag_mu": (B, L),
                "mag_log_var": (B, L),
                "sign_logits": (B, L, 3),
                "mod_logits": (B, L, ~5150),
                "attention_weights": Optional[(B, num_heads, L, L)]
            }
        """
        pass
    
    def supports_attention(self) -> bool:
        """Attention 重み取得をサポートするかどうか"""
        return False
```

### 6.2. IntSeqWrapper 拡張

```python
class IntSeqWrapper(ModelWrapper):
    def predict_with_details(self, batch: Dict) -> Dict:
        with torch.no_grad():
            # Attention 重みを取得するためフックを設定
            attention_weights = []
            
            def hook_fn(module, input, output):
                # output[1] は attention weights
                if len(output) > 1 and output[1] is not None:
                    attention_weights.append(output[1])
            
            hooks = []
            for layer in self.model.bert.encoder.layers:
                hooks.append(layer.self_attn.register_forward_hook(hook_fn))
            
            outputs = self.model(
                mag_features=batch["mag_inputs"].to(self.device),
                mod_features=batch["mod_inputs"].to(self.device),
                src_key_padding_mask=(batch["attention_mask"] == 0).to(self.device)
            )
            
            for hook in hooks:
                hook.remove()
            
            result = outputs["predictions"].copy()
            if attention_weights:
                # 全レイヤー、全ヘッドの平均
                result["attention_weights"] = torch.stack(attention_weights).mean(dim=(0, 1))
            
            return result
    
    def supports_attention(self) -> bool:
        return True
```

### 6.3. VanillaWrapper (Magnitude 復元対応)

Vanilla Transformer は数値を「トークンID」として出力するため、Magnitude Plot を描画する際に **トークンID → 数値 → log10 の逆変換** が必要になる。

> **重要:** `[UNK]` (未知語/語彙外の大きな数) を予測した場合、数値に変換できない。
> この制約は IntSeqBERT の優位性を示す強力な証拠となるため、グラフ上で明示的に可視化する。

```python
class VanillaWrapper(ModelWrapper):
    def __init__(self, checkpoint_path: str, device: str):
        self.model = VanillaTransformerForPreTraining.from_checkpoint(checkpoint_path)
        self.model.to(device).eval()
        self.device = device
        self.tokenizer = VanillaTokenizer()  # vocab 復元用
    
    def predict_with_details(self, batch: Dict) -> Dict:
        with torch.no_grad():
            outputs = self.model(
                token_ids=batch["token_ids"].to(self.device),
                src_key_padding_mask=(batch["attention_mask"] == 0).to(self.device)
            )
        
        result = outputs["predictions"].copy()
        
        # トークンID → Magnitude (log10) への復元
        # mag_mu は直接モデルから出力されるが、可視化のため decode も行う
        result["decoded_magnitude"] = self.decode_magnitude(
            outputs["predictions"]["token_preds"]
        )
        
        return result
    
    def decode_magnitude(self, token_ids: torch.Tensor) -> np.ndarray:
        """
        トークンID列を log10(|value|) に変換
        
        UNK/PAD/MASK は np.nan として返し、グラフ上で途切れさせる。
        これにより「大きな数で予測不能になった」ことが視覚的に明確になる。
        """
        values = []
        special_tokens = {config.VANILLA_PAD_TOKEN_ID, 
                          config.VANILLA_MASK_TOKEN_ID, 
                          config.VANILLA_UNK_TOKEN_ID}
        
        for tid in token_ids.cpu().numpy().flatten():
            if tid in special_tokens:
                values.append(np.nan)  # グラフが途切れる
            else:
                # トークンIDから数値を復元 (vocab設計に依存)
                val = self.tokenizer.convert_id_to_value(int(tid))
                if val is not None:
                    values.append(np.log10(abs(val) + 1))
                else:
                    values.append(np.nan)
        
        return np.array(values)
    
    def supports_attention(self) -> bool:
        return True  # Vanilla も Attention 抽出可能
```

---

## 7. 処理フロー

### 7.1. メインフロー

```
1. 引数パース & ロギング設定
2. モデルラッパー作成
3. 各 OEIS ID について:
   a. 特徴量ファイル読み込み
   b. 推論実行 (predict_with_details)
   c. 4パネル図の生成 (generate_case_figure)
   d. PNG 保存
4. サマリ HTML 生成 (任意)
```

### 7.2. `load_single_sequence` 関数 (フォールバック対応)

`.pt` ファイルがない場合、生データからオンザフライで特徴量を生成するフォールバック機能を提供する。
これにより「気になった数列をすぐに可視化する」サイクルが高速化する。

```python
def load_single_sequence(
    oeis_id: str,
    features_dir: Path,
    raw_data_path: Optional[Path] = None,
    jsonl_path: Optional[Path] = None
) -> Dict[str, torch.Tensor]:
    """
    単一数列の特徴量を読み込み、バッチ形式に変換
    
    優先順位:
    1. features_dir/{oeis_id}.pt が存在すれば高速読み込み
    2. jsonl_path から該当レコードを検索してオンザフライ変換
    3. raw_data_path (stripped.txt) から検索してオンザフライ変換
    
    Returns:
        {
            "mag_inputs": (1, L, 5),
            "mod_inputs": (1, L, 200),
            "attention_mask": (1, L),
            "oeis_id": str
        }
    """
    # 1. 既存の .pt ファイルがあれば高速読み込み
    pt_path = features_dir / f"{oeis_id}.pt"
    if pt_path.exists():
        data = torch.load(pt_path)
        return {
            "mag_inputs": data["mag_features"].unsqueeze(0),
            "mod_inputs": data["mod_features"].unsqueeze(0),
            "attention_mask": torch.ones(1, data["mag_features"].size(0)),
            "oeis_id": oeis_id
        }
    
    # 2. JSONL からオンザフライ変換
    if jsonl_path and jsonl_path.exists():
        record = _find_record_in_jsonl(oeis_id, jsonl_path)
        if record:
            return _convert_record_to_features(record)
    
    # 3. Raw text からオンザフライ変換
    if raw_data_path and raw_data_path.exists():
        sequence = _find_sequence_in_raw(oeis_id, raw_data_path)
        if sequence:
            return _convert_sequence_to_features(oeis_id, sequence)
    
    raise FileNotFoundError(
        f"Feature file not found: {pt_path}. "
        f"Provide --jsonl_path or --raw_data_path for on-the-fly conversion."
    )


def _find_record_in_jsonl(oeis_id: str, jsonl_path: Path) -> Optional[Dict]:
    """JSONL から指定 ID のレコードを検索"""
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if record.get("oeis_id") == oeis_id:
                return record
    return None


def _convert_record_to_features(record: Dict) -> Dict[str, torch.Tensor]:
    """JSONL レコードを特徴量テンソルに変換"""
    from intseq_bert.features import extract_features
    
    sequence = record["values"]
    features = extract_features(sequence)
    
    return {
        "mag_inputs": torch.tensor(features["mag_features"]).unsqueeze(0),
        "mod_inputs": torch.tensor(features["mod_features"]).unsqueeze(0),
        "attention_mask": torch.ones(1, len(sequence)),
        "oeis_id": record["oeis_id"]
    }
```

### 7.3. `generate_case_figure` 関数

```python
# 構造別ソートされた display_mods デフォルト値
# - Primes (数論的構造) を先に配置
# - Composites / Base-10 関連を後に配置
# これにより「Mod 10, 100 だけ色が濃い（= Base-10 バイアス）」が視覚的に明確になる
DEFAULT_DISPLAY_MODS = [
    # Primes (Number Theory)
    2, 3, 5, 7, 11, 13,
    # Composites / Highly Composite
    4, 6, 12,
    # Base-10 関連 (バイアス検出用)
    10, 100
]

def generate_case_figure(
    oeis_id: str,
    model: ModelWrapper,
    batch: Dict,
    output_path: Path,
    display_mods: List[int] = None,  # None の場合 DEFAULT_DISPLAY_MODS を使用
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 150
):
    if display_mods is None:
        display_mods = DEFAULT_DISPLAY_MODS
    """
    4パネル構成のケーススタディ図を生成
    """
    # 推論
    preds = model.predict_with_details(batch)
    
    # Ground truth 抽出
    gt_mag = batch["mag_inputs"][0, :, 0].numpy()  # log10(|x|)
    gt_sign = batch["mag_inputs"][0, :, 1:4].argmax(dim=-1).numpy()
    
    # 予測値抽出
    pred_mu = preds["mag_mu"][0].cpu().numpy()
    pred_sigma = np.sqrt(np.exp(preds["mag_log_var"][0].cpu().numpy()))
    sign_probs = F.softmax(preds["sign_logits"][0], dim=-1).cpu().numpy()
    
    # Modulo 信頼度計算
    mod_confidences = _compute_mod_confidences(
        preds["mod_logits"][0],
        batch["mod_inputs"][0],
        display_mods
    )
    
    # Figure 作成
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Case Study: {oeis_id}', fontsize=14, fontweight='bold')
    
    L = gt_mag.shape[0]
    positions = np.arange(L)
    mask = np.ones(L, dtype=bool)  # 全位置を表示
    
    # Panel 1: Magnitude & Uncertainty
    plot_magnitude_uncertainty(axes[0, 0], positions, gt_mag, pred_mu, pred_sigma, mask)
    
    # Panel 2: Sign Probability
    plot_sign_probability(axes[0, 1], positions, sign_probs, gt_sign)
    
    # Panel 3: Modulo Heatmap
    plot_modulo_heatmap(axes[1, 0], positions, mod_confidences, display_mods, None)
    
    # Panel 4: Attention or Summary
    if model.supports_attention() and "attention_weights" in preds:
        plot_attention_heatmap(axes[1, 1], preds["attention_weights"].cpu().numpy(), positions)
    else:
        _plot_summary_metrics(axes[1, 1], preds, batch)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved: {output_path}")
```

### 7.4. `_compute_mod_confidences` 関数

```python
def _compute_mod_confidences(
    mod_logits: torch.Tensor,      # (L, ~5150)
    mod_targets: torch.Tensor,     # (L, 100)
    display_mods: List[int]
) -> np.ndarray:
    """
    各位置における正解クラスへの予測確率（信頼度）を計算
    
    Returns:
        (L, len(display_mods))
    """
    split_logits = _split_mod_logits(mod_logits)  # List of (L, m)
    
    confidences = []
    for m in display_mods:
        idx = config.MOD_RANGE.index(m)
        logits_m = split_logits[idx]  # (L, m)
        probs_m = F.softmax(logits_m, dim=-1)  # (L, m)
        targets_m = mod_targets[:, idx]  # (L,)
        
        # 正解クラスの確率を取得
        conf_m = probs_m.gather(1, targets_m.unsqueeze(1)).squeeze(1)
        confidences.append(conf_m.cpu().numpy())
    
    return np.stack(confidences, axis=1)  # (L, len(display_mods))
```

---

## 8. マルチモデル比較

複数モデルを同一数列で比較する機能。

### 8.1. CLI 使用例

```bash
python -m intseq_bert.analysis.analyze_cases \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --oeis_ids A000045 \
    --output_dir results/comparison \
    --model_type intseq \
    --compare_checkpoints checkpoints/vanilla_std/best_model.pt,checkpoints/ablation_std/best_model.pt \
    --compare_labels IntSeqBERT,Vanilla,Ablation
```

### 8.2. `generate_comparison_figure` 関数

```python
def generate_comparison_figure(
    oeis_id: str,
    models: List[ModelWrapper],
    labels: List[str],
    batch: Dict,
    output_path: Path
):
    """
    複数モデルの Magnitude 予測を1枚の図で比較
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    gt_mag = batch["mag_inputs"][0, :, 0].numpy()
    positions = np.arange(len(gt_mag))
    
    ax.plot(positions, gt_mag, 'k-', label='Ground Truth', linewidth=2)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for model, label, color in zip(models, labels, colors):
        preds = model.predict_with_details(batch)
        pred_mu = preds["mag_mu"][0].cpu().numpy()
        ax.plot(positions, pred_mu, '--', label=label, color=color, linewidth=1.5)
    
    ax.set_xlabel('Position n')
    ax.set_ylabel('log₁₀(|x|)')
    ax.set_title(f'Model Comparison: {oeis_id}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=150)
    plt.close()
```

---

## 9. 出力ファイル

### 9.1. ディレクトリ構成

```text
results/case_studies/
├── A000045_fibonacci.png         # 4パネル図
├── A000040_primes.png
├── A000290_squares.png
├── A033999_alternating.png
├── A000142_factorial.png
├── comparison_A000045.png        # マルチモデル比較 (オプション)
└── index.html                    # サマリページ (オプション)
```

### 9.2. 図のファイル命名規則

```python
def get_output_filename(oeis_id: str, sequence_name: Optional[str] = None) -> str:
    if sequence_name:
        return f"{oeis_id}_{sequence_name.lower().replace(' ', '_')}.png"
    return f"{oeis_id}.png"
```

---

## 10. エラーハンドリング

| 状況 | 対応 |
|------|------|
| 特徴量ファイル不存在 | `FileNotFoundError` + スキップして続行 |
| モデル非対応の出力 | 該当パネルを空白/メッセージ表示 |
| Matplotlib エラー | ログ出力 + 続行 |

---

## 11. 使用例

### 単一モデルのケーススタディ

```bash
# デフォルト5数列
python -m intseq_bert.analysis.analyze_cases \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --oeis_ids A000045,A000040,A000290,A033999,A000142 \
    --output_dir results/case_studies \
    --model_type intseq
```

### カスタム数列

```bash
python -m intseq_bert.analysis.analyze_cases \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --oeis_ids A000001,A000002,A000003 \
    --output_dir results/custom_cases
```

### モデル間比較

```bash
python -m intseq_bert.analysis.analyze_cases \
    --checkpoint checkpoints/intseq_std/best_model.pt \
    --oeis_ids A000045 \
    --output_dir results/comparison \
    --compare_checkpoints checkpoints/vanilla_std/best_model.pt \
    --compare_labels IntSeqBERT,Vanilla
```
