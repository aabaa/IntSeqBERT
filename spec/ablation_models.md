# `src/intseq_bert/ablation_models.py` 実装仕様書

## 1. 概要

**目的:**
IntSeqBERT v3 から「Modulo Stream (幾何学的特徴)」と「FiLM (特徴融合)」を排除したアブレーションモデル (**Ablation Model**) を実装する。

**実験の意図:**
「Magnitude (数値的大きさ)」の情報のみを入力した場合、モデルの性能（特に数論的推論能力）がどの程度低下するかを測定し、提案手法における Modulo Stream の必要性を証明する。

**ファイル構成:**

* **ファイル名:** `src/intseq_bert/ablation_models.py`
* **依存関係:** `base_models.py` (基底クラスを再利用), `config.py`

---

## 2. クラス設計

IntSeqBERT と極力同じ条件にするため、`BaseTransformerModel` や `BaseForPreTraining` を継承し、**Embedding 層のみを差し替えた構成** とします。

### 2.1. `AblationEmbeddings` (Dual Stream の廃止)

Magnitude 特徴量のみを受け取り、MLP 投影で `d_model` 次元に変換します。FiLM は使用しません。

```python
import torch
import torch.nn as nn
from . import config
from .base_models import (
    BasePreTrainedModel,
    BaseForPreTraining,
    ModLogitsMixin,
    PositionalEncoding,
)


class AblationEmbeddings(nn.Module):
    """
    Magnitude feature only embedding.
    No Modulo stream, No FiLM fusion.
    """
    def __init__(
        self,
        d_model: int = config.D_MODEL,
        dropout: float = config.DROPOUT,
        max_len: int = config.MAX_SEQUENCE_LENGTH
    ):
        super().__init__()
        self.d_model = d_model
        
        # Magnitude入力 (5次元) を d_model に投影
        # IntSeqBERTと同様の MLP 構造を使用
        self.mag_proj = nn.Sequential(
            nn.Linear(config.MAG_EXTENDED_DIM, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 共通コンポーネント
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mag_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mag_features: (B, L, MAG_EXTENDED_DIM)
        
        Returns:
            embeddings: (B, L, d_model)
        """
        # 1. Projection (FP32強制で安定性確保)
        with torch.amp.autocast('cuda', enabled=False):
            x = self.mag_proj(mag_features.float())
            
        # 2. Positional Encoding
        x = self.pos_encoding(x)
        
        # 3. LayerNorm & Dropout
        return self.dropout(self.layer_norm(x))
```

### 2.2. `AblationModel` (Backbone)

`AblationEmbeddings` を使用する Transformer Encoder。

```python
class AblationModel(BasePreTrainedModel):
    """
    Ablation model backbone: Transformer Encoder with Magnitude-only embeddings.
    """
    def __init__(
        self,
        d_model: int = config.D_MODEL,
        nhead: int = config.NHEAD,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Ablation Embedding (Magnitude only)
        self.embeddings = AblationEmbeddings(d_model, dropout, config.MAX_SEQUENCE_LENGTH)
        
        # Standard Transformer Encoder (IntSeqBERTと同じ構成)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * config.FEEDFORWARD_MULTIPLIER,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        mag_features: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            mag_features: (B, L, MAG_EXTENDED_DIM)
            src_key_padding_mask: (B, L) BoolTensor, True where padding
        
        Returns:
            last_hidden_state: (B, L, d_model)
        """
        # mod_features は受け取らない (インターフェース互換のため引数を持たない)
        emb = self.embeddings(mag_features)
        return self.encoder(emb, src_key_padding_mask=src_key_padding_mask)
```

### 2.3. `AblationForPreTraining` (Head & Loss)

学習用ラッパー。**重要:** 比較のため、出力ヘッド（診断用 Mod ヘッド含む）は `IntSeqForPreTraining` と全く同じものを使用します。

```python
class AblationForPreTraining(BaseForPreTraining):
    """
    Ablation model for pre-training.
    Uses Magnitude-only input but predicts all tasks (mag, sign, mod).
    """
    def __init__(
        self,
        d_model: int = config.D_MODEL,
        nhead: int = config.NHEAD,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT
    ):
        super().__init__(d_model)
        self.backbone = AblationModel(d_model, nhead, num_layers, dropout)
        self.apply(self._init_weights)

    def forward(
        self,
        mag_features: torch.Tensor,
        mod_features: torch.Tensor,  # ★ 受け取るが無視 (インターフェース互換性)
        src_key_padding_mask: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Args:
            mag_features: (B, L, MAG_EXTENDED_DIM) - 必須
            mod_features: (B, L, MOD_FEATURE_DIM) - **無視される**
            src_key_padding_mask: (B, L) BoolTensor
            labels: Optional dict with mask_map, mag_targets, sign_targets, mod_targets
        
        Returns:
            Dict with predictions and optional loss
        """
        # Backbone には mag_features のみを渡す
        hidden_state = self.backbone(mag_features, src_key_padding_mask)
        
        # Predictions (IntSeqBERTと同じヘッドを使用)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            mag_preds = self.mag_head(hidden_state.float())
            mag_mu = mag_preds[..., 0]
            mag_log_var = mag_preds[..., 1]
        
        sign_logits = self.sign_head(hidden_state)
        unified_mod_logits = self.mod_head(hidden_state)
        
        outputs = {
            "predictions": {
                "mag_mu": mag_mu,
                "mag_log_var": mag_log_var,
                "sign_logits": sign_logits,
                "mod_logits": unified_mod_logits
            }
        }
        
        # Loss Calculation (IntSeqBERTと全く同じロジック)
        if labels is not None:
            mask_map = labels["mask_map"]
            
            # Filter by mask
            target_mag = labels["mag_targets"][mask_map].float()
            pred_mu = mag_mu[mask_map]
            pred_log_var = mag_log_var[mask_map]
            
            target_sign = labels["sign_targets"][mask_map]
            pred_sign = sign_logits[mask_map]
            
            target_mods = labels["mod_targets"][mask_map]
            pred_mods = unified_mod_logits[mask_map]
            
            # Compute losses using base class methods
            loss_mag = self._compute_mag_loss(pred_mu, pred_log_var, target_mag)
            loss_sign = nn.functional.cross_entropy(pred_sign, target_sign)
            loss_mod = self._compute_mod_loss(pred_mods, target_mods)
            
            # Weighted sum
            w_mag, w_sign, w_mod = self.loss_weights
            weighted_loss = w_mag * loss_mag + w_sign * loss_sign + w_mod * loss_mod
            
            outputs["loss"] = weighted_loss
            outputs["loss_breakdown"] = {
                "raw_mag": loss_mag.detach(),
                "raw_sign": loss_sign.detach(),
                "raw_mod": loss_mod.detach(),
                "w_mag": w_mag,
                "w_sign": w_sign,
                "w_mod": w_mod
            }
        
        return outputs
```

---

## 3. 実行環境への統合

### 3.1. `train.py` への追加

`--model_type ablation` を追加し、モデル初期化ロジックに分岐を追加：

```python
# train.py

def create_model(model_type: str, device: str):
    if model_type == "intseq":
        from intseq_bert.intseq_models import IntSeqForPreTraining
        return IntSeqForPreTraining().to(device)
    elif model_type == "vanilla":
        from intseq_bert.vanilla_models import VanillaForPreTraining
        return VanillaForPreTraining().to(device)
    elif model_type == "ablation":
        from intseq_bert.ablation_models import AblationForPreTraining
        return AblationForPreTraining().to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
```

### 3.2. 分析スクリプトへの追加

`analyze_*.py` の `ModelWrapper` に `AblationWrapper` を追加：

```python
# analysis/common.py

class AblationWrapper(ModelWrapper):
    """Wrapper for Ablation model inference."""
    
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        from intseq_bert.ablation_models import AblationForPreTraining
        self.model = AblationForPreTraining.from_checkpoint(checkpoint_path, device)
        self.model.eval()
        self.device = device
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.model(
                mag_features=batch["mag_features"].to(self.device),
                mod_features=batch["mod_features"].to(self.device),  # 無視される
                src_key_padding_mask=batch["padding_mask"].to(self.device)
            )
        return outputs["predictions"]


def create_model_wrapper(model_type: str, checkpoint_path: str, device: str) -> ModelWrapper:
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

## 4. テスト要件

### 4.1. ユニットテスト (`tests/test_ablation_models.py`)

| テスト項目 | 内容 |
|-----------|------|
| `test_ablation_embeddings_forward` | `AblationEmbeddings` が正しい形状を出力するか |
| `test_ablation_model_forward` | `AblationModel` が `(B, L, d_model)` を返すか |
| `test_ablation_for_pretraining_forward` | 予測出力に `mag_mu`, `sign_logits`, `mod_logits` が含まれるか |
| `test_ablation_loss_computation` | `labels` 指定時に `loss` が返されるか |
| `test_mod_features_ignored` | `mod_features` を変更しても出力が同じか |
| `test_from_checkpoint` | チェックポイントからのロードが成功するか |

---

## 5. 期待される実験結果（仮説）

このモデルを実装・学習させることで、以下の結果が予想されます。これを論文の主張に使用します。

| 指標 | 予想 | 根拠 |
|-----|------|------|
| **Mod Accuracy** | 壊滅的低下 (ランダム付近) | 入力に Modulo 情報がないため、余り予測は推測不可能 |
| **Sign Accuracy** | 低下 (偶奇性の学習困難) | 偶奇性は `mod 2` の情報が重要だが、入力から消失 |
| **Magnitude MAE** | IntSeqBERT と同等 | 大きさの情報は直接入力されているため |
| **Solver Accuracy** | 低下 | 周期的パターンの把握が困難 |

この実装により、「大きさの予測」と「数論的性質の理解」が分離された能力であることを実証できます。

---

## 6. 出力ファイル構成

```text
checkpoints/ablation_std/
├── best_model.pt         # ベストモデル
├── last_checkpoint.pt    # 最終チェックポイント
└── training_log.csv      # 学習ログ

results/analysis/ablation/
├── overall_metrics.csv   # IntSeqBERT との比較用
└── figures/
    └── comparison.png    # 3モデル (IntSeq/Vanilla/Ablation) 比較
```
