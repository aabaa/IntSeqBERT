# `src/intseq_bert/train.py` 実装仕様書

## 1. 概要

本モジュールは、IntSeqBERT モデルの事前学習（Pre-training）を実行するエントリーポイントである。
`preprocess.py` で生成されたデータセット分割と特徴量ファイルを読み込み、**Masked Sequence Modeling** タスクを実行する。
学習プロセスでは、3つのストリーム（Magnitude, Sign, Modulo）の損失バランスを自動調整し、検証時には全ての指標を「パーセント（精度）」で統一して評価する。

## 2. 依存関係

* **モデル:** `src/intseq_bert/models.py` (`IntSeqForPreTraining`)
* **データ:** `src/intseq_bert/loader.py` (`load_dataset`), `src/intseq_bert/collator.py` (`OEISCollator`)
* **設定:** `src/intseq_bert/config.py`
* **ライブラリ:** `torch`, `torch.optim`, `tqdm`, `logging`, `tensorboard` (または `wandb`)

## 3. コマンドライン引数 (CLI)

`argparse` を使用する。

### データ・パス

* `--split_type`: 分割タイプ (例: `std`, `easy`) (必須)
* `--data_root`: データルートディレクトリ (default: `config.DATA_ROOT`)
* `--output_dir`: ログ・チェックポイント保存先 (必須)

### モデル構成 (Config上書き用)

* `--d_model`: 隠れ層次元 (default: `config.D_MODEL`)
* `--nhead`: Head数 (default: `config.NHEAD`)
* `--num_layers`: 層数 (default: `config.NUM_LAYERS`)

### 学習パラメータ

* `--batch_size`: バッチサイズ (default: 32)
* `--lr`: 学習率 (default: 1e-4)
* `--epochs`: エポック数 (default: 20)
* `--accum_steps`: 勾配累積ステップ (default: 1)
* `--weight_decay`: Weight Decay (default: 0.01)
* `--warmup_ratio`: Warmup率 (default: 0.1)
* `--patience`: Early Stopping のエポック数 (default: 5)
* `--num_workers`: DataLoaderワーカー数 (default: 4)
* `--resume`: チェックポイントパス
* `--seed`: ランダムシード (default: `config.SEED`)

---

## 4. Collator 出力からラベルへの変換

`OEISCollator` の出力キーと `IntSeqForPreTraining` が期待するラベル形式は異なるため、学習ループ内で変換を行う。

### Collator 出力

| Collatorキー | 形状 | 内容 |
|--------------|------|------|
| `mag_labels` | `(B, L, 4)` | `[log_val, sign+, sign-, sign0]` |
| `mod_labels` | `(B, L, 100)` | 整数剰余（非マスク位置は `IGNORE_INDEX`） |
| `mask_matrix` | `(B, L)` | Bool、マスク位置 |

### モデルが期待するラベル

| ラベルキー | 形状 | 内容 |
|------------|------|------|
| `mag_targets` | `(B, L)` | `log_val` 値のみ |
| `sign_targets` | `(B, L)` | クラスインデックス (0=Pos, 1=Neg, 2=Zero) |
| `mod_targets` | `(B, L, 100)` | 整数剰余 |
| `mask_map` | `(B, L)` | Bool、マスク位置 |

### 変換ロジック

```python
def prepare_labels(batch: Dict) -> Dict:
    """Collator出力をモデルのlabels形式に変換"""
    mag_labels = batch["mag_labels"]  # (B, L, 4)
    
    # Magnitude: log_val のみ抽出
    mag_targets = mag_labels[:, :, 0]  # (B, L)
    
    # Sign: One-hot → クラスインデックス
    # [sign+, sign-, sign0] → argmax で 0=Pos, 1=Neg, 2=Zero
    sign_one_hot = mag_labels[:, :, 1:4]  # (B, L, 3)
    sign_targets = sign_one_hot.argmax(dim=-1)  # (B, L)
    
    # Modulo: そのまま (非マスク位置の IGNORE_INDEX はモデル側で無視)
    mod_targets = batch["mod_labels"]  # (B, L, 100)
    
    # Mask: そのまま
    mask_map = batch["mask_matrix"]  # (B, L)
    
    return {
        "mag_targets": mag_targets,
        "sign_targets": sign_targets,
        "mod_targets": mod_targets,
        "mask_map": mask_map
    }
```

---

## 5. クラス・関数設計

### 5.1. `evaluate()` 関数 (検証ループ)

検証データセット全体に対して推論を行い、性能評価指標を計算する。
全ての指標を **0〜100% のパーセンテージ** で統一して出力する。

**計算する指標:**

1. **Sign Accuracy (%):**
   * 符号 (Positive, Negative, Zero) の分類正解率。
   * 計算式: `(予測クラス == 正解クラス)[mask_map].mean() * 100`

2. **Mean Modulo Accuracy (%):**
   * 100個の法それぞれの分類正解率を計算し、その平均をとる。
   * 計算式: `Mean( (Mod予測 == Mod正解)[mask_map].float() ) * 100`

3. **Magnitude Accuracy (%):**
   * 回帰タスクを「許容範囲内に入っているか」の正解率として評価。
   * **定義:** `|pred - target| < 0.5` であれば正解。
   * **論理:** `log10` スケールでの `0.5` 誤差 ≈ 元の数値で `√10 ≈ 3.16` 倍の範囲内。
   * 計算式: `(abs(pred - target) < 0.5)[mask_map].float().mean() * 100`

4. **Magnitude MSE:**
   * 回帰精度評価（参考値、低いほど良い）。
   * 計算式: `((pred - target) ** 2)[mask_map].mean()`
   * 補足: 他の Accuracy 指標（高いほど良い）とは方向性が異なる。

**戻り値:**

```python
{
    "val_loss": float,
    "sign_acc": float,   # 0-100
    "mod_acc": float,    # 0-100
    "mag_acc": float,    # 0-100
    "mag_mse": float     # 低いほど良い
}
```

### 5.2. `train()` 関数 (学習ループ)

**セットアップ:**

* Seed固定
* Dataset (`load_dataset`), Collator, DataLoader の初期化
* Model の初期化（GPU転送）
* Optimizer (`AdamW`), Scheduler (`OneCycleLR`)
* Scaler (`torch.amp.GradScaler`)
* Early Stopping カウンタ初期化

**ループ処理 (Epoch単位):**

1. **Training Phase:**
   * モデルを `train()` モードに設定。
   * バッチごとに:
     - `prepare_labels()` でラベル変換
     - Forward → Loss計算 → Backward
   * `accum_steps` ごとに Optimizer Step & Zero Grad。
   * ログ記録: Total Loss と、学習されている重みパラメータ (`s_mag`, `s_sign`, `s_mod`)。

2. **Validation Phase:**
   * モデルを `eval()` モードに設定。
   * `evaluate()` 関数を呼び出し、各 Accuracy (%) を取得。
   * コンソール表示例:
     ```
     Epoch 1: Loss=2.5, Mag Acc=85.2%, Sign Acc=98.1%, Mod Acc=45.3%
     ```

3. **Checkpointing:**
   * `val_loss` が過去最小の場合、`best_model.pt` を保存。
   * 常に `last_checkpoint.pt` を更新。

4. **Early Stopping:**
   * `val_loss` が `patience` エポック連続で改善しない場合、学習を終了。
   * 終了時に `best_model.pt` を最終モデルとして使用。

---

## 6. ロギング設計

**TensorBoard / WandB への記録項目:**

* **Losses:**
  * `train/total_loss`
  * `train/raw_loss_mag` (Gaussian NLL)
  * `train/raw_loss_sign` (CE)
  * `train/raw_loss_mod` (CE)
  * `val/total_loss`

* **Fixed Loss Weights:**
  * 損失重みは固定値: `w_mag = 1.0`, `w_sign = 1.0`, `w_mod = 2.0`
  * Modulo タスクに2倍の重みを与え、周期性情報の学習を促進する

* **Metrics (Accuracy %):**
  * `val/acc_mag`
  * `val/acc_sign`
  * `val/acc_mod`
  * `val/mse_mag`

---

## 7. 実装イメージ

```python
# === Label Preparation (in training loop) ===
batch = next(dataloader)
labels = prepare_labels(batch)

# === Model Forward ===
outputs = model(
    batch["mag_inputs"],
    batch["mod_inputs"],
    src_key_padding_mask=(batch["attention_mask"] == 0),
    labels=labels
)
loss = outputs["loss"]

# === Validation Metrics ===
mask_map = labels["mask_map"]

# 1. Magnitude Accuracy
mag_preds = outputs["predictions"]["mag_mu"]
mag_targets = labels["mag_targets"]
mag_diff = torch.abs(mag_preds - mag_targets)
mag_correct = (mag_diff < 0.5) & mask_map
mag_acc = mag_correct.sum().float() / mask_map.sum() * 100

# 2. Sign Accuracy
sign_logits = outputs["predictions"]["sign_logits"]
sign_preds = sign_logits.argmax(dim=-1)
sign_correct = (sign_preds == labels["sign_targets"]) & mask_map
sign_acc = sign_correct.sum().float() / mask_map.sum() * 100

# 3. Modulo Accuracy (vectorized)
mod_logits = outputs["predictions"]["mod_logits"]
mod_logits_split = model._split_mod_logits(mod_logits)
mod_preds = torch.stack([l.argmax(dim=-1) for l in mod_logits_split], dim=-1)
mod_correct = (mod_preds == labels["mod_targets"]) & mask_map.unsqueeze(-1)
mod_acc = mod_correct.sum().float() / (mask_map.sum() * config.NUM_MODULI) * 100
```

---

## 8. Early Stopping 仕様

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `patience` | 5 | 改善なしで待機するエポック数 |
| `delta` | 0.0 | 改善とみなす最小変化量 |

**動作:**
1. 各エポック終了時に `val_loss` を記録
2. `val_loss` が `best_val_loss - delta` より小さければカウンタをリセット
3. そうでなければカウンタをインクリメント
4. カウンタが `patience` に達したら学習終了

```python
class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
    
    def __call__(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
```

---

## 9. 学習ログ保存仕様

学習を開始するたびに、`output_dir` 直下に以下のファイル群を自動生成・更新する。

### 9.1. ディレクトリ構成

```text
checkpoints/intseq_std/
├── best_model.pt          # 最高精度のモデル重み
├── last_checkpoint.pt     # 最新エポックのモデル重み（再開用）
├── config.json            # 実験設定の完全なスナップショット
├── history.csv            # 全エポックの詳細ログ（Excel/Pandas用）
├── best_metrics.json      # ベストモデル到達時の詳細スコア
└── train.log              # コンソール出力のコピー
```

### 9.2. `config.json` (実験設定)

学習開始時に一度だけ書き出される。後から「あれ、この時 Mod の重み何倍にしたっけ？」とならないための証拠ファイル。

**保存内容:**

* コマンドライン引数 (`args`) のすべて
* `config.py` 内の重要な定数
* 使用したデバイス情報、Gitハッシュ（あれば）
* Python/PyTorch バージョン情報
* データセット統計情報

**例:**

```json
{
  "timestamp": "2026-01-14 10:00:00",
  "args": {
    "lr": 5e-05,
    "batch_size": 32,
    "accum_steps": 2,
    "d_model": 512,
    "num_layers": 8,
    "nhead": 8,
    "patience": 15,
    "split_type": "std"
  },
  "loss_weights": {"mag": 1.0, "sign": 1.0, "mod": 2.0},
  "environment": {
    "python_version": "3.13.3",
    "torch_version": "2.5.0",
    "cuda_version": "12.4",
    "device": "cuda:0",
    "git_hash": "abc1234"
  },
  "data_stats": {
    "train_samples": 45000,
    "val_samples": 2500,
    "test_samples": 2500
  },
  "resume_from": null
}
```

### 9.3. `history.csv` (学習履歴)

エポック終了ごとに1行追記される CSV ファイル。`pandas.read_csv()` で読み込むだけで、すぐに学習曲線のグラフが描ける形式にする。

**カラム定義:**

| カテゴリ | カラム名 | 説明 |
| --- | --- | --- |
| **Meta** | `epoch` | エポック数 (1-based) |
|  | `lr` | そのエポックの最終学習率 |
|  | `time_sec` | エポックにかかった時間（秒） |
|  | `is_best` | そのエポックがベストか (True/False) |
|  | `early_stop_counter` | Early Stopping のカウンタ値 |
| **Loss** | `train_loss` | 学習データ全体の平均損失 |
|  | `val_loss` | 検証データ全体の平均損失 |
| **Magnitude** | `val_mag_acc` | 許容誤差 0.5 以内の正解率 (%) |
|  | `val_mag_mse` | 対数スケールでの平均二乗誤差 |
| **Sign** | `val_sign_acc` | 符号判定の正解率 (%) |
| **Modulo (Summary)** | `val_mod_acc` | 全100法の平均正解率 (%) |
|  | `val_mod_loss` | 剰余タスク単体の損失 |
| **Modulo (Per-Mod)** | `mod_acc_2` 〜 `mod_acc_101` | 各法ごとの正解率 (%, 100カラム) |
| **Weights** | `w_mag`, `w_sign`, `w_mod` | 各タスクの損失重み |

> **Note:** 全100法の精度を個別カラムで保存することにより、後から「mod 7 の学習曲線」「素数法 vs 合成数法」といった分析が可能になる。

**CSVイメージ:**

```csv
epoch,lr,time_sec,is_best,...,val_mod_acc,mod_acc_2,mod_acc_3,...,mod_acc_101,w_mag,w_sign,w_mod
1,5e-5,120.5,True,...,13.24,75.2,62.1,...,8.5,1.0,1.0,2.0
2,5e-5,118.3,True,...,15.74,78.3,65.4,...,9.2,1.0,1.0,2.0
```

### 9.4. コンソール出力 (Representative Mods)

学習中のコンソールには、全100法の精度を表示すると冗長になるため、**代表的な法のみ**を表示する。

**表示対象の法:**

| 法 | 選定理由 |
|-----|---------| 
| `mod 2` | 偶奇（最も基本的な周期性） |
| `mod 3` | 3で割った余り |
| `mod 5` | 小さい素数 |
| `mod 7` | 中程度の素数 |
| `mod 10` | Base-10 バイアス検出用 |
| `mod 100` | 大きな法での学習能力確認 |
| `mod 101` | 大きな素数 |

**コンソール出力例:**

```
Epoch 1 Results:
  Train Loss: 0.6532
  Val Loss:   0.2717
  Mag Acc:    77.55% (MSE: 0.342)
  Sign Acc:   85.21%
  Mod Acc:    13.24% (Mean of 100 mods)
    mod2: 75.2%, mod3: 62.1%, mod5: 48.3%, mod7: 35.6%, mod10: 42.1%, mod100: 8.5%, mod101: 7.9%
```

### 9.5. `best_metrics.json` (ベストモデル素性)

`best_model.pt` が更新されたタイミングでのみ上書き保存される。

**例:**

```json
{
  "best_epoch": 31,
  "val_loss": 0.0892,
  "val_mag_acc": 91.88,
  "val_mag_mse": 0.187,
  "val_sign_acc": 98.45,
  "val_mod_acc": 24.46,
  "val_mod_loss": 0.756,
  "representative_mods": {
    "mod_2": 92.5,
    "mod_3": 78.2,
    "mod_5": 65.1,
    "mod_7": 52.3,
    "mod_10": 48.7,
    "mod_100": 15.2,
    "mod_101": 14.8
  },
  "saved_at": "2026-01-13 17:20:58"
}
```

### 9.6. `*.pt` チェックポイントファイル構造

`best_model.pt` および `last_checkpoint.pt` は以下の辞書構造で保存される。

```python
{
    "epoch": int,                    # エポック番号 (0-based)
    "model_state_dict": OrderedDict, # モデルの重み
    "optimizer_state_dict": dict,    # Optimizer の状態
    "scheduler_state_dict": dict,    # Scheduler の状態（再開用）
    "val_loss": float,               # 検証損失
    "val_metrics": dict,             # 全検証指標
    "config": dict                   # 実験設定 (args)
}
```

### 9.7. `train.log` (コンソールログ)

学習中のコンソール出力をファイルにも保存する。

| 項目 | 値 |
|------|-----|
| フォーマット | `%(asctime)s - %(levelname)s - %(message)s` |
| レベル | `INFO` 以上 |
| エンコーディング | UTF-8 |
| モード | 新規学習時は上書き (`mode="w"`)、Resume時は追記 (`mode="a"`) |

### 9.8. 再開 (Resume) 時の挙動

`--resume` オプションで学習を再開する場合の各ファイルの取り扱い。

| ファイル | 挙動 | 理由 |
|---------|------|------|
| `config.json` | **スキップ** (既存を保持) | 初期設定を維持するため |
| `history.csv` | **追記** | 連続した学習履歴を保持 |
| `best_model.pt` | **条件付き上書き** | 過去ベストを超えた場合のみ更新 |
| `last_checkpoint.pt` | **毎エポック上書き** | 常に最新状態を保持 |
| `best_metrics.json` | **条件付き上書き** | best_model.pt と同期 |
| `train.log` | **追記** | セッション間のログを連結 |

> **Note:** 再開時は `config.json` の `resume_from` フィールドに元のチェックポイントパスを記録する。
