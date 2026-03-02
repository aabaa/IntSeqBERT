# `src/intseq_bert/train.py` 実装仕様書

## 1. 概要

本モジュールは、IntSeqBERT / Vanilla Transformer モデルの事前学習（Pre-training）を実行するエントリーポイントである。
`preprocess.py` で生成されたデータセット分割と特徴量ファイルを読み込み、**Masked Sequence Modeling** タスクを実行する。
学習プロセスでは、3つのストリーム（Magnitude, Sign, Modulo）の損失を**固定重み**（`w_mag=1.0`, `w_sign=1.0`, `w_mod=2.0`）で結合し、検証時には全ての指標を「パーセント（精度）」で統一して評価する。

## 2. 依存関係

* **モデル:**
  - `src/intseq_bert/intseq_models.py` (`IntSeqForPreTraining`)
  - `src/intseq_bert/vanilla_models.py` (`VanillaTransformerForPreTraining`)
  - `src/intseq_bert/ablation_models.py` (`AblationForPreTraining`)
  - `src/intseq_bert/models.py` (再エクスポート用)
* **データ:** `src/intseq_bert/loader.py` (`load_dataset`), `src/intseq_bert/collator.py` (`OEISCollator`)
* **設定:** `src/intseq_bert/config.py`
* **ライブラリ:** `torch`, `torch.optim`, `tqdm`, `logging`, `tensorboard` (または `wandb`)

## 3. コマンドライン引数 (CLI)

`argparse` を使用する。

### データ・パス

* `--split_type`: 分割タイプ (例: `std`, `easy`) (必須)
* `--data_root`: データルートディレクトリ (default: `config.DATA_ROOT`)
* `--output_dir`: ログ・チェックポイント保存先 (必須)

### モデル選択

* `--model_type`: モデル種別 (`intseq`, `vanilla`, `ablation`) (default: `intseq`)
  - `intseq`: IntSeqBERT (Dual Stream + FiLM 融合)
  - `vanilla`: Vanilla Transformer (標準トークン埋め込み)
  - `ablation`: Ablation Model (Magnitude のみ、Modulo 入力なし)

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

`OEISCollator` の出力キーとモデル（`IntSeqForPreTraining` / `VanillaTransformerForPreTraining`）が期待するラベル形式は異なるため、学習ループ内で変換を行う。

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
* Early Stopping カウンタ初期化

> [!IMPORTANT]
> **AMP (Mixed Precision) は無効化されている。**
> OEISデータには `10^{210}` レベルの極端な数値が含まれる（log値 = 210）。
> FP16 の最大値（約65504）を超える中間計算が発生し、勾配がNaN/Infになるため、
> `autocast` と `GradScaler` は使用せず、FP32で訓練する。
>
> 詳細は `spec/intseq_models.md` の数値安定性セクションを参照。

**ループ処理 (Epoch単位):**

1. **Training Phase:**
   * モデルを `train()` モードに設定。
   * バッチごとに:
     - `prepare_labels()` でラベル変換
     - Forward → Loss計算 → Backward
   * `accum_steps` ごとに Optimizer Step & Zero Grad。
   * ログ記録: Total Loss と検証指標（`mag_acc`, `sign_acc`, `mod_acc`, `mag_mse`）。

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

本実装は TensorBoard / WandB 連携ではなく、`TrainingLogger` によるファイル出力を行う。

* `config.json`: 実験設定、環境情報、データ統計
* `history.csv`: エポックごとの `train_loss`, `val_loss`, `val_mag_acc`, `val_sign_acc`, `val_mod_acc`, `val_mag_mse` と法別精度
* `best_metrics.json`: ベストエポック時の主要指標
* `train.log`: コンソール出力の保存

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

---

## 10. テスト専用モード (`--test-only`)

既に学習済みのモデルをテストデータで評価するための専用モードを提供する。
学習ループをスキップし、`history.csv` と同等のフォーマットで評価結果を出力する。

### 10.1. 追加コマンドライン引数

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--test_only` | flag | `False` | テスト専用モードを有効化 |
| `--model_path` | str | `None` | 評価するモデルのパス (`*.pt` ファイル)。`--test_only` 時は必須 |
| `--test_split` | str | `"test"` | 評価に使用する分割 (`train`/`val`/`test`) |
| `--test_output` | str | `None` | 評価結果の出力先 (省略時: `output_dir/test_results.csv`) |

> [!IMPORTANT]
> `--test_only` フラグと `--model_path` 引数は相互依存関係にある。
> - `--test_only` が指定された場合、`--model_path` は必須
> - `--model_path` のみ指定された場合もエラー（`--test_only` を忘れた可能性）

### 10.2. 動作仕様

**1. 引数検証:**

```python
if args.test_only and not args.model_path:
    parser.error("--test_only requires --model_path")
if args.model_path and not args.test_only:
    parser.error("--model_path requires --test_only flag")
```

**2. 処理フロー:**

1. **設定復元:** チェックポイントまたは `config.json` からモデルパラメータを復元
2. モデル読み込み（`model_path` から `model_state_dict` をロード）
3. データセット読み込み（`split_name=args.test_split`）
4. `evaluate()` 関数でデータを評価
5. 結果を CSV 形式で出力

**3. 不要なコンポーネント（スキップされる項目）:**

- Optimizer / Scheduler の初期化
- GradScaler の初期化
- Early Stopping
- 学習ループ全体
- `best_model.pt` / `last_checkpoint.pt` の保存
- `train.log` へのログ出力

### 10.3. 出力形式

#### CSV 出力 (`test_results.csv`)

`history.csv` と互換性のあるカラム構成。学習関連の列は適切なデフォルト値で埋められる。

| カテゴリ | カラム名 | テストモード時の値 |
| --- | --- | --- |
| **Meta** | `epoch` | `0` (テストモードを示す特別な値) |
|  | `lr` | `0.0` |
|  | `time_sec` | 評価にかかった時間（秒） |
|  | `is_best` | `True` |
|  | `early_stop_counter` | `0` |
| **Loss** | `train_loss` | `0.0` (該当なし) |
|  | `val_loss` | テストデータでの損失 |
| **Metrics** | `val_mag_acc` 〜 `mod_acc_101` | テストデータでの評価値 |
| **Weights** | `w_mag`, `w_sign`, `w_mod` | 設定値（参考情報） |

#### コンソール出力

```
========================================
Test-Only Mode Evaluation
========================================
Model: checkpoints/intseq_std/best_model.pt
Split: std (test)
Samples: 2500
----------------------------------------
Test Loss:   0.0892
Mag Acc:     91.88% (MSE: 0.187)
Sign Acc:    98.45%
Mod Acc:     24.46% (Mean of 100 mods)
  mod2: 92.5%, mod3: 78.2%, mod5: 65.1%, mod7: 52.3%, mod10: 48.7%, mod100: 15.2%, mod101: 14.8%
----------------------------------------
Results saved to: checkpoints/intseq_std/test_results.csv
========================================
```

#### JSON 出力 (`test_metrics.json`)

CSV に加えて、詳細なメトリクス JSON も出力する。

```json
{
  "model_path": "checkpoints/intseq_std/best_model.pt",
  "split_type": "std",
  "test_samples": 2500,
  "evaluation_time_sec": 45.2,
  "test_loss": 0.0892,
  "test_mag_acc": 91.88,
  "test_mag_mse": 0.187,
  "test_sign_acc": 98.45,
  "test_mod_acc": 24.46,
  "representative_mods": {
    "mod_2": 92.5,
    "mod_3": 78.2,
    "mod_5": 65.1,
    "mod_7": 52.3,
    "mod_10": 48.7,
    "mod_100": 15.2,
    "mod_101": 14.8
  },
  "all_mod_accuracies": [92.5, 78.2, ...],
  "evaluated_at": "2026-01-16 14:00:00"
}
```

### 10.4. モデルパラメータ復元の優先順位

モデルのハイパーパラメータ（`d_model`, `nhead`, `num_layers`）は、学習時と異なる値で初期化すると `load_state_dict` でサイズ不一致エラーが発生する。
そのため、以下の優先順位で設定を復元する。

| 優先度 | ソース | 説明 |
|--------|--------|------|
| 1 (最優先) | `checkpoint["config"]` | チェックポイント内に保存された設定 |
| 2 | `model_path/../config.json` | 同じディレクトリの `config.json` |
| 3 (最終手段) | コマンドライン引数 | `--d_model` 等のCLI引数 |

```python
def _load_model_config(model_path: Path, checkpoint: Dict, args) -> Dict:
    """
    Load model configuration with fallback priority.
    
    Priority:
    1. checkpoint['config'] (strongest)
    2. config.json in same directory
    3. args (fallback)
    """
    # 1. Try checkpoint config first
    ckpt_config = checkpoint.get("config", {})
    if ckpt_config:
        logger.info("Using config from checkpoint")
        return {
            "d_model": ckpt_config.get("d_model", args.d_model),
            "nhead": ckpt_config.get("nhead", args.nhead),
            "num_layers": ckpt_config.get("num_layers", args.num_layers)
        }
    
    # 2. Try config.json in same directory
    config_path = model_path.parent / "config.json"
    if config_path.exists():
        logger.info(f"Using config from {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)
            saved_args = saved_config.get("args", {})
            return {
                "d_model": saved_args.get("d_model", args.d_model),
                "nhead": saved_args.get("nhead", args.nhead),
                "num_layers": saved_args.get("num_layers", args.num_layers)
            }
    
    # 3. Fallback to args (with warning)
    logger.warning("No saved config found. Using command-line args (may cause size mismatch).")
    return {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers
    }
```

### 10.5. `TrainingLogger` への追加メソッド

CSV ヘッダの重複定義を避けるため、`TrainingLogger` クラスに静的メソッドを追加する。

```python
class TrainingLogger:
    # ... existing code ...
    
    @staticmethod
    def get_csv_headers() -> list:
        """
        Get CSV column headers for history.csv / test_results.csv.
        
        Returns:
            List of column header strings.
        """
        headers = [
            "epoch", "lr", "time_sec", "is_best", "early_stop_counter",
            "train_loss", "val_loss",
            "val_mag_acc", "val_mag_mse", "val_sign_acc",
            "val_mod_acc", "val_mod_loss"
        ]
        
        # Add per-mod accuracy columns (mod_acc_2 through mod_acc_101)
        for m in config.MOD_RANGE:
            headers.append(f"mod_acc_{m}")
        
        # Add weight columns
        headers.extend(["w_mag", "w_sign", "w_mod"])
        
        return headers
```

> [!NOTE]
> `_create_csv_header()` メソッドも内部でこの `get_csv_headers()` を呼び出すようにリファクタリングする。

### 10.6. 実装イメージ

```python
def test_only(args):
    """Test-only mode: evaluate pre-trained model on test/val/train data."""
    import time
    import json
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Header
    logger.info("=" * 40)
    logger.info("Test-Only Mode Evaluation")
    logger.info("=" * 40)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Split: {args.split_type} ({args.test_split})")
    
    # 1. Load Checkpoint
    logger.info("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 2. Restore model config with fallback priority
    model_config = _load_model_config(Path(args.model_path), checkpoint, args)
    
    model = models.IntSeqForPreTraining(
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_layers=model_config["num_layers"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # 3. Load Dataset (using --test_split argument)
    logger.info(f"Loading {args.test_split} dataset...")
    test_dataset = loader.load_dataset(
        split_type=args.split_type,
        split_name=args.test_split,  # train / val / test
        data_root=args.data_root
    )
    logger.info(f"Samples: {len(test_dataset)}")
    
    collator_fn = collator.OEISCollator(mask_prob=config.MASK_PROB)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator_fn,
        pin_memory=True
    )
    
    # 4. Evaluate
    logger.info("-" * 40)
    start_time = time.time()
    test_metrics = evaluate(model, test_loader, device)
    eval_time = time.time() - start_time
    
    # 5. Log Results
    logger.info(f"Loss:        {test_metrics['val_loss']:.4f}")
    logger.info(f"Mag Acc:     {test_metrics['mag_acc']:.2f}% (MSE: {test_metrics['mag_mse']:.4f})")
    logger.info(f"Sign Acc:    {test_metrics['sign_acc']:.2f}%")
    logger.info(f"Mod Acc:     {test_metrics['mod_acc']:.2f}% (Mean of {config.NUM_MODULI} mods)")
    
    # Representative mods
    mod_accs = test_metrics.get("mod_accuracies", [])
    if mod_accs:
        rep_indices = TrainingLogger.get_representative_mod_indices()
        rep_strs = []
        for idx in rep_indices:
            if idx < len(mod_accs):
                mod_val = config.MOD_RANGE[idx]
                rep_strs.append(f"mod{mod_val}: {mod_accs[idx]:.1f}%")
        logger.info(f"  {', '.join(rep_strs)}")
    
    # 6. Save Results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV output (using TrainingLogger.get_csv_headers())
    csv_path = Path(args.test_output) if args.test_output else output_dir / "test_results.csv"
    _save_test_csv(csv_path, test_metrics, eval_time)
    
    # JSON output
    json_path = output_dir / "test_metrics.json"
    _save_test_json(json_path, args, test_metrics, eval_time, len(test_dataset))
    
    logger.info("-" * 40)
    logger.info(f"Results saved to: {csv_path}")
    logger.info("=" * 40)


def _save_test_csv(path: Path, metrics: Dict, time_sec: float) -> None:
    """Save test results in history.csv compatible format."""
    import csv
    
    # Use shared header definition for consistency
    headers = TrainingLogger.get_csv_headers()
    
    row = [
        0,          # epoch (test mode marker)
        0.0,        # lr
        time_sec,
        True,       # is_best
        0,          # early_stop_counter
        0.0,        # train_loss (N/A)
        metrics["val_loss"],
        metrics["mag_acc"],
        metrics["mag_mse"],
        metrics["sign_acc"],
        metrics["mod_acc"],
        metrics["mod_loss"]
    ]
    row.extend(metrics.get("mod_accuracies", [0.0] * config.NUM_MODULI))
    row.extend([config.LOSS_WEIGHT_MAG, config.LOSS_WEIGHT_SIGN, config.LOSS_WEIGHT_MOD])
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(row)


def _save_test_json(
    path: Path, 
    args: argparse.Namespace, 
    metrics: Dict, 
    time_sec: float,
    num_samples: int
) -> None:
    """Save detailed test metrics as JSON."""
    import json
    from datetime import datetime
    
    mod_accs = metrics.get("mod_accuracies", [])
    rep_mods = {}
    for mod in config.REPRESENTATIVE_MODS:
        idx = TrainingLogger._mod_to_index(mod)
        if idx is not None and idx < len(mod_accs):
            rep_mods[f"mod_{mod}"] = mod_accs[idx]
    
    data = {
        "model_path": args.model_path,
        "split_type": args.split_type,
        "test_samples": num_samples,
        "evaluation_time_sec": time_sec,
        "test_loss": metrics["val_loss"],
        "test_mag_acc": metrics["mag_acc"],
        "test_mag_mse": metrics["mag_mse"],
        "test_sign_acc": metrics["sign_acc"],
        "test_mod_acc": metrics["mod_acc"],
        "representative_mods": rep_mods,
        "all_mod_accuracies": mod_accs,
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
```

### 10.7. `main()` 関数の変更

```python
def main():
    parser = argparse.ArgumentParser(description="Train IntSeqBERT Encoder")
    
    # ... existing arguments ...
    
    # Test-Only Mode
    parser.add_argument("--test_only", action="store_true", 
                       help="Run evaluation on test data only (skip training)")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to model checkpoint for test-only mode")
    parser.add_argument("--test_split", type=str, default="test",
                       help="Split name to evaluate (train/val/test)")
    parser.add_argument("--test_output", type=str, default=None,
                       help="Custom output path for test results CSV")
    
    args = parser.parse_args()
    
    # Validation
    if args.test_only and not args.model_path:
        parser.error("--test_only requires --model_path")
    if args.model_path and not args.test_only:
        parser.error("--model_path requires --test_only flag")
    
    # Dispatch
    if args.test_only:
        test_only(args)
    else:
        train(args)
```

### 10.8. 使用例

**基本的な使用方法:**

```bash
# テストデータでモデルを評価（デフォルト）
python -m intseq_bert.train \
  --test_only \
  --model_path checkpoints/intseq_std/best_model.pt \
  --split_type std \
  --output_dir checkpoints/intseq_std/

# 検証データで評価（過学習確認など）
python -m intseq_bert.train \
  --test_only \
  --model_path checkpoints/intseq_std/best_model.pt \
  --split_type std \
  --test_split val \
  --output_dir checkpoints/intseq_std/

# 学習データで評価（過学習具合の確認）
python -m intseq_bert.train \
  --test_only \
  --model_path checkpoints/intseq_std/best_model.pt \
  --split_type std \
  --test_split train \
  --output_dir checkpoints/intseq_std/

# カスタム出力パス
python -m intseq_bert.train \
  --test_only \
  --model_path checkpoints/intseq_std/best_model.pt \
  --split_type std \
  --output_dir results/ \
  --test_output results/my_test_results.csv
```

**異なる分割タイプでの評価:**

```bash
# easy split でトレーニングしたモデルを std split のテストデータで評価
python -m intseq_bert.train \
  --test_only \
  --model_path checkpoints/intseq_easy/best_model.pt \
  --split_type std \
  --test_split test \
  --output_dir cross_eval/easy_on_std/
```
