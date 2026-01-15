# IntSeqBERT 学習ログ保存仕様書

## 1. ディレクトリ構成

学習を開始するたびに、`output_dir` 直下に以下のファイル群を自動生成・更新する。

```text
checkpoints/intseq_std/
├── best_model.pt          # 最高精度のモデル重み
├── last_checkpoint.pt     # 最新エポックのモデル重み（再開用）
├── config.json            # 実験設定の完全なスナップショット
├── history.csv            # 全エポックの詳細ログ（Excel/Pandas用）
├── best_metrics.json      # ベストモデル到達時の詳細スコア
└── train.log              # コンソール出力のコピー
```

---

## 2. ファイル詳細仕様

### 2.1. `config.json` (実験設定)

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

### 2.2. `history.csv` (学習履歴)

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

### 2.3. コンソール出力 (Representative Mods)

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



### 2.4. `best_metrics.json` (ベストモデル素性)

`best_model.pt` が更新されたタイミングでのみ上書き保存される。推論や分析を行う際、モデルをロードしなくても「このモデルの性能」を即座に参照できるようにする。

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

### 2.5. `*.pt` チェックポイントファイル構造

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

### 2.6. `train.log` (コンソールログ)

学習中のコンソール出力をファイルにも保存する。

**仕様:**

| 項目 | 値 |
|------|-----|
| フォーマット | `%(asctime)s - %(levelname)s - %(message)s` |
| レベル | `INFO` 以上 |
| エンコーディング | UTF-8 |
| モード | 新規学習時は上書き (`mode="w"`)、Resume時は追記 (`mode="a"`) |

**実装例:**

```python
mode = "a" if args.resume else "w"
file_handler = logging.FileHandler(output_dir / "train.log", mode=mode)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)
```

---

## 3. 再開 (Resume) 時の挙動

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

## 4. 実装方針 (`train.py` への組み込み)

この仕様を実現するために、`train.py` に以下の機能を組み込むことを想定します。

1. **`TrainingLogger` クラスの導入:**
   * 初期化時に `output_dir` を受け取り、`config.json` を作成。
   * CSVヘッダーが存在しなければ作成。
   * ファイルハンドラを logger に追加。

2. **`log_epoch` メソッド:**
   * エポックごとの辞書データを受け取り、`history.csv` に追記。
   * `is_best` フラグと `early_stop_counter` を自動計算。

3. **`save_checkpoint` メソッドの拡張:**
   * モデル保存 (`torch.save`) と同時に、scheduler の状態も含める。
   * ベスト更新時は `best_metrics.json` もダンプ。

---

## 5. 期待される効果

* **実験管理:** ExcelやSpreadsheetにCSVを貼り付けるだけで、Epoch 31 で崩壊した様子などが可視化できます。
* **論文執筆:** 「最高精度 ○% を達成した」と書く際、`nohup.out` をgrepする必要がなくなり、`best_metrics.json` を見るだけで確定できます。
* **再現性:** パラメータを忘れても `config.json` があれば復活できます。
* **学習再開:** `last_checkpoint.pt` から正確に状態を復元し、学習を継続できます。
