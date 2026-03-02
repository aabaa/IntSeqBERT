# 実験計画

## 解析する数列

### 1. `analyze_attention` に最適な数列（構造・再帰性）

Attentionマップがきれいに（あるいは興味深く）出やすいのは、**「直前の項や特定の過去の項に明確な依存関係がある数列（漸化式）」** や **「幾何学的な構造を持つ数列」** です。

| OEIS ID | 数列名 / 定義 | 選定理由 (Attentionの見どころ) |
| --- | --- | --- |
| **A107413** |  | **【最推奨】** 明確な線形漸化式です。Attentionヘッドが正確に  の位置（直前、3つ前、5つ前）を注目しているかがハッキリ分かります。 |
| **A022433** |  (Hofstadter系) | 相互再帰的な定義です。モデルが複雑な依存関係を追えているかを可視化できます。 |
| **A023622** | Convolution of Lucas numbers... | **畳み込み（Convolution）** は、数列全体の過去の項を広く参照する必要があります。GlobalなAttentionを見るのに適しています。 |
| **A047961** | Coordination sequence (Zeolite) | 結晶構造（コーディネーション数）は通常、有理母関数を持ち、線形漸化式に従います。幾何学的構造をモデルが捉えているか確認できます。 |
| **A106589** | Rauzy substitution () | 置換規則によって生成される数列で、特性多項式を持ちます。記号的なパターンの学習状況が見えます。 |

---

### 2. `analyze_cases` に最適な数列（多様性・難易度）

モデルの総合力を測るため、「単純な計算」「急激な成長」「数論的性質」「カオス/規則性」のバランスを取って選定しました。

| カテゴリ | OEIS ID | 数列名 / 定義 | 選定理由 (診断ポイント) |
| --- | --- | --- | --- |
| **Basic** | **A139249** |  | **【基礎】** 単純な等差数列。これを外すようではモデルに問題があります（Sanity Check）。 |
| **Poly** | **A079414** |  | **【中規模成長】** 4次多項式。Magnitude予測の精度確認に適しています。 |
| **Huge** | **A017408** |  | **【大規模成長】** 値が急激に大きくなります。Scale Invariance（巨大数でも精度が出るか）のテストに最適です。 |
| **Prime** | **A094407** | Primes of form  | **【数論】** 素数の分布規則。ランダムに見える中で「Mod 16」の構造をModulo Streamが捉えているか見ものです。 |
| **Comb** | **A134717** | Odd Motzkin numbers | **【組合せ論】** 有名なMotzkin数の変種。典型的な組合せ爆発をする数列への対応力を見ます。 |
| **CA** | **A284479** | Cellular Automaton Rule 950 | **【アルゴリズム】** セル・オートマトンのような「計算の結果」として現れる数列を推論できるか。 |
| **Logic** | **A196527** | GCD of sums of primes... | **【論理】** 素数和と合成数和のGCDという、複数のステップを経る複雑な定義。推論能力の限界を試せます。 |


## Commands

### Smoke Test
```bash
$ nohup uv run python -m intseq_bert.train --model_type intseq --split_type easy --output_dir checkpoints/intseq_smoke --epochs 2 --batch_size 32 --lr 5e-5 --num_workers 8 &
$ uv run python -m intseq_bert.train --model_type intseq --test_only --test_split test --model_path checkpoints/intseq_smoke/last_checkpoint.pt --split_type easy --output_dir checkpoints/intseq_smoke/
$ uv run python -m intseq_bert.analysis.analyze_magnitude --model_type intseq --checkpoint checkpoints/intseq_smoke/last_checkpoint.pt --split_type easy --output_dir checkpoints/intseq_smoke/analysis/magnitude
$ uv run python -m intseq_bert.analysis.analyze_mod_spectrum --model_type intseq --checkpoint checkpoints/intseq_smoke/last_checkpoint.pt --split_type easy --output_dir checkpoints/intseq_smoke/analysis/mod_spectrum
$ uv run python -m intseq_bert.analysis.analyze_attention --model_type intseq --checkpoint checkpoints/intseq_smoke/last_checkpoint.pt --output_dir checkpoints/intseq_smoke/analysis/attention --oeis_ids A000045,A000290,A033999,A000040,A000142
$ uv run python -m intseq_bert.analysis.analyze_cases --model_type intseq --checkpoint checkpoints/intseq_smoke/last_checkpoint.pt --output_dir checkpoints/intseq_smoke/analysis/cases --oeis_ids A000045,A000290,A033999,A000040,A000142
$ uv run python -m intseq_bert.analysis.analyze_solver --model_type intseq --checkpoint checkpoints/intseq_smoke/last_checkpoint.pt --split_type easy  --output_dir checkpoints/intseq_smoke/analysis/solver --max_samples 10 --top_k 5

$ nohup uv run python -m intseq_bert.train --model_type vanilla --split_type easy --output_dir checkpoints/vanilla_smoke --epochs 2 --batch_size 32 --lr 5e-5 --num_workers 8 &
$ uv run python -m intseq_bert.train --model_type vanilla --test_only --test_split test --model_path checkpoints/vanilla_smoke/last_checkpoint.pt --split_type easy --output_dir checkpoints/vanilla_smoke/
$ uv run python -m intseq_bert.analysis.analyze_magnitude --model_type vanilla --checkpoint checkpoints/vanilla_smoke/last_checkpoint.pt --split_type easy --output_dir checkpoints/vanilla_smoke/analysis/magnitude
$ uv run python -m intseq_bert.analysis.analyze_mod_spectrum --model_type vanilla --checkpoint checkpoints/vanilla_smoke/last_checkpoint.pt --split_type easy --output_dir checkpoints/vanilla_smoke/analysis/mod_spectrum
$ uv run python -m intseq_bert.analysis.analyze_attention --model_type vanilla --checkpoint checkpoints/vanilla_smoke/last_checkpoint.pt --output_dir checkpoints/vanilla_smoke/analysis/attention --oeis_ids A000045,A000290,A033999,A000040,A000142
$ uv run python -m intseq_bert.analysis.analyze_cases --model_type vanilla --checkpoint checkpoints/vanilla_smoke/last_checkpoint.pt --output_dir checkpoints/vanilla_smoke/analysis/cases --oeis_ids A000045,A000290,A033999,A000040,A000142
$ uv run python -m intseq_bert.analysis.analyze_solver --model_type vanilla --checkpoint checkpoints/vanilla_smoke/last_checkpoint.pt --split_type easy  --output_dir checkpoints/vanilla_smoke/analysis/solver --max_samples 10 --top_k 5

$ nohup uv run python -m intseq_bert.train --model_type ablation --split_type easy --output_dir checkpoints/ablation_smoke --epochs 2 --batch_size 32 --lr 5e-5 --num_workers 8 &
$ uv run python -m intseq_bert.train --model_type ablation --test_only --test_split test --model_path checkpoints/ablation_smoke/last_checkpoint.pt --split_type easy --output_dir checkpoints/ablation_smoke/
$ uv run python -m intseq_bert.analysis.analyze_magnitude --model_type ablation --checkpoint checkpoints/ablation_smoke/last_checkpoint.pt --split_type easy --output_dir checkpoints/ablation_smoke/analysis/magnitude
$ uv run python -m intseq_bert.analysis.analyze_mod_spectrum --model_type ablation --checkpoint checkpoints/ablation_smoke/last_checkpoint.pt --split_type easy --output_dir checkpoints/ablation_smoke/analysis/mod_spectrum
$ uv run python -m intseq_bert.analysis.analyze_attention --model_type ablation --checkpoint checkpoints/ablation_smoke/last_checkpoint.pt --output_dir checkpoints/ablation_smoke/analysis/attention --oeis_ids A000045,A000290,A033999,A000040,A000142
$ uv run python -m intseq_bert.analysis.analyze_cases --model_type ablation --checkpoint checkpoints/ablation_smoke/last_checkpoint.pt --output_dir checkpoints/ablation_smoke/analysis/cases --oeis_ids A000045,A000290,A033999,A000040,A000142
$ uv run python -m intseq_bert.analysis.analyze_solver --model_type ablation --checkpoint checkpoints/ablation_smoke/last_checkpoint.pt --split_type easy  --output_dir checkpoints/ablation_smoke/analysis/solver --max_samples 10 --top_k 5
```

### Small (num_layers=6, d_model=256, nhead=4)
#### IntSeq

```bash
$ nohup uv run python -m intseq_bert.train --model_type intseq --split_type std --output_dir checkpoints/small_std_v5/intseq --epochs 200 --patience 200 --batch_size 32 --accum_steps 2 --lr 5e-5 --num_workers 8 --num_layers 6 --d_model 256 --nhead 4 &
$ uv run python -m intseq_bert.analysis.analyze_magnitude --model_type intseq --checkpoint checkpoints/small_std_v5/intseq/last_checkpoint.pt --split_type std --output_dir checkpoints/small_std_v5/intseq/analysis/magnitude
$ uv run python -m intseq_bert.analysis.analyze_mod_spectrum --model_type intseq --checkpoint checkpoints/small_std_v5/intseq/last_checkpoint.pt --split_type std --output_dir checkpoints/small_std_v5/intseq/analysis/mod_spectrum
$ uv run python -m intseq_bert.analysis.analyze_attention --model_type intseq --checkpoint checkpoints/small_std_v5/intseq/last_checkpoint.pt --output_dir checkpoints/small_std_v5/intseq/analysis/attention --oeis_ids A107413,A022433,A023622,A047961,A106589
$ uv run python -m intseq_bert.analysis.analyze_cases --model_type intseq --checkpoint checkpoints/small_std_v5/intseq/last_checkpoint.pt --output_dir checkpoints/small_std_v5/intseq/analysis/cases --oeis_ids A139249,A079414,A017408,A094407,A134717,A284479,A196527
$ uv run python -m intseq_bert.analysis.analyze_solver --model_type intseq --checkpoint checkpoints/small_std_v5/intseq/last_checkpoint.pt --split_type std  --output_dir checkpoints/small_std_v5/intseq/analysis/solver --max_samples 10000 --top_k 10
```

#### Vanilla

```bash
$ nohup uv run python -m intseq_bert.train --model_type vanilla --split_type std --output_dir checkpoints/small_std_v5/vanilla --epochs 200 --patience 200 --batch_size 32 --accum_steps 2 --lr 5e-5 --num_workers 8 --num_layers 6 --d_model 256 --nhead 4 &
$ uv run python -m intseq_bert.analysis.analyze_magnitude --model_type vanilla --checkpoint checkpoints/small_std_v5/vanilla/last_checkpoint.pt --split_type std --output_dir checkpoints/small_std_v5/vanilla/analysis/magnitude
$ uv run python -m intseq_bert.analysis.analyze_mod_spectrum --model_type vanilla --checkpoint checkpoints/small_std_v5/vanilla/last_checkpoint.pt --split_type std --output_dir checkpoints/small_std_v5/vanilla/analysis/mod_spectrum
$ uv run python -m intseq_bert.analysis.analyze_attention --model_type vanilla --checkpoint checkpoints/small_std_v5/vanilla/last_checkpoint.pt --output_dir checkpoints/small_std_v5/vanilla/analysis/attention --oeis_ids A107413,A022433,A023622,A047961,A106589
$ uv run python -m intseq_bert.analysis.analyze_cases --model_type vanilla --checkpoint checkpoints/small_std_v5/vanilla/last_checkpoint.pt --output_dir checkpoints/small_std_v5/vanilla/analysis/cases --oeis_ids A139249,A079414,A017408,A094407,A134717,A284479,A196527
$ uv run python -m intseq_bert.analysis.analyze_solver --model_type vanilla --checkpoint checkpoints/small_std_v5/vanilla/last_checkpoint.pt --split_type std  --output_dir checkpoints/small_std_v5/vanilla/analysis/solver --max_samples 10000 --top_k 10
```


#### Ablation

```bash
$ nohup uv run python -m intseq_bert.train --model_type ablation --split_type std --output_dir checkpoints/small_std_v5/ablation --epochs 200 --patience 200 --batch_size 32 --accum_steps 2 --lr 5e-5 --num_workers 8 --num_layers 6 --d_model 256 --nhead 4 &
$ uv run python -m intseq_bert.analysis.analyze_magnitude --model_type ablation --checkpoint checkpoints/small_std_v5/ablation/last_checkpoint.pt --split_type std --output_dir checkpoints/small_std_v5/ablation/analysis/magnitude
$ uv run python -m intseq_bert.analysis.analyze_mod_spectrum --model_type ablation --checkpoint checkpoints/small_std_v5/ablation/last_checkpoint.pt --split_type std --output_dir checkpoints/small_std_v5/ablation/analysis/mod_spectrum
$ uv run python -m intseq_bert.analysis.analyze_attention --model_type ablation --checkpoint checkpoints/small_std_v5/ablation/last_checkpoint.pt --output_dir checkpoints/small_std_v5/ablation/analysis/attention --oeis_ids A107413,A022433,A023622,A047961,A106589
$ uv run python -m intseq_bert.analysis.analyze_cases --model_type ablation --checkpoint checkpoints/small_std_v5/ablation/last_checkpoint.pt --output_dir checkpoints/small_std_v5/ablation/analysis/cases --oeis_ids A139249,A079414,A017408,A094407,A134717,A284479,A196527
$ uv run python -m intseq_bert.analysis.analyze_solver --model_type ablation --checkpoint checkpoints/small_std_v5/ablation/last_checkpoint.pt --split_type std  --output_dir checkpoints/small_std_v5/ablation/analysis/solver --max_samples 10000 --top_k 10
```

### Middle (num_layers=8, d_model=512, nhead=8)

#### IntSeq

```bash
$ nohup uv run python -m intseq_bert.train --model_type intseq --split_type std --output_dir checkpoints/middle_std_v5/intseq --epochs 200 --patience 200 --batch_size 32 --accum_steps 2 --lr 5e-5 --num_workers 8 --num_layers 8 --d_model 512 --nhead 8 &
$ uv run python -m intseq_bert.analysis.analyze_magnitude --model_type intseq --checkpoint checkpoints/middle_std_v5/intseq/last_checkpoint.pt --split_type std --output_dir checkpoints/middle_std_v5/intseq/analysis/magnitude
$ uv run python -m intseq_bert.analysis.analyze_mod_spectrum --model_type intseq --checkpoint checkpoints/middle_std_v5/intseq/last_checkpoint.pt --split_type std --output_dir checkpoints/middle_std_v5/intseq/analysis/mod_spectrum
$ uv run python -m intseq_bert.analysis.analyze_attention --model_type intseq --checkpoint checkpoints/middle_std_v5/intseq/last_checkpoint.pt --output_dir checkpoints/middle_std_v5/intseq/analysis/attention --oeis_ids A107413,A022433,A023622,A047961,A106589
$ uv run python -m intseq_bert.analysis.analyze_cases --model_type intseq --checkpoint checkpoints/middle_std_v5/intseq/last_checkpoint.pt --output_dir checkpoints/middle_std_v5/intseq/analysis/cases --oeis_ids A139249,A079414,A017408,A094407,A134717,A284479,A196527
$ uv run python -m intseq_bert.analysis.analyze_solver --model_type intseq --checkpoint checkpoints/middle_std_v5/intseq/last_checkpoint.pt --split_type std  --output_dir checkpoints/middle_std_v5/intseq/analysis/solver --max_samples 10000 --top_k 10
```

#### Vanilla

```bash
$ nohup uv run python -m intseq_bert.train --model_type vanilla --split_type std --output_dir checkpoints/middle_std_v5/vanilla --epochs 200 --patience 200 --batch_size 32 --accum_steps 2 --lr 5e-5 --num_workers 8 --num_layers 8 --d_model 512 --nhead 8 &
$ uv run python -m intseq_bert.analysis.analyze_magnitude --model_type vanilla --checkpoint checkpoints/middle_std_v5/vanilla/last_checkpoint.pt --split_type std --output_dir checkpoints/middle_std_v5/vanilla/analysis/magnitude
$ uv run python -m intseq_bert.analysis.analyze_mod_spectrum --model_type vanilla --checkpoint checkpoints/middle_std_v5/vanilla/last_checkpoint.pt --split_type std --output_dir checkpoints/middle_std_v5/vanilla/analysis/mod_spectrum
$ uv run python -m intseq_bert.analysis.analyze_attention --model_type vanilla --checkpoint checkpoints/middle_std_v5/vanilla/last_checkpoint.pt --output_dir checkpoints/middle_std_v5/vanilla/analysis/attention --oeis_ids A107413,A022433,A023622,A047961,A106589
$ uv run python -m intseq_bert.analysis.analyze_cases --model_type vanilla --checkpoint checkpoints/middle_std_v5/vanilla/last_checkpoint.pt --output_dir checkpoints/middle_std_v5/vanilla/analysis/cases --oeis_ids A139249,A079414,A017408,A094407,A134717,A284479,A196527
$ uv run python -m intseq_bert.analysis.analyze_solver --model_type vanilla --checkpoint checkpoints/middle_std_v5/vanilla/last_checkpoint.pt --split_type std  --output_dir checkpoints/middle_std_v5/vanilla/analysis/solver --max_samples 10000 --top_k 10
```

#### Ablation

```bash
$ nohup uv run python -m intseq_bert.train --model_type ablation --split_type std --output_dir checkpoints/middle_std_v5/ablation --epochs 200 --patience 200 --batch_size 32 --accum_steps 2 --lr 5e-5 --num_workers 8 --num_layers 8 --d_model 512 --nhead 8 &
$ uv run python -m intseq_bert.analysis.analyze_magnitude --model_type ablation --checkpoint checkpoints/middle_std_v5/ablation/last_checkpoint.pt --split_type std --output_dir checkpoints/middle_std_v5/ablation/analysis/magnitude
$ uv run python -m intseq_bert.analysis.analyze_mod_spectrum --model_type ablation --checkpoint checkpoints/middle_std_v5/ablation/last_checkpoint.pt --split_type std --output_dir checkpoints/middle_std_v5/ablation/analysis/mod_spectrum
$ uv run python -m intseq_bert.analysis.analyze_attention --model_type ablation --checkpoint checkpoints/middle_std_v5/ablation/last_checkpoint.pt --output_dir checkpoints/middle_std_v5/ablation/analysis/attention --oeis_ids A107413,A022433,A023622,A047961,A106589
$ uv run python -m intseq_bert.analysis.analyze_cases --model_type ablation --checkpoint checkpoints/middle_std_v5/ablation/last_checkpoint.pt --output_dir checkpoints/middle_std_v5/ablation/analysis/cases --oeis_ids A139249,A079414,A017408,A094407,A134717,A284479,A196527
$ uv run python -m intseq_bert.analysis.analyze_solver --model_type ablation --checkpoint checkpoints/middle_std_v5/ablation/last_checkpoint.pt --split_type std  --output_dir checkpoints/middle_std_v5/ablation/analysis/solver --max_samples 10000 --top_k 10
```


### Large (num_layers=12, d_model=768, nhead=12)

#### IntSeq

```bash
$ nohup uv run python -m intseq_bert.train --model_type intseq --split_type std --output_dir checkpoints/large_std_v5/intseq --epochs 200 --patience 200 --batch_size 32 --accum_steps 2 --lr 5e-5 --num_workers 8 --num_layers 12 --d_model 768 --nhead 12 &
$ uv run python -m intseq_bert.analysis.analyze_magnitude --model_type intseq --checkpoint checkpoints/large_std_v5/intseq/last_checkpoint.pt --split_type std --output_dir checkpoints/large_std_v5/intseq/analysis/magnitude
$ uv run python -m intseq_bert.analysis.analyze_mod_spectrum --model_type intseq --checkpoint checkpoints/large_std_v5/intseq/last_checkpoint.pt --split_type std --output_dir checkpoints/large_std_v5/intseq/analysis/mod_spectrum
$ uv run python -m intseq_bert.analysis.analyze_attention --model_type intseq --checkpoint checkpoints/large_std_v5/intseq/last_checkpoint.pt --output_dir checkpoints/large_std_v5/intseq/analysis/attention --oeis_ids A107413,A022433,A023622,A047961,A106589
$ uv run python -m intseq_bert.analysis.analyze_cases --model_type intseq --checkpoint checkpoints/large_std_v5/intseq/last_checkpoint.pt --output_dir checkpoints/large_std_v5/intseq/analysis/cases --oeis_ids A139249,A079414,A017408,A094407,A134717,A284479,A196527
$ uv run python -m intseq_bert.analysis.analyze_solver --model_type intseq --checkpoint checkpoints/large_std_v5/intseq/last_checkpoint.pt --split_type std  --output_dir checkpoints/large_std_v5/intseq/analysis/solver --max_samples 10000 --top_k 10
```

#### Vanilla

```bash
$ nohup uv run python -m intseq_bert.train --model_type vanilla --split_type std --output_dir checkpoints/large_std_v5/vanilla --epochs 200 --patience 200 --batch_size 32 --accum_steps 2 --lr 5e-5 --num_workers 8 --num_layers 12 --d_model 768 --nhead 12 &
$ uv run python -m intseq_bert.analysis.analyze_magnitude --model_type vanilla --checkpoint checkpoints/large_std_v5/vanilla/last_checkpoint.pt --split_type std --output_dir checkpoints/large_std_v5/vanilla/analysis/magnitude
$ uv run python -m intseq_bert.analysis.analyze_mod_spectrum --model_type vanilla --checkpoint checkpoints/large_std_v5/vanilla/last_checkpoint.pt --split_type std --output_dir checkpoints/large_std_v5/vanilla/analysis/mod_spectrum
$ uv run python -m intseq_bert.analysis.analyze_attention --model_type vanilla --checkpoint checkpoints/large_std_v5/vanilla/last_checkpoint.pt --output_dir checkpoints/large_std_v5/vanilla/analysis/attention --oeis_ids A107413,A022433,A023622,A047961,A106589
$ uv run python -m intseq_bert.analysis.analyze_cases --model_type vanilla --checkpoint checkpoints/large_std_v5/vanilla/last_checkpoint.pt --output_dir checkpoints/large_std_v5/vanilla/analysis/cases --oeis_ids A139249,A079414,A017408,A094407,A134717,A284479,A196527
$ uv run python -m intseq_bert.analysis.analyze_solver --model_type vanilla --checkpoint checkpoints/large_std_v5/vanilla/last_checkpoint.pt --split_type std  --output_dir checkpoints/large_std_v5/vanilla/analysis/solver --max_samples 10000 --top_k 10
```

#### Ablation

```bash
$ nohup uv run python -m intseq_bert.train --model_type ablation --split_type std --output_dir checkpoints/large_std_v5/ablation --epochs 200 --patience 200 --batch_size 32 --accum_steps 2 --lr 5e-5 --num_workers 8 --num_layers 12 --d_model 768 --nhead 12 &
$ uv run python -m intseq_bert.analysis.analyze_magnitude --model_type ablation --checkpoint checkpoints/large_std_v5/ablation/last_checkpoint.pt --split_type std --output_dir checkpoints/large_std_v5/ablation/analysis/magnitude
$ uv run python -m intseq_bert.analysis.analyze_mod_spectrum --model_type ablation --checkpoint checkpoints/large_std_v5/ablation/last_checkpoint.pt --split_type std --output_dir checkpoints/large_std_v5/ablation/analysis/mod_spectrum
$ uv run python -m intseq_bert.analysis.analyze_attention --model_type ablation --checkpoint checkpoints/large_std_v5/ablation/last_checkpoint.pt --output_dir checkpoints/large_std_v5/ablation/analysis/attention --oeis_ids A107413,A022433,A023622,A047961,A106589
$ uv run python -m intseq_bert.analysis.analyze_cases --model_type ablation --checkpoint checkpoints/large_std_v5/ablation/last_checkpoint.pt --output_dir checkpoints/large_std_v5/ablation/analysis/cases --oeis_ids A139249,A079414,A017408,A094407,A134717,A284479,A196527
$ uv run python -m intseq_bert.analysis.analyze_solver --model_type ablation --checkpoint checkpoints/large_std_v5/ablation/last_checkpoint.pt --split_type std  --output_dir checkpoints/large_std_v5/ablation/analysis/solver --max_samples 10000 --top_k 10
```
