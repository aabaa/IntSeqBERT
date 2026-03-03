# 5. 実験

<!-- 目標: ~4ページ (~1800語 + 表1 + 表2 + 図2) -->

## 5.1 主要結果

表~\ref{tab:main} に全 3 モデルサイズ・全バリアントのテスト性能を示す。
IntSeqBERT は全スケール・全指標において両ベースラインを一貫して上回る。

<!-- 表1 -->
**表1.** テスト結果。Mag Acc = Accuracy$_{0.5}$（%）、Sign Acc（%）、MMA = 平均 Modulo 精度（%）。各サイズグループ内の最高値を **太字** で示す。

| サイズ | モデル      | Mag Acc  | Sign Acc | MMA    |
|--------|------------|--------:|---------:|-------:|
| Small  | **IntSeq** | **94.73** | **97.78** | **40.43** |
| Small  | Vanilla    | 85.73   | 96.91    | 36.21  |
| Small  | Ablation   | 93.72   | 97.39    | 25.97  |
| Middle | **IntSeq** | **95.71** | **98.34** | **46.88** |
| Middle | Vanilla    | 87.37   | 97.42    | 42.53  |
| Middle | Ablation   | 92.45   | 97.90    | 31.93  |
| Large  | **IntSeq** | **95.85** | **98.54** | **50.38** |
| Large  | Vanilla    | 86.97   | 97.66    | 45.85  |
| Large  | Ablation   | 89.70   | 98.29    | 35.22  |


Large スケールの IntSeqBERT は Vanilla と比較して Mag Acc で **+8.9pt**、MMA で **+4.5pt** の向上を達成した。
Modulo ストリームを取り除いたアブレーションモデルは最大の MMA 低下（Large で −15.2pt）を示し、算術的周期性特徴量の寄与を直接定量化している。
注目すべき点として、アブレーションモデルは競争力のある Mag Acc を維持しており、符号・Modulo 情報が Magnitude 回帰への寄与は限定的である一方、Modulo 予測には不可欠であることが示唆される。


**学習曲線。** 図~\ref{fig:learning_curve} に Large モデル（全バリアント）の検証損失の推移を示す。Large IntSeqBERT は Epoch 1 で Val Loss = 2.17 から始まり、Epoch 10 で 1.70、Epoch 50 で 1.15、Epoch 100 で 1.05、Epoch 200 で 1.01 まで継続的に低下した。学習損失と検証損失はほぼ一致して推移しており（Epoch 200: Train 1.00 / Val 1.01）、過学習は観察されない。Vanilla（最終 Val Loss = 1.77）は全期間を通じて IntSeqBERT（1.01）および Ablation（1.39）より高い損失を示し、モデルサイズが等しくても Modulo ストリームの有無が学習収束に本質的な差をもたらすことが確認された。

<!-- 図4（experiment/cicm2026/fig4_learning_curves.{pdf,png}）
     \begin{figure}[t]
       \centering
       \includegraphics[width=\linewidth]{figures/fig4_learning_curves}
       \caption{全スケール（Small / Middle / Large）・全バリアントの検証損失の学習曲線。
                IntSeqBERT（実線青）は一貫して Vanilla（破線橙）・Ablation（一点鎖線緑）を下回り、
                Large スケールでは Epoch 200 時点で Val Loss = 1.01 に収束する。
                学習損失と検証損失の推移（過学習確認）については
                補足図~\ref{fig:train_val_curve} を参照。}
       \label{fig:learning_curve}
     \end{figure}

     補足図4b（experiment/cicm2026/fig4b_train_val_curves.{pdf,png}）
     \begin{figure}[t]
       \centering
       \includegraphics[width=\linewidth]{figures/fig4b_train_val_curves}
       \caption{学習損失（点線）と検証損失（実線）の比較。各パネルはモデルバリアント、
                色は Small（緑）/ Middle（橙）/ Large（青）を表す。
                全バリアント・全スケールで Train と Val がほぼ一致して推移しており、
                過学習が生じていないことを示す。}
       \label{fig:train_val_curve}
     \end{figure} -->

## 5.2 Magnitude 予測

**スケール別解析。** 表~\ref{tab:scale} に Large モデルを用いたテストの Magnitude バケット別 MSE を示す。ここでバケットは $u=\log_{10}(|x|)$（$x=0$ のとき $u=0$）で定義する。

<!-- 表2 -->
**表2.** テストにおけるスケール別 MSE（Large モデル）。値が小さいほど良い。

| バケット       | IntSeq  | Vanilla  | Ablation |
|----------------|--------:|---------:|---------:|
| Small ($u<2$)           | 0.111 | 0.138  | **0.103**    |
| Medium ($2\le u<5$)     | **0.051** | 0.071  | 0.116    |
| Large ($5\le u<20$)     | **0.162** | 2.100  | 0.381    |
| Huge ($20\le u<50$)     | **2.082** | 22.73  | 5.021    |
| Astronomical ($u\ge50$) | **110.4** | 840.0 | 532.6    |

<!-- 注: スケール別データは analysis/magnitude/scale_wise_metrics.csv（Large モデル）より -->

Vanilla モデルは Large バケットで壊滅的な精度低下を示し（MSE = 2.10、IntSeq の 13 倍）、これは語彙外の整数がすべて `UNK` トークンに吸収されることに起因する。
IntSeqBERT は Small を除く全バケットで最良の MSE を達成し、特に Medium 以上で優位性が顕著である。
アブレーションモデルは Medium バケットでは Modulo コンテキストの欠如により精度が低下する（MSE = 0.116 対 IntSeq 0.051）。
注目すべきは Huge・Astronomical バケットにおけるアブレーションの大幅な劣化である（Huge: MSE 5.02 対 IntSeq 2.08、Astronomical: MSE 533 対 IntSeq 110）。
小さな整数ではモジュラス情報がなくても Magnitude 推定は安定するが、巨大な整数では Magnitude ストリーム単体の不確かさが極めて大きくなり、Modulo ストリームによる FiLM 変調が算術的な構造制約として Magnitude ヘッドの推定を引き締める役割を果たしていると解釈できる。

図~\ref{fig:scatter} に Large モデル 3 バリアントの予測 Magnitude 対真の Magnitude 散布図を示す。
IntSeqBERT は決定係数 $R^2 = 0.988$ と最も高い線形一致度を達成し、Vanilla（$R^2 = 0.943$）と比較して Large・Huge バケット（黄橙・赤マーカー）での対角線からの乖離が著しく小さい。
これは表~\ref{tab:scale} の MSE 結果を視覚的に裏付けるものであり、語彙外の整数を `UNK` で吸収するトークン化ベースラインが高スケール領域で予測精度を失う様子が散布図上でも明瞭に確認される。

<!-- 図5（experiment/cicm2026/fig5_magnitude_scatter.{pdf,png}）
     \begin{figure}[t]
       \centering
       \includegraphics[width=\linewidth]{figures/fig5_magnitude_scatter}
       \caption{Large モデルの予測 Magnitude 対真の Magnitude（$\log_{10}$ スケール）。
                各点はバケットで色分けされる（Small=青丸、Medium=緑丸、Large=黄橙四角、
                Huge=赤三角、Astronomical=紫菱形）。
                IntSeqBERT は $R^2 = 0.988$、Vanilla は $R^2 = 0.943$ であり、
                Vanilla では Large 以上のバケットで顕著なばらつきが見られる。}
       \label{fig:scatter}
     \end{figure} -->

**校正精度（Calibration）。** 図~\ref{fig:calibration} に Large モデルの不確かさ校正曲線を示す。
X 軸は予測不確かさ $\sigma$（ビン平均）、Y 軸は同ビン内の実際の RMSE であり、完全校正では $y = x$ に乗る。
Vanilla モデルは σ レンジが 0.007 〜 46.7 と極端に広がり、ECE = 5.36 と著しく校正ずれしている。
これは低スケール領域で $\sigma \approx 0$ にもかかわらず RMSE が 1〜2（過信）、高スケール領域では逆に $\sigma$ が爆発的に拡大する（過大推定）という二重の校正失敗を示す。
一方 IntSeqBERT は ECE = 0.65 と大幅に良好であり、Ablation（ECE = 0.66）と同等である。
Modulo ストリームを加えることで異分散 Magnitude ヘッドの不確かさ推定が改善されるが、校正精度そのものは Modulo 有無よりも Vanilla との構造的差異（連続 vs. トークン化）に強く依存することがわかる。

<!-- 図6（experiment/cicm2026/fig6_calibration.{pdf,png}）
     \begin{figure}[t]
       \centering
       \includegraphics[width=\linewidth]{figures/fig6_calibration}
       \caption{Large モデルの不確かさ校正曲線（X 軸：予測 $\sigma$（ログスケール）、
                Y 軸：実際の RMSE）。赤領域（対角線上側）は過信、青領域は過大推定を示す。
                IntSeqBERT（ECE = 0.648）と Ablation（ECE = 0.662）は比較的対角線近傍に
                位置するのに対し、Vanilla（ECE = 5.360）は低 $\sigma$ 域で深刻な過信を示す。}
       \label{fig:calibration}
     \end{figure} -->

## 5.3 Modulo スペクトル解析

Large IntSeqBERT を用いて各法 $m \in \{2, \ldots, 101\}$ に対する正規化情報利得（NIG）を評価する。

図~\ref{fig:nig_spectrum} に Large モデル 3 バリアントの NIG スペクトル（$m = 2, \ldots, 101$）を示す。
IntSeqBERT（実線青）は全域で Vanilla（破線橙）・Ablation（一点鎖線緑）を上回り、
素数法（グレー背景）と合成数法のコントラストが視覚的にも明瞭である。

<!-- 図2（experiment/cicm2026/fig2_nig_spectrum.{pdf,png}）
     \begin{figure}[t]
       \centering
       \includegraphics[width=\linewidth]{figures/fig2_nig_spectrum}
       \caption{法 $m=2,\ldots,101$ に対する NIG スペクトル（Large モデル）。
                灰色背景は素数法を示す。IntSeqBERT の 95\% CI（水色帯）は
                bootstrapping により算出。}
       \label{fig:nig_spectrum}
     \end{figure} -->

**発見1: NIG はオイラーのトーシェント比 $\varphi(m)/m$ と強い負の相関を示す。** NIG と $\varphi(m)/m = \prod_{p \mid m}(1-1/p)$ の間には Pearson $r = -0.851$（$p < 10^{-28}$）という極めて強い負の相関が確認される（図~\ref{fig:nig_phi}）。
小さな素因数を多く持つ合成数を法とするほど NIG が高く、これは中国剰余定理（CRT）による集約効果として解釈できる：法 $m$ が複数の小さな法 $m_1, m_2, \ldots$ の公倍数であれば、$x \bmod m$ はそれら全法の情報を一括して保持するからである。
全モデル・全スケールで最高 NIG を達成するのは $m = 96 = 2^5 \times 3$（$\varphi(96)/96 = 1/3$、Large IntSeq で NIG = 0.629、95% CI [0.622, 0.634]）であり、これは同じトーシェント比 $1/3$ を持つ $m \in \{12, 24, 48, 72\}$ の中で最大の $m$ として最も広い値域を区別できることとも整合する。
ただし $m = 2$（素数で $\varphi(2)/2 = 0.5$ と比較的高いトーシェント比を持つ）が NIG = 0.628 と第 2 位を記録するのは例外的であり、パリティ（奇偶）が OEIS 数列のほぼ全体に普遍的に現れるコーパス固有の偏りを反映している。

<!-- 図2b（experiment/cicm2026/fig2b_nig_vs_phi.{pdf,png}）
     \begin{figure}[t]
       \centering
       \includegraphics[width=0.72\linewidth]{figures/fig2b_nig_vs_phi}
       \caption{NIG 対 Euler トーシェント比 $\varphi(m)/m$（Large IntSeqBERT）。
                合成数法（青丸、色の濃さは $m$ の値）と素数法（赤三角）を色分けする。
                回帰直線（灰色破線）は Pearson $r=-0.851$（$p < 10^{-28}$）を示す。
                $m=2$（パリティ）・$m=60$（バビロニア数）・$m=96$（高度合成数）を注記。}
       \label{fig:nig_phi}
     \end{figure} -->

**発見2: パリティ（mod 2）精度がモデルを層別化する。** Large スケールにおける mod-2 精度は IntSeq（85.65%）・Vanilla（81.40%）・Ablation（72.13%）であり、Modulo ストリームの有無が最も明確に現れる単一モジュラスの指標となっている。ストリームを除去すると 13.5pt の低下が生じ、トークン化ベースラインは IntSeq から 4.2pt 低い。

**代表的なモジュラス精度（Large モデル）：**

| 法    | IntSeq | Vanilla | Ablation | 解釈                 |
|------:|-------:|--------:|---------:|----------------------|
| 2     | 85.65  | 81.40   | 72.13    | パリティ             |
| 3     | 72.62  | 65.22   | 53.72    | 三進剰余             |
| 5     | 60.37  | 50.07   | 42.63    | 最下位桁のパリティ   |
| 10    | 58.38  | 49.25   | 39.47    | 十進最下位桁         |
| 60    | 53.97  | 47.87   | 35.12    | バビロニア数（高度合成数） |
| 96    | 51.82  | 47.29   | 34.44    | 高度合成数           |
| 100   | 48.51  | 45.60   | 33.51    | 百分剰余             |

## 5.4 Solver による次項予測

Solver モジュールはモデルの Magnitude・符号・Modulo 予測から候補整数を再構成し、尤度でランキングする。
10,000 サンプルのテストデータで完全一致精度を評価する。
各サンプルは OEIS の 1 数列に対応し、その系列の**最後の項**を予測ターゲットとする（先行する項をすべて文脈として入力）。
テストデータの系列長は最小 10 項・中央値 36 項（平均 42.5 項）であり、Solver には常に 9 項以上の先行文脈が与えられる。

**有効率**は Solver が少なくとも 1 件の候補整数を返したサンプルの割合である。Vanilla はトークン語彙の softmax から常に候補を返すため常に 100% となる。IntSeq・Ablation では探索範囲内に有効な候補が得られない場合（「候補なし」モード、Large IntSeq で 13.4%）に候補を返さないため、有効率が下がる。

**表3.** Solver 評価：Top-1・Top-10 完全一致精度（%）と有効候補率。

| サイズ | モデル      | Top-1  | Top-10 | 符号精度 | 有効率（%） |
|--------|------------|-------:|-------:|---------:|------------|
| Small  | **IntSeq** | **14.05** | **21.00** | **98.73** | 90.59 |
| Small  | Vanilla    | 2.43   | 3.24   | 92.92    | 100.0 |
| Small  | Ablation   | 7.42   | 17.33  | 98.50    | 90.17 |
| Middle | **IntSeq** | **17.02** | **22.62** | **99.02** | 86.31 |
| Middle | Vanilla    | 2.43   | 3.41   | 92.71    | 100.0 |
| Middle | Ablation   | 9.88   | 20.52  | 98.74    | 90.34 |
| Large  | **IntSeq** | **19.09** | **26.23** | **99.02** | 86.64 |
| Large  | Vanilla    | 2.59   | 3.80   | 92.05    | 100.0 |
| Large  | Ablation   | 11.75  | 21.79  | 98.94    | 86.99 |

Large スケールの IntSeqBERT は Top-1 精度 19.09% を達成し、Vanilla ベースライン（2.59%）の **7.4 倍**。

**Magnitude 別精度（Large IntSeqBERT）。** Solver 精度は Magnitude に強く依存する：

| バケット     | Top-1  | Top-10 |
|--------------|-------:|-------:|
| Small        | 68.34  | 88.50  |
| Medium       | 20.82  | 31.50  |
| Large        |  0.31  |  0.67  |
| Huge         |  0.09  |  0.18  |
| Astronomical |  0.00  |  0.00  |

大きな整数に対して Solver 精度は崩壊し、第 5.2 節の Magnitude MSE 結果と整合する。
一方、Small バケットでは Top-1 精度 68.34% を達成しており、低スケール領域では次項復元が実用的な精度に達している。

**モード別内訳（Large IntSeqBERT）。** Solver は探索幅 $\Delta n = n_{\max} - n_{\min}$ に応じて、dense（$\Delta n \le 10^6$）/ sieve（$10^6 < \Delta n \le 10^{14}$）/ crt（$\Delta n > 10^{14}$）の 3 モードを切り替える（第 3.9 節）。
加えて終了状態として zero / none をとる：

| モード | 使用率  | Top-1 | 説明                              |
|--------|--------:|------:|-----------------------------------|
| dense  | 24.0%   | 61.06 | 整数範囲の直接列挙                 |
| sieve  | 36.7%   |  5.36 | ふるい法による列挙                 |
| crt    | 23.2%   |  0.09 | 中国剰余定理（CRT）による再構成    |
| zero   |  2.7%   | 89.96 | 次項を 0 と予測                   |
| none   | 13.4%   |  0.00 | 有効な候補が得られなかったケース   |

中国剰余定理を用いて Modulo 予測から整数を再構成しようとする CRT モードは、現状ほぼゼロに近い精度しか達成していない。
この制限については第 6.4 節で議論する。

**Sieve モードの詳細**: Sieve モードでは、Modulo 予測確率の確信度が高い法を「アンカー」として貪欲に選定し、それらの最小公倍数が探索幅（候補数目標 100,000）を下回るまで追加する。各アンカーの確率上位余りをビームサーチで組み合わせて CRT による候補整数を生成し、スコアリング式で top_k 件に絞り込む。
