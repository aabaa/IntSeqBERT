# IntSeqBERT Paper Results (2026-03-02)

This directory contains the lightweight result artifacts used for the CICM 2026 paper tables
and figures.

- `summary.md`: human-readable experiment summary and tables.
- `manifest.json`: provenance metadata, split counts, and checksums.
- `checkpoint_manifest.json`: per-run checkpoint metadata, including copied log/config paths
  and SHA256 hashes for external model-weight artifacts.
- `checkpoints/`: lightweight checkpoint metadata copied from each run:
  `config.json`, `best_metrics.json`, `test_metrics.json`, `history.csv`, `train.log`, and
  `test_results.csv`.
- `cache/attention_summary.csv`: attention-analysis summary used for case-study figures.
- `cache/scatter_cache_{intseq,vanilla,ablation}.csv`: cached prediction-vs-truth points for
  Figure 5.

Large artifacts are intentionally not stored in Git:

- OEIS raw data and extracted features: regenerate with the README data-preparation commands.
- Model weight files (`*.pt`): publish separately via artifact storage such as GitHub
  Releases, Zenodo, or Hugging Face Hub. Their sizes and SHA256 hashes are recorded in
  `checkpoint_manifest.json`.
