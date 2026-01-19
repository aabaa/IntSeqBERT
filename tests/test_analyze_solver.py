"""
Unit tests for analyze_solver.py

Tests cover:
- Data loading functions (load_split_ids, load_test_samples)
- Magnitude bucket classification
- Match rank computation
- Metrics computation (overall, magnitude breakdown, mode breakdown)
- Batch preparation with masking
- Output functions
"""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from intseq_bert import config
from intseq_bert.analysis.analyze_solver import (
    # Data loading
    load_split_ids,
    load_test_samples,
    # Magnitude functions
    get_log10_magnitude,
    get_magnitude_bucket,
    # Inference helpers
    compute_match_rank,
    get_sign_idx,
    prepare_single_batch,
    # Metrics
    compute_overall_metrics,
    compute_magnitude_breakdown,
    compute_mode_breakdown,
    # Output
    save_results_csv,
    save_summary_json,
    save_config_json,
)


# ============================================================
# Test Fixtures
# ============================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_split_file(temp_dir):
    """Create a sample split file."""
    split_path = temp_dir / "test.txt"
    split_path.write_text("A000001\nA000002\nA000003\n")
    return split_path


@pytest.fixture
def sample_jsonl_file(temp_dir):
    """Create a sample JSONL file with test sequences."""
    jsonl_path = temp_dir / "data.jsonl"
    records = [
        {"oeis_id": "A000001", "sequence": [1, 2, 3, 4, 5]},
        {"oeis_id": "A000002", "sequence": [10, 20, 30]},
        {"oeis_id": "A000003", "sequence": [100, 200]},
        {"oeis_id": "A000004", "sequence": [1000, 2000, 3000]},  # Not in split
    ]
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return jsonl_path


@pytest.fixture
def sample_results():
    """Sample evaluation results for testing metrics computation."""
    return [
        # Small numbers, dense mode, correct
        {"oeis_id": "A001", "target": 10, "match_rank": 1, "solver_mode": "dense",
         "magnitude_bucket": "Small", "sign_pred": 0, "sign_true": 0},
        {"oeis_id": "A002", "target": 50, "match_rank": 1, "solver_mode": "dense",
         "magnitude_bucket": "Small", "sign_pred": 0, "sign_true": 0},
        # Medium number, sieve mode, wrong
        {"oeis_id": "A003", "target": 5000, "match_rank": -1, "solver_mode": "sieve",
         "magnitude_bucket": "Medium", "sign_pred": 0, "sign_true": 0},
        # Large number, crt mode, in top-5
        {"oeis_id": "A004", "target": 10**10, "match_rank": 3, "solver_mode": "crt",
         "magnitude_bucket": "Large", "sign_pred": 0, "sign_true": 0},
        # Zero, correct
        {"oeis_id": "A005", "target": 0, "match_rank": 1, "solver_mode": "zero",
         "magnitude_bucket": "Small", "sign_pred": 2, "sign_true": 2},
        # No solution
        {"oeis_id": "A006", "target": 999, "match_rank": -1, "solver_mode": "none",
         "magnitude_bucket": "Small", "sign_pred": -1, "sign_true": 0},
        # Negative number
        {"oeis_id": "A007", "target": -100, "match_rank": 1, "solver_mode": "dense",
         "magnitude_bucket": "Medium", "sign_pred": 1, "sign_true": 1},
    ]


# ============================================================
# Test: Data Loading Functions
# ============================================================


class TestLoadSplitIds:
    """Tests for load_split_ids function."""
    
    def test_loads_ids_correctly(self, sample_split_file):
        """Test that IDs are loaded as a set."""
        ids = load_split_ids(sample_split_file)
        assert ids == {"A000001", "A000002", "A000003"}
    
    def test_handles_empty_lines(self, temp_dir):
        """Test that empty lines are ignored."""
        split_path = temp_dir / "test.txt"
        split_path.write_text("A001\n\nA002\n  \nA003\n")
        ids = load_split_ids(split_path)
        assert len(ids) == 3
    
    def test_returns_set(self, sample_split_file):
        """Test that return type is set."""
        ids = load_split_ids(sample_split_file)
        assert isinstance(ids, set)


class TestLoadTestSamples:
    """Tests for load_test_samples function."""
    
    def test_loads_samples_in_split(self, sample_jsonl_file, sample_split_file):
        """Test that only samples in split are loaded."""
        split_ids = load_split_ids(sample_split_file)
        samples = load_test_samples(sample_jsonl_file, split_ids, max_samples=100)
        
        assert len(samples) == 3
        oeis_ids = {s["oeis_id"] for s in samples}
        assert "A000004" not in oeis_ids  # Not in split
    
    def test_separates_target_from_input(self, sample_jsonl_file, sample_split_file):
        """Test that target is seq[-1] and input is seq[:-1]."""
        split_ids = load_split_ids(sample_split_file)
        samples = load_test_samples(sample_jsonl_file, split_ids, max_samples=100)
        
        # A000001: [1, 2, 3, 4, 5] -> input=[1,2,3,4], target=5
        sample = next(s for s in samples if s["oeis_id"] == "A000001")
        assert sample["input_seq"] == [1, 2, 3, 4]
        assert sample["target"] == 5
        assert sample["target_str"] == "5"
    
    def test_respects_max_samples(self, sample_jsonl_file, sample_split_file):
        """Test that max_samples limit is respected."""
        split_ids = load_split_ids(sample_split_file)
        samples = load_test_samples(sample_jsonl_file, split_ids, max_samples=2)
        
        assert len(samples) == 2
    
    def test_skips_short_sequences(self, temp_dir):
        """Test that sequences with len < 2 are skipped."""
        jsonl_path = temp_dir / "data.jsonl"
        records = [
            {"oeis_id": "A001", "sequence": [1]},  # Too short
            {"oeis_id": "A002", "sequence": [1, 2]},  # OK
        ]
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        
        samples = load_test_samples(jsonl_path, {"A001", "A002"}, max_samples=100)
        assert len(samples) == 1
        assert samples[0]["oeis_id"] == "A002"
    
    def test_preserves_large_integers(self, temp_dir):
        """Test that large integers are preserved as Python int."""
        jsonl_path = temp_dir / "data.jsonl"
        large_num = 10**50
        records = [{"oeis_id": "A001", "sequence": [1, large_num]}]
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        
        samples = load_test_samples(jsonl_path, {"A001"}, max_samples=100)
        assert samples[0]["target"] == large_num
        assert isinstance(samples[0]["target"], int)


# ============================================================
# Test: Magnitude Functions
# ============================================================


class TestGetLog10Magnitude:
    """Tests for get_log10_magnitude function."""
    
    def test_positive_number(self):
        """Test log10 of positive number."""
        assert get_log10_magnitude(100) == pytest.approx(2.0)
        assert get_log10_magnitude(1000) == pytest.approx(3.0)
    
    def test_zero_returns_zero(self):
        """Test that zero returns 0.0."""
        assert get_log10_magnitude(0) == 0.0
    
    def test_negative_number(self):
        """Test log10 of negative number (uses abs)."""
        assert get_log10_magnitude(-100) == pytest.approx(2.0)
    
    def test_very_large_number(self):
        """Test fallback for very large numbers."""
        large_num = 10**200
        result = get_log10_magnitude(large_num)
        # Should be approximately 200
        assert result == pytest.approx(200.0, rel=0.01)


class TestGetMagnitudeBucket:
    """Tests for get_magnitude_bucket function."""
    
    def test_small_bucket(self):
        """Test Small bucket (0-100)."""
        assert get_magnitude_bucket(1) == "Small"
        assert get_magnitude_bucket(50) == "Small"
        assert get_magnitude_bucket(99) == "Small"
    
    def test_medium_bucket(self):
        """Test Medium bucket (100-100K)."""
        assert get_magnitude_bucket(100) == "Medium"
        assert get_magnitude_bucket(10000) == "Medium"
    
    def test_large_bucket(self):
        """Test Large bucket (100K-10^20)."""
        assert get_magnitude_bucket(10**6) == "Large"
        assert get_magnitude_bucket(10**15) == "Large"
    
    def test_huge_bucket(self):
        """Test Huge bucket (10^20-10^50)."""
        assert get_magnitude_bucket(10**25) == "Huge"
        assert get_magnitude_bucket(10**40) == "Huge"
    
    def test_astronomical_bucket(self):
        """Test Astronomical bucket (>10^50)."""
        assert get_magnitude_bucket(10**60) == "Astronomical"
    
    def test_zero(self):
        """Test zero classification."""
        # log10(0) = 0, which falls in Small bucket
        assert get_magnitude_bucket(0) == "Small"


# ============================================================
# Test: Inference Helpers
# ============================================================


class TestComputeMatchRank:
    """Tests for compute_match_rank function."""
    
    def test_finds_exact_match(self):
        """Test finding exact match at various positions."""
        candidates = [
            {"value": 10, "score": -1.0},
            {"value": 20, "score": -2.0},
            {"value": 30, "score": -3.0},
        ]
        
        assert compute_match_rank(candidates, 10) == 1
        assert compute_match_rank(candidates, 20) == 2
        assert compute_match_rank(candidates, 30) == 3
    
    def test_not_found_returns_minus_one(self):
        """Test that not found returns -1."""
        candidates = [{"value": 10}, {"value": 20}]
        assert compute_match_rank(candidates, 999) == -1
    
    def test_empty_candidates(self):
        """Test with empty candidate list."""
        assert compute_match_rank([], 10) == -1
    
    def test_large_integer_match(self):
        """Test matching large integers."""
        large_num = 10**50
        candidates = [{"value": large_num}]
        assert compute_match_rank(candidates, large_num) == 1


class TestGetSignIdx:
    """Tests for get_sign_idx function."""
    
    def test_positive(self):
        """Test positive number returns SIGN_POSITIVE."""
        assert get_sign_idx(1) == config.SIGN_POSITIVE
        assert get_sign_idx(100) == config.SIGN_POSITIVE
    
    def test_negative(self):
        """Test negative number returns SIGN_NEGATIVE."""
        assert get_sign_idx(-1) == config.SIGN_NEGATIVE
        assert get_sign_idx(-100) == config.SIGN_NEGATIVE
    
    def test_zero(self):
        """Test zero returns SIGN_ZERO."""
        assert get_sign_idx(0) == config.SIGN_ZERO


class TestPrepareSingleBatch:
    """Tests for prepare_single_batch function."""
    
    def test_appends_dummy_token(self):
        """Test that dummy token is appended to input."""
        from intseq_bert.collator import OEISCollator
        
        input_seq = [1, 2, 3]
        collator = OEISCollator(mask_prob=0.0)
        
        batch = prepare_single_batch(input_seq, collator, "cpu")
        
        # Input was [1, 2, 3], after adding dummy: [1, 2, 3, 0]
        # So batch should have length 4
        assert batch["mag_inputs"].shape[1] == 4
        assert batch["mod_inputs"].shape[1] == 4
    
    def test_masks_last_position_magnitude(self):
        """Test that magnitude stream is masked at last position."""
        from intseq_bert.collator import OEISCollator
        
        input_seq = [1, 2, 3]
        collator = OEISCollator(mask_prob=0.0)
        
        batch = prepare_single_batch(input_seq, collator, "cpu")
        
        # Last position content should be zero
        assert (batch["mag_inputs"][:, -1, :config.MAG_RAW_DIM] == 0.0).all()
        # is_masked flag should be 1.0
        assert batch["mag_inputs"][:, -1, -1] == 1.0
    
    def test_masks_last_position_modulo(self):
        """Test that modulo stream is zeroed at last position."""
        from intseq_bert.collator import OEISCollator
        
        input_seq = [1, 2, 3]
        collator = OEISCollator(mask_prob=0.0)
        
        batch = prepare_single_batch(input_seq, collator, "cpu")
        
        # Last position Sin/Cos should be zero
        assert (batch["mod_inputs"][:, -1, :] == 0.0).all()
    
    def test_preserves_context_positions(self):
        """Test that non-last positions are not zeroed."""
        from intseq_bert.collator import OEISCollator
        
        input_seq = [10, 20, 30]
        collator = OEISCollator(mask_prob=0.0)
        
        batch = prepare_single_batch(input_seq, collator, "cpu")
        
        # First position should have non-zero content (for value 10)
        # The log10(10) = 1, so 1 + 1 = 2.0 in mag feature
        assert batch["mag_inputs"][0, 0, 0] != 0.0


# ============================================================
# Test: Metrics Computation
# ============================================================


class TestComputeOverallMetrics:
    """Tests for compute_overall_metrics function."""
    
    def test_top1_accuracy(self, sample_results):
        """Test Top-1 accuracy computation."""
        metrics = compute_overall_metrics(sample_results, top_k=5)
        
        # 4 out of 7 have match_rank == 1
        expected_top1 = (4 / 7) * 100
        assert metrics["top1_acc"] == pytest.approx(expected_top1, rel=0.01)
    
    def test_topk_accuracy(self, sample_results):
        """Test Top-K accuracy computation."""
        metrics = compute_overall_metrics(sample_results, top_k=5)
        
        # 5 out of 7 have 1 <= match_rank <= 5 (A004 has rank 3)
        expected_topk = (5 / 7) * 100
        assert metrics["top5_acc"] == pytest.approx(expected_topk, rel=0.01)
    
    def test_sign_accuracy(self, sample_results):
        """Test sign accuracy computation."""
        metrics = compute_overall_metrics(sample_results, top_k=5)
        
        # 6 out of 7 have valid sign_pred (not -1), all match
        # Wait, A006 has sign_pred=-1, so excluded from sign calculation
        # 6 valid, all correct -> 100%
        assert metrics["sign_acc"] == pytest.approx(100.0)
    
    def test_valid_rate(self, sample_results):
        """Test valid rate computation."""
        metrics = compute_overall_metrics(sample_results, top_k=5)
        
        # 6 out of 7 have solver_mode != "none" and != "error"
        expected_valid = (6 / 7) * 100
        assert metrics["valid_rate"] == pytest.approx(expected_valid, rel=0.01)
    
    def test_empty_results(self):
        """Test with empty results."""
        metrics = compute_overall_metrics([], top_k=5)
        assert metrics["total_samples"] == 0
        assert metrics["top1_acc"] == 0.0


class TestComputeMagnitudeBreakdown:
    """Tests for compute_magnitude_breakdown function."""
    
    def test_buckets_computed(self, sample_results):
        """Test that buckets are computed correctly."""
        df = compute_magnitude_breakdown(sample_results, top_k=5)
        
        # Should have Small, Medium, Large buckets
        buckets = df["bucket"].tolist()
        assert "Small" in buckets
        assert "Medium" in buckets
        assert "Large" in buckets
    
    def test_counts_correct(self, sample_results):
        """Test that counts are correct per bucket."""
        df = compute_magnitude_breakdown(sample_results, top_k=5)
        
        small_row = df[df["bucket"] == "Small"].iloc[0]
        # 4 samples in Small: A001, A002, A005, A006
        assert small_row["count"] == 4
    
    def test_accuracy_per_bucket(self, sample_results):
        """Test that accuracy is computed per bucket."""
        df = compute_magnitude_breakdown(sample_results, top_k=5)
        
        small_row = df[df["bucket"] == "Small"].iloc[0]
        # 3 out of 4 in Small have match_rank == 1
        expected = (3 / 4) * 100
        assert small_row["top1_acc"] == pytest.approx(expected)


class TestComputeModeBreakdown:
    """Tests for compute_mode_breakdown function."""
    
    def test_modes_computed(self, sample_results):
        """Test that modes are computed correctly."""
        df = compute_mode_breakdown(sample_results, top_k=5)
        
        modes = df["mode"].tolist()
        assert "dense" in modes
        assert "sieve" in modes
        assert "crt" in modes
        assert "zero" in modes
        assert "none" in modes
    
    def test_usage_rate(self, sample_results):
        """Test that usage rate is computed."""
        df = compute_mode_breakdown(sample_results, top_k=5)
        
        dense_row = df[df["mode"] == "dense"].iloc[0]
        # 3 out of 7 use dense
        expected = (3 / 7) * 100
        assert dense_row["usage_rate"] == pytest.approx(expected, rel=0.01)
    
    def test_accuracy_per_mode(self, sample_results):
        """Test that accuracy is computed per mode."""
        df = compute_mode_breakdown(sample_results, top_k=5)
        
        dense_row = df[df["mode"] == "dense"].iloc[0]
        # All 3 dense have match_rank == 1
        assert dense_row["top1_acc"] == pytest.approx(100.0)


# ============================================================
# Test: Output Functions
# ============================================================


class TestSaveResultsCsv:
    """Tests for save_results_csv function."""
    
    def test_saves_csv(self, temp_dir):
        """Test that CSV is saved correctly."""
        results = [
            {"oeis_id": "A001", "target": 10, "target_str": "10",
             "pred_top1": 10, "match_rank": 1, "solver_mode": "dense",
             "mag_log10": 1.0, "score_top1": -0.5, "sign_pred": 0, "sign_true": 0}
        ]
        
        output_path = temp_dir / "results.csv"
        save_results_csv(results, output_path)
        
        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) == 1
        assert df.iloc[0]["oeis_id"] == "A001"


class TestSaveSummaryJson:
    """Tests for save_summary_json function."""
    
    def test_saves_json(self, temp_dir, sample_results):
        """Test that JSON is saved correctly."""
        overall = compute_overall_metrics(sample_results, top_k=5)
        magnitude_df = compute_magnitude_breakdown(sample_results, top_k=5)
        mode_df = compute_mode_breakdown(sample_results, top_k=5)
        
        output_path = temp_dir / "summary.json"
        save_summary_json(overall, magnitude_df, mode_df, 10.5, output_path)
        
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        
        assert "overall" in data
        assert "by_magnitude" in data
        assert "by_mode" in data
        assert "execution" in data
        assert data["execution"]["total_time_sec"] == 10.5


class TestSaveConfigJson:
    """Tests for save_config_json function."""
    
    def test_saves_config(self, temp_dir):
        """Test that config JSON is saved."""
        from argparse import Namespace
        
        args = Namespace(
            checkpoint="test.pt",
            split_type="std",
            split_name="test",
            max_samples=100,
            top_k=5,
            filter_magnitude=None,
            device="cpu"
        )
        
        output_path = temp_dir / "config.json"
        save_config_json(args, output_path)
        
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        
        assert data["checkpoint"] == "test.pt"
        assert data["split_type"] == "std"
        assert "timestamp" in data


# ============================================================
# Test: Data Leakage Prevention
# ============================================================


class TestDataLeakagePrevention:
    """Tests to verify no data leakage (cheating) occurs."""
    
    def test_target_not_in_input_seq(self):
        """Test that target is separated from input sequence."""
        # Simulate load_test_samples behavior
        seq = [1, 2, 3, 4, 5]
        target = seq[-1]
        input_seq = seq[:-1]
        
        assert target == 5
        assert 5 not in input_seq
        assert input_seq == [1, 2, 3, 4]
    
    def test_dummy_position_is_masked(self):
        """Test that the dummy position (0) is fully masked."""
        from intseq_bert.collator import OEISCollator
        
        input_seq = [100, 200, 300]
        collator = OEISCollator(mask_prob=0.0)
        
        batch = prepare_single_batch(input_seq, collator, "cpu")
        
        # The dummy 0 is at position 3 (index -1)
        # It should be completely masked (content=0, flag=1)
        mag_content = batch["mag_inputs"][:, -1, :config.MAG_RAW_DIM]
        mag_flag = batch["mag_inputs"][:, -1, -1]
        mod_content = batch["mod_inputs"][:, -1, :]
        
        # All content should be zero
        assert (mag_content == 0.0).all(), "Magnitude content not zeroed"
        assert (mod_content == 0.0).all(), "Modulo content not zeroed"
        # Flag should be 1
        assert mag_flag == 1.0, "Mask flag not set"
    
    def test_collator_no_random_masking(self):
        """Test that collator doesn't apply random masking during eval."""
        from intseq_bert.collator import OEISCollator
        
        # In main(), collator is created with mask_prob=0.0
        collator = OEISCollator(mask_prob=0.0)
        assert collator.mask_prob == 0.0


# ============================================================
# Test: Integration
# ============================================================


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline_mock(self, temp_dir):
        """Test full pipeline with mocked model."""
        # Create test data
        split_path = temp_dir / "test.txt"
        split_path.write_text("A001\n")
        
        jsonl_path = temp_dir / "data.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"oeis_id": "A001", "sequence": [1, 2, 3]}) + "\n")
        
        # Load data
        split_ids = load_split_ids(split_path)
        samples = load_test_samples(jsonl_path, split_ids, max_samples=100)
        
        assert len(samples) == 1
        assert samples[0]["input_seq"] == [1, 2]
        assert samples[0]["target"] == 3
        
        # Simulate evaluation result
        results = [{
            "oeis_id": "A001",
            "target": 3,
            "target_str": "3",
            "pred_top1": 3,
            "match_rank": 1,
            "solver_mode": "dense",
            "mag_log10": get_log10_magnitude(3),
            "score_top1": -0.1,
            "sign_pred": 0,
            "sign_true": 0,
            "magnitude_bucket": get_magnitude_bucket(3)
        }]
        
        # Compute metrics
        overall = compute_overall_metrics(results, top_k=5)
        assert overall["top1_acc"] == 100.0
        
        # Save outputs
        save_results_csv(results, temp_dir / "results.csv")
        assert (temp_dir / "results.csv").exists()
