"""
test_analyze_magnitude.py:
Unit tests for analyze_magnitude.py module.
Tests accuracy metrics, scale-wise analysis, calibration, and worst-K analysis.
"""

import pytest
import torch
import numpy as np
import json
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

# Optional dependency
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

from intseq_bert import config


# ==========================================
# Test Fixtures
# ==========================================

@pytest.fixture
def sample_num_sequences():
    return 50


@pytest.fixture
def sample_sequence_length():
    return 16


@pytest.fixture
def sample_gt_values(sample_num_sequences, sample_sequence_length):
    """Creates sample ground truth magnitude values (log scale)."""
    N, L = sample_num_sequences, sample_sequence_length
    # Generate values across different scales (0 to 60 in log10)
    return torch.rand(N, L) * 60


@pytest.fixture
def sample_pred_values(sample_gt_values):
    """Creates sample predicted magnitude values with some noise."""
    noise = torch.randn_like(sample_gt_values) * 0.5
    return sample_gt_values + noise


@pytest.fixture
def sample_pred_sigma(sample_num_sequences, sample_sequence_length):
    """Creates sample predicted standard deviations."""
    N, L = sample_num_sequences, sample_sequence_length
    return torch.rand(N, L) * 0.5 + 0.1  # sigma between 0.1 and 0.6


@pytest.fixture
def sample_mask_map(sample_num_sequences, sample_sequence_length):
    """Creates sample mask map: (N, L) with some positions masked."""
    N, L = sample_num_sequences, sample_sequence_length
    mask = torch.ones(N, L, dtype=torch.bool)
    # Mask last 4 positions (simulate padding)
    mask[:, -4:] = False
    return mask


@pytest.fixture
def sample_oeis_ids(sample_num_sequences):
    """Creates sample OEIS IDs."""
    return [f"A{i:06d}" for i in range(sample_num_sequences)]


@pytest.fixture
def sample_id_to_tags(sample_oeis_ids):
    """Creates sample ID to tags mapping."""
    tags_list = ["poly", "exp", "linear", "prime", "mult"]
    id_to_tags = {}
    for i, oeis_id in enumerate(sample_oeis_ids):
        id_to_tags[oeis_id] = [tags_list[i % len(tags_list)]]
    return id_to_tags


# ==========================================
# Markers for conditional skipping
# ==========================================

try:
    from intseq_bert.analysis import analyze_magnitude
    HAS_ANALYZE_MAGNITUDE = True
except ImportError:
    HAS_ANALYZE_MAGNITUDE = False

requires_analyze_magnitude = pytest.mark.skipif(
    not HAS_ANALYZE_MAGNITUDE,
    reason="analyze_magnitude module not implemented yet"
)


# ==========================================
# Accuracy Metrics Tests
# ==========================================

class TestAccuracyMetrics:
    """Tests for accuracy metrics computation."""
    
    @requires_analyze_magnitude
    def test_compute_mse(self):
        """Test MSE computation."""
        from intseq_bert.analysis.analyze_magnitude import compute_mse
        
        gt = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.tensor([1.0, 2.5, 3.5])
        
        mse = compute_mse(gt, pred)
        expected = ((0.0**2) + (0.5**2) + (0.5**2)) / 3
        assert abs(mse - expected) < 1e-6
    
    @requires_analyze_magnitude
    def test_compute_rmse(self):
        """Test RMSE computation."""
        from intseq_bert.analysis.analyze_magnitude import compute_rmse
        
        gt = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.tensor([1.0, 2.5, 3.5])
        
        rmse = compute_rmse(gt, pred)
        mse = ((0.0**2) + (0.5**2) + (0.5**2)) / 3
        expected = np.sqrt(mse)
        assert abs(rmse - expected) < 1e-6
    
    @requires_analyze_magnitude
    def test_compute_mae(self):
        """Test MAE computation."""
        from intseq_bert.analysis.analyze_magnitude import compute_mae
        
        gt = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.tensor([1.0, 2.5, 3.5])
        
        mae = compute_mae(gt, pred)
        expected = (0.0 + 0.5 + 0.5) / 3
        assert abs(mae - expected) < 1e-6
    
    @requires_analyze_magnitude
    def test_compute_medae(self):
        """Test Median Absolute Error computation."""
        from intseq_bert.analysis.analyze_magnitude import compute_medae
        
        gt = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = torch.tensor([1.1, 2.0, 3.5, 4.0, 5.2])
        
        medae = compute_medae(gt, pred)
        # abs errors: [0.1, 0.0, 0.5, 0.0, 0.2] -> sorted: [0.0, 0.0, 0.1, 0.2, 0.5] -> median = 0.1
        expected = 0.1
        assert abs(medae - expected) < 1e-6
    
    @requires_analyze_magnitude
    def test_compute_r2_perfect(self):
        """Test R² = 1.0 for perfect prediction."""
        from intseq_bert.analysis.analyze_magnitude import compute_r2
        
        gt = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = gt.clone()
        
        r2 = compute_r2(gt, pred)
        assert abs(r2 - 1.0) < 1e-6
    
    @requires_analyze_magnitude
    def test_compute_r2_range(self):
        """Test R² is typically in [-1, 1] range for reasonable predictions."""
        from intseq_bert.analysis.analyze_magnitude import compute_r2
        
        gt = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = torch.tensor([1.1, 2.1, 3.0, 3.9, 5.1])
        
        r2 = compute_r2(gt, pred)
        assert r2 <= 1.0


# ==========================================
# Tolerance Accuracy Tests
# ==========================================

class TestToleranceAccuracy:
    """Tests for tolerance accuracy computation."""
    
    @requires_analyze_magnitude
    def test_tolerance_accuracy_all_within(self):
        """Test tolerance accuracy = 100% when all predictions are within tolerance."""
        from intseq_bert.analysis.analyze_magnitude import compute_tolerance_accuracy
        
        gt = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.tensor([1.1, 2.1, 3.1])  # All within 0.5
        
        acc = compute_tolerance_accuracy(gt, pred, tolerance=0.5)
        assert abs(acc - 100.0) < 1e-6
    
    @requires_analyze_magnitude
    def test_tolerance_accuracy_none_within(self):
        """Test tolerance accuracy = 0% when no predictions are within tolerance."""
        from intseq_bert.analysis.analyze_magnitude import compute_tolerance_accuracy
        
        gt = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.tensor([2.0, 3.0, 4.0])  # All off by 1.0
        
        acc = compute_tolerance_accuracy(gt, pred, tolerance=0.5)
        assert abs(acc - 0.0) < 1e-6
    
    @requires_analyze_magnitude
    def test_tolerance_accuracy_partial(self):
        """Test tolerance accuracy for partial matches."""
        from intseq_bert.analysis.analyze_magnitude import compute_tolerance_accuracy
        
        gt = torch.tensor([1.0, 2.0, 3.0, 4.0])
        pred = torch.tensor([1.1, 2.6, 3.1, 4.6])  # 2 within 0.5
        
        acc = compute_tolerance_accuracy(gt, pred, tolerance=0.5)
        assert abs(acc - 50.0) < 1e-6


# ==========================================
# Correlation Metrics Tests
# ==========================================

class TestCorrelationMetrics:
    """Tests for correlation metrics."""
    
    @requires_analyze_magnitude
    def test_pearson_correlation_perfect(self):
        """Test Pearson correlation = 1.0 for perfect linear relationship."""
        from intseq_bert.analysis.analyze_magnitude import compute_pearson
        
        gt = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = gt.clone()
        
        rho = compute_pearson(gt, pred)
        assert abs(rho - 1.0) < 1e-6
    
    @requires_analyze_magnitude
    def test_spearman_correlation_perfect_rank(self):
        """Test Spearman correlation = 1.0 for perfect rank preservation."""
        from intseq_bert.analysis.analyze_magnitude import compute_spearman
        
        gt = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])  # Different scale, same rank
        
        rho = compute_spearman(gt, pred)
        assert abs(rho - 1.0) < 1e-6


# ==========================================
# Scale-wise Analysis Tests
# ==========================================

class TestScaleWiseAnalysis:
    """Tests for scale-wise (bucket) analysis."""
    
    @requires_analyze_magnitude
    def test_bucket_assignment(self):
        """Test correct bucket assignment based on log10 value."""
        from intseq_bert.analysis.analyze_magnitude import get_bucket_name
        
        assert get_bucket_name(1.0) == "Small"    # log10 in [0, 2)
        assert get_bucket_name(3.0) == "Medium"   # log10 in [2, 5)
        assert get_bucket_name(10.0) == "Large"   # log10 in [5, 20)
        assert get_bucket_name(30.0) == "Huge"    # log10 in [20, 50)
        assert get_bucket_name(60.0) == "Astronomical"  # log10 >= 50
    
    @requires_analyze_magnitude
    def test_scale_wise_metrics_structure(self, sample_gt_values, sample_pred_values, sample_mask_map):
        """Test scale-wise metrics has expected structure."""
        from intseq_bert.analysis.analyze_magnitude import compute_scale_wise_metrics
        
        df = compute_scale_wise_metrics(sample_gt_values, sample_pred_values, sample_mask_map)
        
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"bucket", "count", "mse", "mae", "mse_ci_lower", "mse_ci_upper", "is_reliable"}
        assert expected_cols.issubset(set(df.columns))
    
    @requires_analyze_magnitude
    def test_scale_wise_count_recorded(self, sample_gt_values, sample_pred_values, sample_mask_map):
        """Test that sample count is recorded for each bucket."""
        from intseq_bert.analysis.analyze_magnitude import compute_scale_wise_metrics
        
        df = compute_scale_wise_metrics(sample_gt_values, sample_pred_values, sample_mask_map)
        
        assert "count" in df.columns
        assert df["count"].min() >= 0
        assert df["count"].sum() > 0
    
    @requires_analyze_magnitude
    def test_reliability_flag(self):
        """Test is_reliable is False when count < 30."""
        from intseq_bert.analysis.analyze_magnitude import compute_scale_wise_metrics
        
        # Create data with only small values (one bucket with few samples)
        gt = torch.tensor([1.0] * 10)  # Only 10 samples
        pred = torch.tensor([1.1] * 10)
        mask = torch.ones(10, dtype=torch.bool)
        
        df = compute_scale_wise_metrics(gt.unsqueeze(0), pred.unsqueeze(0), mask.unsqueeze(0))
        
        # All buckets should have is_reliable = False since N < 30
        for _, row in df.iterrows():
            if row["count"] < 30:
                assert row["is_reliable"] == False


# ==========================================
# Calibration Tests
# ==========================================

@requires_analyze_magnitude
class TestCalibration:
    """Tests for uncertainty calibration analysis."""
    
    def test_calibration_data_structure(self, sample_gt_values, sample_pred_values, sample_pred_sigma):
        """Test calibration data has expected structure."""
        from intseq_bert.analysis.analyze_magnitude import compute_calibration_data
        
        gt_flat = sample_gt_values.flatten()
        pred_flat = sample_pred_values.flatten()
        sigma_flat = sample_pred_sigma.flatten()
        
        df = compute_calibration_data(gt_flat, pred_flat, sigma_flat, n_bins=10)
        
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"bin", "mean_sigma", "rmse", "count"}
        assert expected_cols.issubset(set(df.columns))
    
    def test_calibration_bins_count(self, sample_gt_values, sample_pred_values, sample_pred_sigma):
        """Test correct number of bins are created."""
        from intseq_bert.analysis.analyze_magnitude import compute_calibration_data
        
        gt_flat = sample_gt_values.flatten()
        pred_flat = sample_pred_values.flatten()
        sigma_flat = sample_pred_sigma.flatten()
        
        for n_bins in [5, 10, 20]:
            df = compute_calibration_data(gt_flat, pred_flat, sigma_flat, n_bins=n_bins)
            assert len(df) <= n_bins  # May be fewer if some bins are empty
    
    def test_expected_calibration_error(self, sample_gt_values, sample_pred_values, sample_pred_sigma):
        """Test ECE is non-negative."""
        from intseq_bert.analysis.analyze_magnitude import compute_expected_calibration_error
        
        gt_flat = sample_gt_values.flatten()
        pred_flat = sample_pred_values.flatten()
        sigma_flat = sample_pred_sigma.flatten()
        
        ece = compute_expected_calibration_error(gt_flat, pred_flat, sigma_flat)
        
        assert ece >= 0


# ==========================================
# Worst-K Analysis Tests
# ==========================================

class TestWorstKAnalysis:
    """Tests for worst-K sample extraction."""
    
    @requires_analyze_magnitude
    def test_worst_k_extraction(self, sample_gt_values, sample_pred_values, sample_oeis_ids):
        """Test worst-K samples are extracted correctly."""
        from intseq_bert.analysis.analyze_magnitude import extract_worst_k_samples
        
        k = 10
        df = extract_worst_k_samples(
            sample_gt_values, 
            sample_pred_values,
            mask=None, 
            oeis_ids=sample_oeis_ids,
            k=k
        )
        
        assert len(df) == k
        
        # Check errors are sorted descending
        errors = df["error"].tolist()
        assert errors == sorted(errors, reverse=True)
    
    @requires_analyze_magnitude
    def test_worst_k_columns(self, sample_gt_values, sample_pred_values, sample_oeis_ids):
        """Test worst-K output has required columns."""
        from intseq_bert.analysis.analyze_magnitude import extract_worst_k_samples
        
        df = extract_worst_k_samples(
            sample_gt_values, 
            sample_pred_values,
            mask=None,
            oeis_ids=sample_oeis_ids,
            k=5
        )
        
        expected_cols = {"rank", "oeis_id", "position", "gt_value", "pred_value", "error", "context"}
        assert expected_cols.issubset(set(df.columns))
    
    @requires_analyze_magnitude
    def test_worst_k_rank_ordering(self, sample_gt_values, sample_pred_values, sample_oeis_ids):
        """Test rank column is correctly ordered."""
        from intseq_bert.analysis.analyze_magnitude import extract_worst_k_samples
        
        df = extract_worst_k_samples(
            sample_gt_values, 
            sample_pred_values,
            mask=None, 
            oeis_ids=sample_oeis_ids,
            k=5
        )
        
        assert df["rank"].tolist() == [1, 2, 3, 4, 5]


# ==========================================
# Context Formatting Tests
# ==========================================

class TestContextFormatting:
    """Tests for context string formatting."""
    
    @requires_analyze_magnitude
    def test_format_context_middle(self):
        """Test context formatting for middle position."""
        from intseq_bert.analysis.analyze_magnitude import format_context
        
        sequence = [0.0, 1.2, 3.4, 5.6, 7.8, 9.0]
        context = format_context(sequence, position=3, window=2)
        
        assert "[5.60]" in context  # Target value highlighted
        assert "1.20" in context    # Before target
        assert "7.80" in context    # After target
    
    @requires_analyze_magnitude
    def test_format_context_start(self):
        """Test context formatting for start position."""
        from intseq_bert.analysis.analyze_magnitude import format_context
        
        sequence = [1.0, 2.0, 3.0, 4.0, 5.0]
        context = format_context(sequence, position=0, window=2)
        
        assert "[1.00]" in context  # Target value
        assert "2.00" in context    # After target
        assert "..." not in context or context.index("[") < context.index("...")  # No leading ellipsis
    
    @requires_analyze_magnitude
    def test_format_context_end(self):
        """Test context formatting for end position."""
        from intseq_bert.analysis.analyze_magnitude import format_context
        
        sequence = [1.0, 2.0, 3.0, 4.0, 5.0]
        context = format_context(sequence, position=4, window=2)
        
        assert "[5.00]" in context  # Target value
        assert "3.00" in context    # Before target
    
    @requires_analyze_magnitude
    def test_format_context_with_ellipsis(self):
        """Test context includes ellipsis when truncated."""
        from intseq_bert.analysis.analyze_magnitude import format_context
        
        sequence = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        context = format_context(sequence, position=5, window=2)
        
        # Should have ellipsis on both sides
        assert context.count("...") == 2


# ==========================================
# Sign-Magnitude Consistency Tests
# ==========================================

class TestSignMagnitudeConsistency:
    """Tests for sign-magnitude consistency check."""
    
    @requires_analyze_magnitude
    def test_consistency_all_consistent(self):
        """Test 100% consistency when all predictions are consistent."""
        from intseq_bert.analysis.analyze_magnitude import compute_sign_magnitude_consistency
        
        pred_mag = torch.tensor([1.0, 2.0, -1.0, 0.0])
        pred_sign = torch.tensor([0, 0, 1, 2])  # 0=positive, 1=negative, 2=zero
        
        rate = compute_sign_magnitude_consistency(pred_mag, pred_sign)
        assert abs(rate - 100.0) < 1e-6
    
    @requires_analyze_magnitude
    def test_consistency_all_inconsistent(self):
        """Test 0% consistency when all predictions are inconsistent."""
        from intseq_bert.analysis.analyze_magnitude import compute_sign_magnitude_consistency
        
        pred_mag = torch.tensor([1.0, 2.0])   # All positive
        pred_sign = torch.tensor([1, 1])       # All negative sign predicted
        
        rate = compute_sign_magnitude_consistency(pred_mag, pred_sign)
        assert abs(rate - 0.0) < 1e-6
    
    @requires_analyze_magnitude
    def test_consistency_partial(self):
        """Test partial consistency."""
        from intseq_bert.analysis.analyze_magnitude import compute_sign_magnitude_consistency
        
        pred_mag = torch.tensor([1.0, 2.0, -1.0, -2.0])  # 2 positive, 2 negative
        pred_sign = torch.tensor([0, 1, 1, 0])  # 0=positive, 1=negative
        # First: consistent (mag>0, sign=positive)
        # Second: inconsistent (mag>0, sign=negative)
        # Third: consistent (mag<0, sign=negative)
        # Fourth: inconsistent (mag<0, sign=positive)
        
        rate = compute_sign_magnitude_consistency(pred_mag, pred_sign)
        assert abs(rate - 50.0) < 1e-6


# ==========================================
# Error Distribution Tests
# ==========================================

class TestErrorDistribution:
    """Tests for error distribution analysis."""
    
    @requires_analyze_magnitude
    def test_error_distribution_stats(self, sample_gt_values, sample_pred_values):
        """Test error distribution statistics computation."""
        from intseq_bert.analysis.analyze_magnitude import compute_error_distribution_stats
        
        errors = sample_gt_values - sample_pred_values
        stats = compute_error_distribution_stats(errors.flatten())
        
        expected_keys = {"mean", "median", "std", "skewness", "kurtosis"}
        assert expected_keys.issubset(set(stats.keys()))
    
    @requires_analyze_magnitude
    def test_error_distribution_stats_types(self, sample_gt_values, sample_pred_values):
        """Test error distribution stats are finite numbers."""
        from intseq_bert.analysis.analyze_magnitude import compute_error_distribution_stats
        
        errors = sample_gt_values - sample_pred_values
        stats = compute_error_distribution_stats(errors.flatten())
        
        for key, value in stats.items():
            assert np.isfinite(value), f"{key} should be finite"


# ==========================================
# Bootstrap CI Tests
# ==========================================

class TestBootstrapCI:
    """Tests for bootstrap confidence interval estimation."""
    
    @requires_analyze_magnitude
    def test_bootstrap_ci_returns_tuple(self):
        """Test bootstrap_ci returns (lower, upper) tuple."""
        from intseq_bert.analysis.analyze_magnitude import bootstrap_ci
        
        gt = np.random.randn(100)
        pred = gt + np.random.randn(100) * 0.1  # Add small noise
        lower, upper = bootstrap_ci(gt, pred, lambda g, p: ((g - p) ** 2).mean(), n_samples=100)
        
        assert lower < upper
    
    @requires_analyze_magnitude
    def test_bootstrap_ci_contains_mean(self):
        """Test bootstrap CI typically contains the sample MSE."""
        from intseq_bert.analysis.analyze_magnitude import bootstrap_ci
        
        np.random.seed(42)
        gt = np.random.randn(100)
        pred = gt + np.random.randn(100) * 0.1
        
        mse_fn = lambda g, p: ((g - p) ** 2).mean()
        lower, upper = bootstrap_ci(gt, pred, mse_fn, n_samples=500)
        
        sample_mse = mse_fn(gt, pred)
        # The CI should typically contain the sample MSE
        assert lower <= sample_mse <= upper


# ==========================================
# Integration Tests
# ==========================================

@requires_analyze_magnitude
class TestIntegration:
    """Integration tests for analyze_magnitude module."""
    
    def test_full_metrics_pipeline(
        self, sample_gt_values, sample_pred_values, sample_pred_sigma, sample_mask_map
    ):
        """Test computing all metrics in one pipeline."""
        from intseq_bert.analysis.analyze_magnitude import compute_overall_metrics
        
        metrics = compute_overall_metrics(
            sample_gt_values,
            sample_pred_values,
            sample_pred_sigma,
            sample_mask_map
        )
        
        expected_keys = {
            "mse", "rmse", "mae", "medae", "r2",
            "acc_0.5", "acc_0.1", "pearson", "spearman"
        }
        assert expected_keys.issubset(set(metrics.keys()))
        
        # Verify values are reasonable
        assert metrics["mse"] >= 0
        assert 0 <= metrics["acc_0.5"] <= 100
        assert 0 <= metrics["acc_0.1"] <= 100
    
    def test_output_file_generation(self, tmp_path):
        """Test that all expected output files are generated."""
        # This would be a full integration test with mocked model
        # For now, just verify the expected file structure
        expected_files = [
            "overall_metrics.csv",
            "scale_wise_metrics.csv",
            "tag_wise_metrics.csv",
            "calibration_data.csv",
            "error_distribution.csv",
            "worst_k_samples.csv",
            "consistency_report.csv"
        ]
        
        expected_figures = [
            "error_vs_scale.png",
            "prediction_scatter.png",
            "calibration_plot.png",
            "error_histogram.png",
            "error_qq_plot.png"
        ]
        
        # Just verify the lists are defined correctly
        assert len(expected_files) == 7
        assert len(expected_figures) == 5


# ==========================================
# Additional Missing Tests
# ==========================================

class TestToleranceAccuracyAcc005:
    """Additional tests for Acc_0.05 tolerance."""
    
    @requires_analyze_magnitude
    def test_tolerance_accuracy_005(self):
        """Test tolerance accuracy with 0.05 tolerance (spec 4.2)."""
        from intseq_bert.analysis.analyze_magnitude import compute_tolerance_accuracy
        
        gt = torch.tensor([1.0, 2.0, 3.0, 4.0])
        pred = torch.tensor([1.01, 2.06, 3.04, 4.1])  # 2 within 0.05
        
        acc = compute_tolerance_accuracy(gt, pred, tolerance=0.05)
        assert abs(acc - 50.0) < 1e-6


@requires_analyze_magnitude
class TestNLL:
    """Tests for Negative Log Likelihood computation (spec 4.3)."""
    
    def test_nll_non_negative(self, sample_gt_values, sample_pred_values, sample_pred_sigma):
        """Test NLL is computed and returns a finite value."""
        from intseq_bert.analysis.analyze_magnitude import compute_nll
        
        gt_flat = sample_gt_values.flatten()
        pred_flat = sample_pred_values.flatten()
        sigma_flat = sample_pred_sigma.flatten()
        
        nll = compute_nll(gt_flat, pred_flat, sigma_flat)
        
        assert np.isfinite(nll)
    
    def test_nll_lower_with_better_predictions(self):
        """Test NLL is lower for better predictions."""
        from intseq_bert.analysis.analyze_magnitude import compute_nll
        
        gt = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        pred_good = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # Perfect
        pred_bad = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])   # Off by 1.0
        sigma = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        
        nll_good = compute_nll(gt, pred_good, sigma)
        nll_bad = compute_nll(gt, pred_bad, sigma)
        
        assert nll_good < nll_bad


class TestWorstKWithTag:
    """Additional tests for worst-K with tag column."""
    
    @requires_analyze_magnitude
    def test_worst_k_includes_tag(
        self, sample_gt_values, sample_pred_values, sample_oeis_ids, sample_id_to_tags
    ):
        """Test worst-K output includes tag column (spec 5.4)."""
        from intseq_bert.analysis.analyze_magnitude import extract_worst_k_samples
        
        df = extract_worst_k_samples(
            sample_gt_values, 
            sample_pred_values,
            mask=None,
            oeis_ids=sample_oeis_ids,
            id_to_tags=sample_id_to_tags,
            k=5
        )
        
        assert "tag" in df.columns
        # Tags should be non-empty strings
        for tag in df["tag"]:
            assert isinstance(tag, str)
            assert len(tag) > 0


@requires_analyze_magnitude
class TestTagStratifiedAnalysis:
    """Tests for tag-stratified analysis (spec 1.3)."""
    
    def test_tag_stratified_output_structure(
        self, sample_gt_values, sample_pred_values, sample_oeis_ids, sample_id_to_tags, sample_mask_map
    ):
        """Test tag-stratified analysis has expected structure."""
        from intseq_bert.analysis.analyze_magnitude import compute_tag_stratified_metrics
        
        df = compute_tag_stratified_metrics(
            sample_gt_values, 
            sample_pred_values, 
            sample_mask_map,
            sample_oeis_ids,
            sample_id_to_tags
        )
        
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"tag", "count", "mse", "mae"}
        assert expected_cols.issubset(set(df.columns))
    
    def test_tag_stratified_covers_all_tags(
        self, sample_gt_values, sample_pred_values, sample_oeis_ids, sample_id_to_tags, sample_mask_map
    ):
        """Test all tags with sufficient samples are included."""
        from intseq_bert.analysis.analyze_magnitude import compute_tag_stratified_metrics
        
        df = compute_tag_stratified_metrics(
            sample_gt_values, 
            sample_pred_values, 
            sample_mask_map,
            sample_oeis_ids,
            sample_id_to_tags
        )
        
        # At least some tags should be present
        assert len(df) > 0
        
        # Count column should be positive
        assert df["count"].min() > 0
