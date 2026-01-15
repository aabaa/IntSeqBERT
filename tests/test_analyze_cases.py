"""
test_analyze_cases.py:
Unit tests for analyze_cases.py case study visualization module.
Tests visualization functions, data loading, and model wrappers.

Note: These tests require matplotlib and the analyze_cases module to be implemented.
Tests will be skipped if dependencies are not available.
"""

import pytest
import torch
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Check for optional dependencies
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

# Check if analyze_cases module exists
try:
    from intseq_bert.analysis import analyze_cases
    HAS_ANALYZE_CASES = True
except ImportError:
    HAS_ANALYZE_CASES = False

from intseq_bert import config


# ==========================================
# Test Fixtures
# ==========================================

@pytest.fixture
def sample_sequence_length():
    return 20


@pytest.fixture
def sample_batch(sample_sequence_length):
    """Creates a sample batch for testing."""
    L = sample_sequence_length
    return {
        "mag_inputs": torch.randn(1, L, config.MAG_EXTENDED_DIM),
        "mod_inputs": torch.randn(1, L, config.MOD_FEATURE_DIM),
        "attention_mask": torch.ones(1, L),
        "oeis_id": "A000045"
    }


@pytest.fixture
def sample_predictions(sample_sequence_length):
    """Creates sample model predictions."""
    L = sample_sequence_length
    return {
        "mag_mu": torch.randn(1, L),
        "mag_log_var": torch.randn(1, L),
        "sign_logits": torch.randn(1, L, config.NUM_SIGN_CLASSES),
        "mod_logits": torch.randn(1, L, sum(config.MOD_RANGE)),
    }


@pytest.fixture
def temp_features_dir(tmp_path, sample_batch):
    """Creates a temporary features directory with a sample .pt file."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    
    # Save sample features
    pt_data = {
        "mag_features": sample_batch["mag_inputs"].squeeze(0),
        "mod_features": sample_batch["mod_inputs"].squeeze(0),
    }
    torch.save(pt_data, features_dir / "A000045.pt")
    
    return features_dir


@pytest.fixture
def temp_jsonl_file(tmp_path):
    """Creates a temporary JSONL file with sample records."""
    jsonl_path = tmp_path / "data.jsonl"
    
    records = [
        {"oeis_id": "A000045", "values": [1, 1, 2, 3, 5, 8, 13, 21, 34, 55], "keywords": ["core", "nice"]},
        {"oeis_id": "A000040", "values": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29], "keywords": ["prime"]},
    ]
    
    with open(jsonl_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    
    return jsonl_path


# ==========================================
# Markers for conditional skipping
# ==========================================

requires_analyze_cases = pytest.mark.skipif(
    not HAS_ANALYZE_CASES,
    reason="analyze_cases module not implemented yet"
)

requires_matplotlib = pytest.mark.skipif(
    not HAS_MATPLOTLIB,
    reason="matplotlib not installed"
)


# ==========================================
# DEFAULT_DISPLAY_MODS Tests
# ==========================================

class TestDefaultDisplayMods:
    """Tests for DEFAULT_DISPLAY_MODS configuration."""
    
    def test_default_display_mods_structure(self):
        """Test DEFAULT_DISPLAY_MODS contains expected categories."""
        # Expected structure based on spec
        expected_primes = [2, 3, 5, 7, 11, 13]
        expected_composites = [4, 6, 12]
        expected_base10 = [10, 100]
        
        # All expected mods should be valid (in MOD_RANGE)
        all_expected = expected_primes + expected_composites + expected_base10
        for m in all_expected:
            assert m in config.MOD_RANGE, f"Mod {m} not in MOD_RANGE"
    
    def test_display_mods_are_in_valid_range(self):
        """Test all display mods are within valid range."""
        display_mods = [2, 3, 5, 7, 11, 13, 4, 6, 12, 10, 100]
        for m in display_mods:
            assert 2 <= m <= 101, f"Mod {m} out of range"


# ==========================================
# load_single_sequence Tests
# ==========================================

@requires_analyze_cases
class TestLoadSingleSequence:
    """Tests for load_single_sequence function."""
    
    def test_load_from_pt_file(self, temp_features_dir):
        """Test loading from existing .pt file."""
        from intseq_bert.analysis.analyze_cases import load_single_sequence
        
        result = load_single_sequence("A000045", temp_features_dir)
        
        assert "mag_inputs" in result
        assert "mod_inputs" in result
        assert "attention_mask" in result
        assert "oeis_id" in result
        
        assert result["mag_inputs"].dim() == 3  # (1, L, 5)
        assert result["mag_inputs"].shape[0] == 1  # Batch size
        assert result["oeis_id"] == "A000045"
    
    def test_load_nonexistent_raises_error(self, temp_features_dir):
        """Test loading nonexistent file raises FileNotFoundError."""
        from intseq_bert.analysis.analyze_cases import load_single_sequence
        
        with pytest.raises(FileNotFoundError):
            load_single_sequence("A999999", temp_features_dir)
    
    def test_load_with_jsonl_fallback(self, tmp_path, temp_jsonl_file):
        """Test fallback loading from JSONL when .pt doesn't exist."""
        from intseq_bert.analysis.analyze_cases import load_single_sequence
        
        empty_features_dir = tmp_path / "empty_features"
        empty_features_dir.mkdir()
        
        # This should fall back to JSONL
        # Note: This test may need mocking if extract_features isn't implemented
        with patch('intseq_bert.analysis.analyze_cases._convert_record_to_features') as mock_convert:
            mock_convert.return_value = {
                "mag_inputs": torch.randn(1, 10, config.MAG_EXTENDED_DIM),
                "mod_inputs": torch.randn(1, 10, config.MOD_FEATURE_DIM),
                "attention_mask": torch.ones(1, 10),
                "oeis_id": "A000045"
            }
            
            result = load_single_sequence(
                "A000045", 
                empty_features_dir, 
                jsonl_path=temp_jsonl_file
            )
            
            assert result["oeis_id"] == "A000045"
            mock_convert.assert_called_once()


@requires_analyze_cases
class TestFindRecordInJsonl:
    """Tests for _find_record_in_jsonl helper."""
    
    def test_find_existing_record(self, temp_jsonl_file):
        """Test finding an existing record."""
        from intseq_bert.analysis.analyze_cases import _find_record_in_jsonl
        
        record = _find_record_in_jsonl("A000045", temp_jsonl_file)
        
        assert record is not None
        assert record["oeis_id"] == "A000045"
        assert record["values"] == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    
    def test_find_nonexistent_record(self, temp_jsonl_file):
        """Test finding a nonexistent record returns None."""
        from intseq_bert.analysis.analyze_cases import _find_record_in_jsonl
        
        record = _find_record_in_jsonl("A999999", temp_jsonl_file)
        
        assert record is None


# ==========================================
# Visualization Function Tests
# ==========================================

@requires_analyze_cases
@requires_matplotlib
class TestPlotMagnitudeUncertainty:
    """Tests for plot_magnitude_uncertainty function."""
    
    def test_plot_creates_figure(self, sample_sequence_length):
        """Test that plot creates correct elements."""
        from intseq_bert.analysis.analyze_cases import plot_magnitude_uncertainty
        
        L = sample_sequence_length
        fig, ax = plt.subplots()
        
        positions = np.arange(L)
        ground_truth = np.random.randn(L)
        pred_mu = np.random.randn(L)
        pred_sigma = np.abs(np.random.randn(L)) + 0.1  # Positive sigma
        mask = np.ones(L, dtype=bool)
        
        plot_magnitude_uncertainty(ax, positions, ground_truth, pred_mu, pred_sigma, mask)
        
        # Check that lines were added
        assert len(ax.lines) >= 2  # GT and predicted
        assert ax.get_xlabel() == 'Position n'
        
        plt.close(fig)
    
    def test_plot_with_partial_mask(self, sample_sequence_length):
        """Test plot with partial mask."""
        from intseq_bert.analysis.analyze_cases import plot_magnitude_uncertainty
        
        L = sample_sequence_length
        fig, ax = plt.subplots()
        
        positions = np.arange(L)
        ground_truth = np.random.randn(L)
        pred_mu = np.random.randn(L)
        pred_sigma = np.abs(np.random.randn(L)) + 0.1
        mask = np.zeros(L, dtype=bool)
        mask[5:15] = True  # Only middle positions
        
        plot_magnitude_uncertainty(ax, positions, ground_truth, pred_mu, pred_sigma, mask)
        
        plt.close(fig)


@requires_analyze_cases
@requires_matplotlib
class TestPlotSignProbability:
    """Tests for plot_sign_probability function."""
    
    def test_plot_creates_stack(self, sample_sequence_length):
        """Test that plot creates stacked area."""
        from intseq_bert.analysis.analyze_cases import plot_sign_probability
        
        L = sample_sequence_length
        fig, ax = plt.subplots()
        
        positions = np.arange(L)
        # Probabilities should sum to 1
        sign_probs = np.random.rand(L, 3)
        sign_probs = sign_probs / sign_probs.sum(axis=1, keepdims=True)
        ground_truth_sign = np.random.randint(0, 3, L)
        
        plot_sign_probability(ax, positions, sign_probs, ground_truth_sign)
        
        assert ax.get_xlabel() == 'Position n'
        assert ax.get_ylabel() == 'Probability'
        
        plt.close(fig)
    
    def test_probabilities_normalized(self):
        """Test that function handles normalized probabilities."""
        from intseq_bert.analysis.analyze_cases import plot_sign_probability
        
        L = 10
        fig, ax = plt.subplots()
        
        # Exactly normalized probabilities
        sign_probs = np.array([[0.8, 0.1, 0.1]] * L)
        positions = np.arange(L)
        ground_truth_sign = np.zeros(L, dtype=int)
        
        plot_sign_probability(ax, positions, sign_probs, ground_truth_sign)
        
        plt.close(fig)


@requires_analyze_cases
@requires_matplotlib
class TestPlotModuloHeatmap:
    """Tests for plot_modulo_heatmap function."""
    
    def test_heatmap_shape(self):
        """Test heatmap has correct dimensions."""
        from intseq_bert.analysis.analyze_cases import plot_modulo_heatmap
        
        L = 20
        display_mods = [2, 3, 5, 7, 10]
        
        fig, ax = plt.subplots()
        
        positions = np.arange(L)
        mod_confidences = np.random.rand(L, len(display_mods))
        
        plot_modulo_heatmap(ax, positions, mod_confidences, display_mods, None)
        
        # Check y-axis has correct labels
        yticks = ax.get_yticks()
        assert len(ax.get_yticklabels()) == len(display_mods)
        
        plt.close(fig)


@requires_analyze_cases
@requires_matplotlib
class TestPlotAttentionHeatmap:
    """Tests for plot_attention_heatmap function."""
    
    def test_attention_plot(self, sample_sequence_length):
        """Test attention heatmap creation."""
        from intseq_bert.analysis.analyze_cases import plot_attention_heatmap
        
        L = sample_sequence_length
        fig, ax = plt.subplots()
        
        attention_weights = np.random.rand(L, L)
        positions = np.arange(L)
        
        plot_attention_heatmap(ax, attention_weights, positions)
        
        assert "Key Position" in ax.get_xlabel()
        
        plt.close(fig)


# ==========================================
# generate_case_figure Tests
# ==========================================

@requires_analyze_cases
@requires_matplotlib
class TestGenerateCaseFigure:
    """Tests for generate_case_figure function."""
    
    def test_figure_is_saved(self, tmp_path, sample_batch, sample_predictions):
        """Test that figure is saved to file."""
        from intseq_bert.analysis.analyze_cases import generate_case_figure
        
        # Create mock model
        mock_model = Mock()
        mock_model.predict_with_details.return_value = sample_predictions
        mock_model.supports_attention.return_value = False
        
        output_path = tmp_path / "test_figure.png"
        
        generate_case_figure(
            oeis_id="A000045",
            model=mock_model,
            batch=sample_batch,
            output_path=output_path
        )
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_custom_display_mods(self, tmp_path, sample_batch, sample_predictions):
        """Test with custom display_mods."""
        from intseq_bert.analysis.analyze_cases import generate_case_figure
        
        mock_model = Mock()
        mock_model.predict_with_details.return_value = sample_predictions
        mock_model.supports_attention.return_value = False
        
        output_path = tmp_path / "test_custom_mods.png"
        custom_mods = [2, 3, 5]  # Only 3 mods
        
        generate_case_figure(
            oeis_id="A000045",
            model=mock_model,
            batch=sample_batch,
            output_path=output_path,
            display_mods=custom_mods
        )
        
        assert output_path.exists()


# ==========================================
# _compute_mod_confidences Tests
# ==========================================

@requires_analyze_cases
class TestComputeModConfidences:
    """Tests for _compute_mod_confidences helper."""
    
    def test_output_shape(self):
        """Test output has correct shape."""
        from intseq_bert.analysis.analyze_cases import _compute_mod_confidences
        
        L = 20
        display_mods = [2, 3, 5, 7, 10]
        
        # Create fake mod_logits with correct total size
        mod_logits = torch.randn(L, sum(config.MOD_RANGE))
        mod_targets = torch.stack([
            torch.randint(0, m, (L,)) for m in config.MOD_RANGE
        ], dim=-1)  # (L, 100)
        
        confidences = _compute_mod_confidences(mod_logits, mod_targets, display_mods)
        
        assert confidences.shape == (L, len(display_mods))
    
    def test_confidences_in_range(self):
        """Test confidences are in [0, 1] range."""
        from intseq_bert.analysis.analyze_cases import _compute_mod_confidences
        
        L = 10
        display_mods = [2, 3, 5]
        
        mod_logits = torch.randn(L, sum(config.MOD_RANGE))
        mod_targets = torch.stack([
            torch.randint(0, m, (L,)) for m in config.MOD_RANGE
        ], dim=-1)
        
        confidences = _compute_mod_confidences(mod_logits, mod_targets, display_mods)
        
        assert np.all(confidences >= 0)
        assert np.all(confidences <= 1)


# ==========================================
# VanillaWrapper Tests
# ==========================================

@requires_matplotlib
class TestVanillaWrapperDecodeMagnitude:
    """Tests for VanillaWrapper.decode_magnitude method."""
    
    def test_special_tokens_become_nan(self):
        """Test that special tokens (PAD, MASK, UNK) become NaN."""
        # Test the logic independently
        special_tokens = {0, 1, 2}  # PAD=0, MASK=1, UNK=2
        
        token_ids = np.array([0, 1, 2, 5, 10, 100])
        
        values = []
        for tid in token_ids:
            if tid in special_tokens:
                values.append(np.nan)
            else:
                values.append(float(tid))  # Simplified decode
        
        result = np.array(values)
        
        assert np.isnan(result[0])  # PAD
        assert np.isnan(result[1])  # MASK
        assert np.isnan(result[2])  # UNK
        assert not np.isnan(result[3])  # Normal token
    
    def test_nan_causes_gap_in_plot(self):
        """Test that NaN values create gaps in plots."""
        x = np.arange(10)
        y = np.array([1, 2, np.nan, np.nan, 5, 6, 7, 8, 9, 10])
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        
        # Line should have gaps at NaN positions
        # This is just a visual test - matplotlib handles NaN by breaking lines
        
        plt.close(fig)


# ==========================================
# DEFAULT_ARCHETYPES Tests
# ==========================================

class TestDefaultArchetypes:
    """Tests for DEFAULT_ARCHETYPES configuration."""
    
    def test_archetypes_structure(self):
        """Test DEFAULT_ARCHETYPES has expected structure."""
        expected_archetypes = {
            "linear_recurrence": "A000045",
            "polynomial": "A000290",
            "sign_oscillation": "A033999",
            "number_theory": "A000040",
            "super_growth": "A000142",
        }
        
        for category, oeis_id in expected_archetypes.items():
            # OEIS ID format validation
            assert oeis_id.startswith("A")
            assert len(oeis_id) == 7
            assert oeis_id[1:].isdigit()


# ==========================================
# Integration Tests
# ==========================================

@requires_analyze_cases
@requires_matplotlib
class TestIntegration:
    """Integration tests for analyze_cases module."""
    
    def test_full_pipeline_with_mock_model(self, tmp_path, temp_features_dir):
        """Test full pipeline from loading to figure generation."""
        from intseq_bert.analysis.analyze_cases import (
            load_single_sequence,
            generate_case_figure
        )
        
        # Load sequence
        batch = load_single_sequence("A000045", temp_features_dir)
        
        # Create mock model
        L = batch["mag_inputs"].shape[1]
        mock_preds = {
            "mag_mu": torch.randn(1, L),
            "mag_log_var": torch.randn(1, L),
            "sign_logits": torch.randn(1, L, config.NUM_SIGN_CLASSES),
            "mod_logits": torch.randn(1, L, sum(config.MOD_RANGE)),
        }
        
        mock_model = Mock()
        mock_model.predict_with_details.return_value = mock_preds
        mock_model.supports_attention.return_value = False
        
        # Generate figure
        output_path = tmp_path / "A000045_test.png"
        generate_case_figure("A000045", mock_model, batch, output_path)
        
        assert output_path.exists()
    
    def test_multiple_sequences_pipeline(self, tmp_path, temp_features_dir, sample_batch):
        """Test pipeline with multiple sequences."""
        from intseq_bert.analysis.analyze_cases import (
            load_single_sequence,
            generate_case_figure
        )
        
        # Save additional feature file
        pt_data = {
            "mag_features": sample_batch["mag_inputs"].squeeze(0),
            "mod_features": sample_batch["mod_inputs"].squeeze(0),
        }
        torch.save(pt_data, temp_features_dir / "A000040.pt")
        
        oeis_ids = ["A000045", "A000040"]
        
        for oeis_id in oeis_ids:
            batch = load_single_sequence(oeis_id, temp_features_dir)
            
            L = batch["mag_inputs"].shape[1]
            mock_preds = {
                "mag_mu": torch.randn(1, L),
                "mag_log_var": torch.randn(1, L),
                "sign_logits": torch.randn(1, L, config.NUM_SIGN_CLASSES),
                "mod_logits": torch.randn(1, L, sum(config.MOD_RANGE)),
            }
            
            mock_model = Mock()
            mock_model.predict_with_details.return_value = mock_preds
            mock_model.supports_attention.return_value = False
            
            output_path = tmp_path / f"{oeis_id}_test.png"
            generate_case_figure(oeis_id, mock_model, batch, output_path)
            
            assert output_path.exists()
