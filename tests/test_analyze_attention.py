"""
test_analyze_attention.py:
Unit tests for analyze_attention.py module.
Tests AttentionExtractor, visualization functions, and recurrence pattern analysis.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from intseq_bert import config

# Optional dependency
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

# Module import check
try:
    from intseq_bert.analysis import analyze_attention
    HAS_ANALYZE_ATTENTION = True
except ImportError:
    HAS_ANALYZE_ATTENTION = False


# ==========================================
# Test Fixtures
# ==========================================

@pytest.fixture
def sample_num_layers():
    return 4


@pytest.fixture
def sample_num_heads():
    return 8


@pytest.fixture
def sample_sequence_length():
    return 32


@pytest.fixture
def sample_attention_tensor(sample_num_layers, sample_num_heads, sample_sequence_length):
    """Creates sample attention tensor: (num_layers, num_heads, L, L)."""
    L = sample_sequence_length
    # Create random attention (softmax-like - rows sum to 1)
    attn = torch.rand(sample_num_layers, sample_num_heads, L, L)
    attn = attn / attn.sum(dim=-1, keepdim=True)  # Normalize rows
    return attn


@pytest.fixture
def sample_batch_attention(sample_num_layers, sample_num_heads, sample_sequence_length):
    """Creates sample batched attention: (num_layers, B, num_heads, L, L)."""
    B = 1
    L = sample_sequence_length
    attn = torch.rand(sample_num_layers, B, sample_num_heads, L, L)
    attn = attn / attn.sum(dim=-1, keepdim=True)
    return attn


@pytest.fixture
def recurrence_attention(sample_num_heads, sample_sequence_length):
    """Creates attention with strong recurrence pattern (attention to n-1, n-2)."""
    L = sample_sequence_length
    num_layers = 4
    
    # Create attention that focuses on previous positions
    attn = torch.zeros(num_layers, sample_num_heads, L, L)
    for layer in range(num_layers):
        for head in range(sample_num_heads):
            for q in range(L):
                if q >= 2:
                    attn[layer, head, q, q-1] = 0.4  # n-1
                    attn[layer, head, q, q-2] = 0.3  # n-2
                    attn[layer, head, q, q] = 0.2    # self
                    # Distribute rest
                    remaining = 0.1
                    for k in range(q-2):
                        attn[layer, head, q, k] = remaining / max(1, q-2)
                elif q == 1:
                    attn[layer, head, q, q-1] = 0.6
                    attn[layer, head, q, q] = 0.4
                else:
                    attn[layer, head, q, q] = 1.0
    
    return attn


# ==========================================
# Markers for conditional skipping
# ==========================================

requires_analyze_attention = pytest.mark.skipif(
    not HAS_ANALYZE_ATTENTION,
    reason="analyze_attention module not implemented yet"
)

requires_matplotlib = pytest.mark.skipif(
    not HAS_MATPLOTLIB,
    reason="matplotlib not installed"
)


# ==========================================
# AttentionExtractor Tests
# ==========================================

@requires_analyze_attention
class TestAttentionExtractor:
    """Tests for AttentionExtractor class."""
    
    def test_initialization(self):
        """Test AttentionExtractor initializes correctly."""
        from intseq_bert.analysis.analyze_attention import AttentionExtractor
        
        mock_model = Mock()
        extractor = AttentionExtractor(mock_model)
        
        assert extractor.model is mock_model
        assert extractor.attention_weights == []
        assert extractor.hooks == []
    
    def test_clear(self):
        """Test clear method resets attention_weights."""
        from intseq_bert.analysis.analyze_attention import AttentionExtractor
        
        mock_model = Mock()
        extractor = AttentionExtractor(mock_model)
        extractor.attention_weights = [torch.rand(1, 8, 32, 32)]
        
        extractor.clear()
        
        assert extractor.attention_weights == []
    
    def test_get_attention_tensor_shape(self, sample_batch_attention):
        """Test get_attention_tensor returns correct shape."""
        from intseq_bert.analysis.analyze_attention import AttentionExtractor
        
        mock_model = Mock()
        extractor = AttentionExtractor(mock_model)
        
        # Simulate collected attention from 4 layers
        for layer_attn in sample_batch_attention:
            extractor.attention_weights.append(layer_attn)
        
        result = extractor.get_attention_tensor()
        
        assert result.shape[0] == sample_batch_attention.shape[0]  # num_layers


# ==========================================
# analyze_recurrence_pattern Tests
# ==========================================

@requires_analyze_attention
class TestAnalyzeRecurrencePattern:
    """Tests for analyze_recurrence_pattern function."""
    
    def test_output_keys(self, sample_attention_tensor):
        """Test output contains expected keys."""
        from intseq_bert.analysis.analyze_attention import analyze_recurrence_pattern
        
        result = analyze_recurrence_pattern(sample_attention_tensor)
        
        expected_keys = {"prev_1_ratio", "prev_2_ratio", "diagonal_ratio", "total_local_ratio"}
        assert set(result.keys()) == expected_keys
    
    def test_ratios_in_range(self, sample_attention_tensor):
        """Test all ratios are in [0, 1] range."""
        from intseq_bert.analysis.analyze_attention import analyze_recurrence_pattern
        
        result = analyze_recurrence_pattern(sample_attention_tensor)
        
        for key, value in result.items():
            assert 0.0 <= value <= 1.0, f"{key} = {value} out of range"
    
    def test_recurrence_attention_high_local_ratio(self, recurrence_attention):
        """Test recurrence-focused attention has high local ratio."""
        from intseq_bert.analysis.analyze_attention import analyze_recurrence_pattern
        
        result = analyze_recurrence_pattern(recurrence_attention)
        
        # Should have high prev_1 and prev_2 ratios
        assert result["prev_1_ratio"] > 0.2
        assert result["prev_2_ratio"] > 0.1
        assert result["total_local_ratio"] > 0.5
    
    def test_uniform_attention_low_local_ratio(self, sample_num_heads, sample_sequence_length):
        """Test uniform attention has lower local ratio."""
        from intseq_bert.analysis.analyze_attention import analyze_recurrence_pattern
        
        L = sample_sequence_length
        num_layers = 4
        
        # Create uniform attention (equal attention to all positions)
        attn = torch.ones(num_layers, sample_num_heads, L, L) / L
        
        result = analyze_recurrence_pattern(attn)
        
        # Local ratio should be small since attention is spread uniformly
        assert result["total_local_ratio"] < 0.3


# ==========================================
# EXPECTED_PATTERNS Tests
# ==========================================

class TestExpectedPatterns:
    """Tests for EXPECTED_PATTERNS configuration."""
    
    @requires_analyze_attention
    def test_expected_patterns_exists(self):
        """Test EXPECTED_PATTERNS is defined."""
        from intseq_bert.analysis.analyze_attention import EXPECTED_PATTERNS
        
        assert isinstance(EXPECTED_PATTERNS, dict)
        assert len(EXPECTED_PATTERNS) > 0
    
    @requires_analyze_attention
    def test_fibonacci_pattern(self):
        """Test Fibonacci has correct expected pattern."""
        from intseq_bert.analysis.analyze_attention import EXPECTED_PATTERNS
        
        assert "A000045" in EXPECTED_PATTERNS
        fib = EXPECTED_PATTERNS["A000045"]
        assert fib["type"] == "linear_recurrence"
        assert fib["recurrence_depth"] == 2
    
    @requires_analyze_attention
    def test_primes_pattern(self):
        """Test Primes has non-local pattern."""
        from intseq_bert.analysis.analyze_attention import EXPECTED_PATTERNS
        
        assert "A000040" in EXPECTED_PATTERNS
        primes = EXPECTED_PATTERNS["A000040"]
        assert primes["type"] == "non_local"


# ==========================================
# check_pattern_alignment Tests
# ==========================================

@requires_analyze_attention
class TestCheckPatternAlignment:
    """Tests for check_pattern_alignment function."""
    
    def test_aligned_recurrence(self):
        """Test aligned detection for recurrence patterns."""
        from intseq_bert.analysis.analyze_attention import check_pattern_alignment
        
        stats = {
            "prev_1_ratio": 0.3,
            "prev_2_ratio": 0.2,
            "diagonal_ratio": 0.1,
            "total_local_ratio": 0.7  # High local ratio
        }
        
        result = check_pattern_alignment("A000045", stats)
        assert result == "ALIGNED"
    
    def test_misaligned_recurrence(self):
        """Test misaligned detection for recurrence patterns."""
        from intseq_bert.analysis.analyze_attention import check_pattern_alignment
        
        stats = {
            "prev_1_ratio": 0.1,
            "prev_2_ratio": 0.05,
            "diagonal_ratio": 0.1,
            "total_local_ratio": 0.3  # Low local ratio
        }
        
        result = check_pattern_alignment("A000045", stats)
        assert result == "MISALIGNED"
    
    def test_unknown_sequence(self):
        """Test unknown sequence ID returns UNKNOWN."""
        from intseq_bert.analysis.analyze_attention import check_pattern_alignment
        
        stats = {"prev_1_ratio": 0.3, "prev_2_ratio": 0.2, 
                 "diagonal_ratio": 0.1, "total_local_ratio": 0.6}
        
        result = check_pattern_alignment("A999999", stats)
        assert result == "UNKNOWN"


# ==========================================
# Visualization Function Tests
# ==========================================

@requires_analyze_attention
@requires_matplotlib
class TestPlotLayerwiseAttention:
    """Tests for plot_layerwise_attention function."""
    
    def test_creates_figure(self, sample_attention_tensor, tmp_path):
        """Test that function creates and saves figure."""
        from intseq_bert.analysis.analyze_attention import plot_layerwise_attention
        
        output_path = tmp_path / "layerwise.png"
        plot_layerwise_attention(sample_attention_tensor, output_path, "A000045")
        
        assert output_path.exists()
    
    def test_with_valid_len_trims(self, sample_attention_tensor, tmp_path):
        """Test valid_len parameter trims attention."""
        from intseq_bert.analysis.analyze_attention import plot_layerwise_attention
        
        output_path = tmp_path / "layerwise_trimmed.png"
        valid_len = 16  # Half of sample_sequence_length
        
        plot_layerwise_attention(
            sample_attention_tensor, output_path, "A000045", 
            valid_len=valid_len
        )
        
        assert output_path.exists()
    
    def test_with_layer_ids(self, sample_attention_tensor, tmp_path):
        """Test with specific layer IDs."""
        from intseq_bert.analysis.analyze_attention import plot_layerwise_attention
        
        output_path = tmp_path / "layerwise_subset.png"
        plot_layerwise_attention(
            sample_attention_tensor, output_path, "A000045",
            layer_ids=[0, 2]
        )
        
        assert output_path.exists()


@requires_analyze_attention
@requires_matplotlib
class TestPlotHeadwiseAttention:
    """Tests for plot_headwise_attention function."""
    
    def test_creates_figure(self, sample_attention_tensor, tmp_path):
        """Test that function creates and saves figure."""
        from intseq_bert.analysis.analyze_attention import plot_headwise_attention
        
        output_path = tmp_path / "headwise.png"
        plot_headwise_attention(sample_attention_tensor, 0, output_path, "A000045")
        
        assert output_path.exists()
    
    def test_with_valid_len(self, sample_attention_tensor, tmp_path):
        """Test with valid_len trimming."""
        from intseq_bert.analysis.analyze_attention import plot_headwise_attention
        
        output_path = tmp_path / "headwise_trimmed.png"
        plot_headwise_attention(
            sample_attention_tensor, 0, output_path, "A000045",
            valid_len=16
        )
        
        assert output_path.exists()


@requires_analyze_attention
@requires_matplotlib
class TestPlotAggregatedAttention:
    """Tests for plot_aggregated_attention function."""
    
    def test_creates_figure(self, sample_attention_tensor, tmp_path):
        """Test that function creates and saves figure."""
        from intseq_bert.analysis.analyze_attention import plot_aggregated_attention
        
        output_path = tmp_path / "aggregated.png"
        plot_aggregated_attention(sample_attention_tensor, output_path, "A000045")
        
        assert output_path.exists()
    
    def test_with_valid_len(self, sample_attention_tensor, tmp_path):
        """Test with valid_len trimming."""
        from intseq_bert.analysis.analyze_attention import plot_aggregated_attention
        
        output_path = tmp_path / "aggregated_trimmed.png"
        plot_aggregated_attention(
            sample_attention_tensor, output_path, "A000045",
            valid_len=16
        )
        
        assert output_path.exists()


# ==========================================
# Integration Tests
# ==========================================

@requires_analyze_attention
class TestIntegration:
    """Integration tests for analyze_attention module."""
    
    def test_full_analysis_pipeline(self, sample_attention_tensor):
        """Test full analysis pipeline from attention to stats."""
        from intseq_bert.analysis.analyze_attention import (
            analyze_recurrence_pattern,
            check_pattern_alignment
        )
        
        # Analyze pattern
        stats = analyze_recurrence_pattern(sample_attention_tensor)
        
        # Check alignment (might be UNKNOWN since random)
        result = check_pattern_alignment("A000045", stats)
        
        assert result in {"ALIGNED", "MISALIGNED", "UNKNOWN"}
    
    @requires_matplotlib
    def test_all_visualization_functions(self, sample_attention_tensor, tmp_path):
        """Test all visualization functions work together."""
        from intseq_bert.analysis.analyze_attention import (
            plot_layerwise_attention,
            plot_headwise_attention,
            plot_aggregated_attention
        )
        
        oeis_id = "A000045"
        
        # Layer-wise
        plot_layerwise_attention(
            sample_attention_tensor, 
            tmp_path / "layer.png", 
            oeis_id
        )
        
        # Head-wise for layer 0
        plot_headwise_attention(
            sample_attention_tensor, 
            0, 
            tmp_path / "head.png", 
            oeis_id
        )
        
        # Aggregated
        plot_aggregated_attention(
            sample_attention_tensor, 
            tmp_path / "agg.png", 
            oeis_id
        )
        
        assert (tmp_path / "layer.png").exists()
        assert (tmp_path / "head.png").exists()
        assert (tmp_path / "agg.png").exists()
