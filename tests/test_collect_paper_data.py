"""
Tests for collect_paper_data.py (Paper Data Collection Script).
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any

from intseq_bert import collect_paper_data


# ==========================================
# 1. create_empty_mod_stats Tests
# ==========================================

class TestCreateEmptyModStats:
    """Tests for create_empty_mod_stats function."""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        result = collect_paper_data.create_empty_mod_stats()
        assert isinstance(result, dict)
    
    def test_contains_all_mods(self):
        """Test that dict contains all mods from 2 to 101."""
        result = collect_paper_data.create_empty_mod_stats()
        
        for m in range(2, 102):
            assert m in result
    
    def test_initial_values_are_zero(self):
        """Test that all counts start at zero."""
        result = collect_paper_data.create_empty_mod_stats()
        
        for m in range(2, 102):
            assert result[m]["correct"] == 0
            assert result[m]["total"] == 0


# ==========================================
# 2. calculate_mod_accuracy_for_single_mod Tests
# ==========================================

class TestCalculateModAccuracyForSingleMod:
    """Tests for calculate_mod_accuracy_for_single_mod function."""
    
    def test_all_correct(self):
        """Test when all predictions are correct."""
        # Logits: (1, 3, 3) - mod 3
        logits = torch.tensor([[[10.0, 0.0, 0.0],   # predicts 0
                                [0.0, 10.0, 0.0],    # predicts 1
                                [0.0, 0.0, 10.0]]])  # predicts 2
        targets = torch.tensor([[0, 1, 2]])
        mask = torch.tensor([[1, 1, 1]])
        
        correct, total = collect_paper_data.calculate_mod_accuracy_for_single_mod(
            logits, targets, mask, mod_size=3
        )
        
        assert correct == 3
        assert total == 3
    
    def test_all_wrong(self):
        """Test when all predictions are wrong."""
        logits = torch.tensor([[[10.0, 0.0, 0.0],   # predicts 0
                                [10.0, 0.0, 0.0],    # predicts 0
                                [10.0, 0.0, 0.0]]])  # predicts 0
        targets = torch.tensor([[1, 2, 1]])  # all different from 0
        mask = torch.tensor([[1, 1, 1]])
        
        correct, total = collect_paper_data.calculate_mod_accuracy_for_single_mod(
            logits, targets, mask, mod_size=3
        )
        
        assert correct == 0
        assert total == 3
    
    def test_partial_correct(self):
        """Test partial correctness."""
        logits = torch.tensor([[[10.0, 0.0],  # predicts 0
                                [0.0, 10.0]]])  # predicts 1
        targets = torch.tensor([[0, 0]])  # only first is correct
        mask = torch.tensor([[1, 1]])
        
        correct, total = collect_paper_data.calculate_mod_accuracy_for_single_mod(
            logits, targets, mask, mod_size=2
        )
        
        assert correct == 1
        assert total == 2
    
    def test_respects_mask(self):
        """Test that only masked positions are evaluated."""
        logits = torch.tensor([[[10.0, 0.0],   # predicts 0 (masked)
                                [0.0, 10.0],    # predicts 1 (not masked)
                                [10.0, 0.0]]])  # predicts 0 (masked)
        targets = torch.tensor([[0, 0, 0]])  # all should be 0
        mask = torch.tensor([[1, 0, 1]])  # only positions 0 and 2
        
        correct, total = collect_paper_data.calculate_mod_accuracy_for_single_mod(
            logits, targets, mask, mod_size=2
        )
        
        assert total == 2  # only 2 masked positions
        assert correct == 2  # both masked positions correct
    
    def test_ignores_negative_100(self):
        """Test that -100 targets are ignored."""
        logits = torch.tensor([[[10.0, 0.0],   # predicts 0
                                [10.0, 0.0]]])  # predicts 0
        targets = torch.tensor([[0, -100]])  # second is padding
        mask = torch.tensor([[1, 1]])
        
        correct, total = collect_paper_data.calculate_mod_accuracy_for_single_mod(
            logits, targets, mask, mod_size=2
        )
        
        assert total == 1  # only 1 valid target
        assert correct == 1
    
    def test_empty_mask(self):
        """Test with no masked positions."""
        logits = torch.tensor([[[10.0, 0.0]]])
        targets = torch.tensor([[0]])
        mask = torch.tensor([[0]])  # nothing masked
        
        correct, total = collect_paper_data.calculate_mod_accuracy_for_single_mod(
            logits, targets, mask, mod_size=2
        )
        
        assert correct == 0
        assert total == 0


# ==========================================
# 3. extract_magnitude_pairs Tests
# ==========================================

class TestExtractMagnitudePairs:
    """Tests for extract_magnitude_pairs function."""
    
    def test_extracts_pairs(self):
        """Test that pairs are extracted correctly."""
        pred_mag = torch.tensor([[[1.0, 0, 0, 0, 0], [2.0, 0, 0, 0, 0]]])  # (1, 2, 5)
        target_mag = torch.tensor([[[1.5, 0, 0, 0, 0], [2.5, 0, 0, 0, 0]]])
        mask = torch.tensor([[1, 1]])
        
        pairs = collect_paper_data.extract_magnitude_pairs(pred_mag, target_mag, mask)
        
        assert len(pairs) == 2
        assert pairs[0] == (1.5, 1.0)  # (target, pred)
        assert pairs[1] == (2.5, 2.0)
    
    def test_respects_mask(self):
        """Test that only masked positions are extracted."""
        pred_mag = torch.tensor([[[1.0, 0, 0, 0, 0], [2.0, 0, 0, 0, 0], [3.0, 0, 0, 0, 0]]])
        target_mag = torch.tensor([[[0.5, 0, 0, 0, 0], [1.5, 0, 0, 0, 0], [2.5, 0, 0, 0, 0]]])
        mask = torch.tensor([[1, 0, 1]])  # skip middle
        
        pairs = collect_paper_data.extract_magnitude_pairs(pred_mag, target_mag, mask)
        
        assert len(pairs) == 2
        assert pairs[0] == (0.5, 1.0)
        assert pairs[1] == (2.5, 3.0)
    
    def test_empty_mask(self):
        """Test with no masked positions."""
        pred_mag = torch.tensor([[[1.0, 0, 0, 0, 0]]])
        target_mag = torch.tensor([[[0.5, 0, 0, 0, 0]]])
        mask = torch.tensor([[0]])
        
        pairs = collect_paper_data.extract_magnitude_pairs(pred_mag, target_mag, mask)
        
        assert len(pairs) == 0


# ==========================================
# 4. calculate_magnitude_statistics Tests
# ==========================================

class TestCalculateMagnitudeStatistics:
    """Tests for calculate_magnitude_statistics function."""
    
    def test_perfect_correlation(self):
        """Test with perfect correlation."""
        scatter_data = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        
        stats = collect_paper_data.calculate_magnitude_statistics(scatter_data)
        
        assert stats["correlation_r"] == pytest.approx(1.0, rel=1e-5)
        assert stats["r_squared"] == pytest.approx(1.0, rel=1e-5)
        assert stats["mae"] == pytest.approx(0.0, abs=1e-5)
        assert stats["rmse"] == pytest.approx(0.0, abs=1e-5)
    
    def test_zero_error(self):
        """Test MAE and RMSE with zero error."""
        scatter_data = [(5.0, 5.0), (10.0, 10.0)]
        
        stats = collect_paper_data.calculate_magnitude_statistics(scatter_data)
        
        assert stats["mae"] == 0.0
        assert stats["rmse"] == 0.0
    
    def test_constant_offset(self):
        """Test with constant prediction offset."""
        scatter_data = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]  # pred = target + 1
        
        stats = collect_paper_data.calculate_magnitude_statistics(scatter_data)
        
        # Perfect correlation despite offset
        assert stats["correlation_r"] == pytest.approx(1.0, rel=1e-5)
        assert stats["mae"] == pytest.approx(1.0, rel=1e-5)
        assert stats["rmse"] == pytest.approx(1.0, rel=1e-5)
    
    def test_empty_data(self):
        """Test with empty scatter data."""
        stats = collect_paper_data.calculate_magnitude_statistics([])
        
        assert stats["correlation_r"] == 0.0
        assert stats["r_squared"] == 0.0
        assert stats["mae"] == 0.0
        assert stats["rmse"] == 0.0
    
    def test_varied_errors(self):
        """Test with varied prediction errors."""
        scatter_data = [(0.0, 1.0), (0.0, -1.0)]  # errors of +1 and -1
        
        stats = collect_paper_data.calculate_magnitude_statistics(scatter_data)
        
        # MAE = mean(|1|, |-1|) = 1.0
        assert stats["mae"] == pytest.approx(1.0, rel=1e-5)
        # RMSE = sqrt(mean(1^2, 1^2)) = 1.0
        assert stats["rmse"] == pytest.approx(1.0, rel=1e-5)


# ==========================================
# 5. aggregate_mod_results Tests
# ==========================================

class TestAggregateModResults:
    """Tests for aggregate_mod_results function."""
    
    def test_calculates_accuracy(self):
        """Test accuracy calculation."""
        mod_stats = {
            2: {"correct": 80, "total": 100},
            3: {"correct": 60, "total": 100}
        }
        # Fill rest with zeros
        for m in range(4, 102):
            mod_stats[m] = {"correct": 0, "total": 0}
        
        results = collect_paper_data.aggregate_mod_results(mod_stats)
        
        assert results[2]["accuracy"] == 0.8
        assert results[3]["accuracy"] == 0.6
    
    def test_handles_zero_total(self):
        """Test that zero total doesn't cause division error."""
        mod_stats = collect_paper_data.create_empty_mod_stats()
        
        results = collect_paper_data.aggregate_mod_results(mod_stats)
        
        for m in range(2, 102):
            assert results[m]["accuracy"] == 0.0
    
    def test_preserves_counts(self):
        """Test that correct and total counts are preserved."""
        mod_stats = collect_paper_data.create_empty_mod_stats()
        mod_stats[10]["correct"] = 42
        mod_stats[10]["total"] = 100
        
        results = collect_paper_data.aggregate_mod_results(mod_stats)
        
        assert results[10]["correct"] == 42
        assert results[10]["total"] == 100


# ==========================================
# 6. calculate_mod_accuracies Tests
# ==========================================

class TestCalculateModAccuracies:
    """Tests for calculate_mod_accuracies function (integration)."""
    
    def test_updates_stats(self):
        """Test that stats dictionary is updated."""
        outputs = {
            "mod2": torch.tensor([[[10.0, 0.0]]]),  # predicts 0
            "mod3": torch.tensor([[[0.0, 10.0, 0.0]]])  # predicts 1
        }
        targets = {
            "mod2": torch.tensor([[0]]),  # correct
            "mod3": torch.tensor([[1]])   # correct
        }
        mask = torch.tensor([[1]])
        mod_stats = collect_paper_data.create_empty_mod_stats()
        
        collect_paper_data.calculate_mod_accuracies(outputs, targets, mask, mod_stats)
        
        assert mod_stats[2]["correct"] == 1
        assert mod_stats[2]["total"] == 1
        assert mod_stats[3]["correct"] == 1
        assert mod_stats[3]["total"] == 1
    
    def test_skips_missing_keys(self):
        """Test that missing keys are skipped without error."""
        outputs = {"mod2": torch.tensor([[[10.0, 0.0]]])}
        targets = {"mod2": torch.tensor([[0]])}
        mask = torch.tensor([[1]])
        mod_stats = collect_paper_data.create_empty_mod_stats()
        
        # Should not raise error for missing mod3-mod101 keys
        collect_paper_data.calculate_mod_accuracies(outputs, targets, mask, mod_stats)
        
        assert mod_stats[2]["total"] == 1
        assert mod_stats[3]["total"] == 0  # not updated
