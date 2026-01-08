"""
Tests for evaluate_final.py (Final Evaluation Script).
"""

import pytest
import torch
import json
import math
from pathlib import Path
from typing import Dict, Any

from intseq_bert import evaluate_final


# ==========================================
# 1. calculate_metrics Tests
# ==========================================

class TestCalculateMetrics:
    """Tests for calculate_metrics function."""
    
    def test_top1_correct(self):
        """Test when target is top-1 prediction."""
        candidates = [(42, 0.1), (50, 0.2), (60, 0.3)]
        result = evaluate_final.calculate_metrics(42, candidates, pred_mag=40.0)
        
        assert result["top1"] == True
        assert result["top5"] == True
        assert result["top10"] == True
    
    def test_top5_correct(self):
        """Test when target is in top-5 but not top-1."""
        candidates = [(10, 0.1), (20, 0.2), (30, 0.3), (42, 0.4), (50, 0.5)]
        result = evaluate_final.calculate_metrics(42, candidates, pred_mag=40.0)
        
        assert result["top1"] == False
        assert result["top5"] == True
        assert result["top10"] == True
    
    def test_top10_correct(self):
        """Test when target is in top-10 but not top-5."""
        candidates = [(i, 0.1) for i in range(10)]  # 0-9
        candidates[7] = (42, 0.1)  # Put 42 at position 7
        result = evaluate_final.calculate_metrics(42, candidates, pred_mag=40.0)
        
        assert result["top1"] == False
        assert result["top5"] == False
        assert result["top10"] == True
    
    def test_not_in_candidates(self):
        """Test when target is not in candidates."""
        candidates = [(10, 0.1), (20, 0.2), (30, 0.3)]
        result = evaluate_final.calculate_metrics(999, candidates, pred_mag=40.0)
        
        assert result["top1"] == False
        assert result["top5"] == False
        assert result["top10"] == False
    
    def test_empty_candidates(self):
        """Test with empty candidates list."""
        result = evaluate_final.calculate_metrics(42, [], pred_mag=40.0)
        
        assert result["top1"] == False
        assert result["top5"] == False
        assert result["top10"] == False
    
    def test_magnitude_error_calculation(self):
        """Test magnitude error calculation."""
        # target=100 -> log10(101) ≈ 2.004
        # pred_mag=1000 -> log10(1001) ≈ 3.000
        candidates = [(100, 0.1)]
        result = evaluate_final.calculate_metrics(100, candidates, pred_mag=1000.0)
        
        expected_error = abs(math.log10(101) - math.log10(1001))
        assert result["mag_error"] == pytest.approx(expected_error, rel=1e-5)
    
    def test_target_log_mag(self):
        """Test target log magnitude is computed correctly."""
        candidates = [(100, 0.1)]
        result = evaluate_final.calculate_metrics(100, candidates, pred_mag=100.0)
        
        expected = math.log10(101)
        assert result["target_log_mag"] == pytest.approx(expected, rel=1e-5)
    
    def test_negative_target(self):
        """Test with negative target value."""
        candidates = [(-50, 0.1)]
        result = evaluate_final.calculate_metrics(-50, candidates, pred_mag=50.0)
        
        assert result["top1"] == True
        # log10(|-50| + 1) = log10(51)
        assert result["target_log_mag"] == pytest.approx(math.log10(51), rel=1e-5)
    
    def test_zero_target(self):
        """Test with zero target value."""
        candidates = [(0, 0.1)]
        result = evaluate_final.calculate_metrics(0, candidates, pred_mag=0.0)
        
        assert result["top1"] == True
        # log10(0 + 1) = 0
        assert result["target_log_mag"] == 0.0


# ==========================================
# 2. create_empty_results Tests
# ==========================================

class TestCreateEmptyResults:
    """Tests for create_empty_results function."""
    
    def test_default_structure(self):
        """Test default empty results structure."""
        results = evaluate_final.create_empty_results()
        
        assert "config" in results
        assert "summary" in results
        assert "details_by_magnitude" in results
        assert "logs" in results
    
    def test_summary_fields(self):
        """Test summary has all required fields."""
        results = evaluate_final.create_empty_results()
        
        assert results["summary"]["total"] == 0
        assert results["summary"]["correct_top1"] == 0
        assert results["summary"]["correct_top5"] == 0
        assert results["summary"]["correct_top10"] == 0
        assert results["summary"]["total_mag_error"] == 0.0
    
    def test_with_config(self):
        """Test with config provided."""
        config = {"model_path": "test.pt", "beam_width": 20}
        results = evaluate_final.create_empty_results(config)
        
        assert results["config"] == config


# ==========================================
# 3. update_results Tests
# ==========================================

class TestUpdateResults:
    """Tests for update_results function."""
    
    def test_updates_total(self):
        """Test that total is incremented."""
        results = evaluate_final.create_empty_results()
        metrics = {"top1": False, "top5": False, "top10": False, 
                   "mag_error": 0.5, "target_log_mag": 2.0}
        record = {"oeis_id": "A000001", "sequence": [1, 2, 3]}
        output = {"candidates": [], "predicted_magnitude": 100}
        
        evaluate_final.update_results(results, metrics, record, output, log_sample=False)
        
        assert results["summary"]["total"] == 1
    
    def test_updates_correct_counts(self):
        """Test that correct counts are updated."""
        results = evaluate_final.create_empty_results()
        metrics = {"top1": True, "top5": True, "top10": True, 
                   "mag_error": 0.1, "target_log_mag": 2.0}
        record = {"oeis_id": "A000001", "sequence": [1, 2, 3]}
        output = {"candidates": [(3, 0.1)], "predicted_magnitude": 3}
        
        evaluate_final.update_results(results, metrics, record, output, log_sample=False)
        
        assert results["summary"]["correct_top1"] == 1
        assert results["summary"]["correct_top5"] == 1
        assert results["summary"]["correct_top10"] == 1
    
    def test_updates_magnitude_bucket(self):
        """Test that magnitude bucket is updated."""
        results = evaluate_final.create_empty_results()
        metrics = {"top1": True, "top5": True, "top10": True, 
                   "mag_error": 0.1, "target_log_mag": 2.5}  # bucket = 2
        record = {"oeis_id": "A000001", "sequence": [1, 2, 100]}
        output = {"candidates": [(100, 0.1)], "predicted_magnitude": 100}
        
        evaluate_final.update_results(results, metrics, record, output, log_sample=False)
        
        assert 2 in results["details_by_magnitude"]
        assert results["details_by_magnitude"][2]["total"] == 1
        assert results["details_by_magnitude"][2]["correct"] == 1
    
    def test_adds_log_when_requested(self):
        """Test that log is added when log_sample=True."""
        results = evaluate_final.create_empty_results()
        metrics = {"top1": False, "top5": False, "top10": False, 
                   "mag_error": 1.5, "target_log_mag": 2.0}
        record = {"oeis_id": "A000001", "sequence": [1, 2, 100]}
        output = {"candidates": [(50, 0.1), (60, 0.2)], "predicted_magnitude": 50}
        
        evaluate_final.update_results(results, metrics, record, output, log_sample=True)
        
        assert len(results["logs"]) == 1
        assert results["logs"][0]["oeis_id"] == "A000001"
        assert results["logs"][0]["target"] == 100
        assert results["logs"][0]["correct"] == False
    
    def test_no_log_when_not_requested(self):
        """Test that no log is added when log_sample=False."""
        results = evaluate_final.create_empty_results()
        metrics = {"top1": True, "top5": True, "top10": True, 
                   "mag_error": 0.1, "target_log_mag": 2.0}
        record = {"oeis_id": "A000001", "sequence": [1, 2, 100]}
        output = {"candidates": [(100, 0.1)], "predicted_magnitude": 100}
        
        evaluate_final.update_results(results, metrics, record, output, log_sample=False)
        
        assert len(results["logs"]) == 0


# ==========================================
# 4. load_sequences_by_ids Tests
# ==========================================

class TestLoadSequencesByIds:
    """Tests for load_sequences_by_ids function."""
    
    def test_filters_by_ids(self, tmp_path):
        """Test that only matching IDs are loaded."""
        jsonl_path = tmp_path / "test.jsonl"
        
        with open(jsonl_path, 'w') as f:
            f.write('{"oeis_id": "A000001", "sequence": [1, 2, 3]}\n')
            f.write('{"oeis_id": "A000002", "sequence": [2, 4, 6]}\n')
            f.write('{"oeis_id": "A000003", "sequence": [3, 6, 9]}\n')
        
        target_ids = {"A000001", "A000003"}
        result = evaluate_final.load_sequences_by_ids(
            str(jsonl_path), target_ids, verbose=False
        )
        
        assert len(result) == 2
        ids = {r["oeis_id"] for r in result}
        assert ids == {"A000001", "A000003"}
    
    def test_empty_target_ids(self, tmp_path):
        """Test with empty target IDs set."""
        jsonl_path = tmp_path / "test.jsonl"
        
        with open(jsonl_path, 'w') as f:
            f.write('{"oeis_id": "A000001", "sequence": [1, 2, 3]}\n')
        
        result = evaluate_final.load_sequences_by_ids(
            str(jsonl_path), set(), verbose=False
        )
        
        assert len(result) == 0
    
    def test_handles_invalid_json(self, tmp_path):
        """Test that invalid JSON lines are skipped."""
        jsonl_path = tmp_path / "test.jsonl"
        
        with open(jsonl_path, 'w') as f:
            f.write('{"oeis_id": "A000001", "sequence": [1, 2, 3]}\n')
            f.write('invalid json line\n')
            f.write('{"oeis_id": "A000002", "sequence": [2, 4, 6]}\n')
        
        target_ids = {"A000001", "A000002"}
        result = evaluate_final.load_sequences_by_ids(
            str(jsonl_path), target_ids, verbose=False
        )
        
        assert len(result) == 2


# ==========================================
# 5. Integration Tests
# ==========================================

class TestEvaluationIntegration:
    """Integration tests for evaluation workflow."""
    
    def test_full_metrics_workflow(self):
        """Test complete workflow: calculate_metrics -> update_results."""
        results = evaluate_final.create_empty_results({"test": True})
        
        # Simulate evaluating 3 samples
        test_cases = [
            {"target": 10, "candidates": [(10, 0.1)], "pred_mag": 10},  # top1 hit
            {"target": 20, "candidates": [(5, 0.1), (20, 0.2)], "pred_mag": 15},  # top5 hit
            {"target": 999, "candidates": [(1, 0.1)], "pred_mag": 1},  # miss
        ]
        
        for i, tc in enumerate(test_cases):
            metrics = evaluate_final.calculate_metrics(
                tc["target"], tc["candidates"], tc["pred_mag"]
            )
            record = {"oeis_id": f"A{i:06d}", "sequence": [1, 2, tc['target']]}
            output = {"candidates": tc["candidates"], "predicted_magnitude": tc["pred_mag"]}
            evaluate_final.update_results(results, metrics, record, output, log_sample=True)
        
        assert results["summary"]["total"] == 3
        assert results["summary"]["correct_top1"] == 1
        assert results["summary"]["correct_top5"] == 2  # top1 also counts as top5
        assert len(results["logs"]) == 3
