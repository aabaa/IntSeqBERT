"""
test_analyze_mod_spectrum.py:
Unit tests for analyze_mod_spectrum.py module.
Tests NIG computation, per-modulus metrics, Bootstrap CI, and tag-stratified analysis.
"""

import pytest
import torch
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
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
    return 10


@pytest.fixture
def sample_sequence_length():
    return 16


@pytest.fixture
def sample_mod_logits(sample_num_sequences, sample_sequence_length):
    """Creates sample mod logits: (N, L, sum(MOD_RANGE))."""
    N, L = sample_num_sequences, sample_sequence_length
    total_classes = sum(config.MOD_RANGE)
    return torch.randn(N, L, total_classes)


@pytest.fixture
def sample_mod_targets(sample_num_sequences, sample_sequence_length):
    """Creates sample mod targets: (N, L, 100) with valid class indices."""
    N, L = sample_num_sequences, sample_sequence_length
    targets = torch.stack([
        torch.randint(0, m, (N, L)) for m in config.MOD_RANGE
    ], dim=-1)  # (N, L, 100)
    return targets


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
    tags_list = ["core", "nice", "mult", "prime", "easy"]
    id_to_tags = {}
    for i, oeis_id in enumerate(sample_oeis_ids):
        # Assign 2-3 tags per ID
        id_to_tags[oeis_id] = [tags_list[i % len(tags_list)], tags_list[(i + 1) % len(tags_list)]]
    return id_to_tags


@pytest.fixture
def temp_jsonl_file(tmp_path, sample_oeis_ids, sample_id_to_tags):
    """Creates a temporary JSONL file with sample records."""
    jsonl_path = tmp_path / "data.jsonl"
    
    with open(jsonl_path, "w") as f:
        for oeis_id in sample_oeis_ids:
            record = {
                "oeis_id": oeis_id,
                "values": list(range(1, 11)),
                "keywords": sample_id_to_tags.get(oeis_id, [])
            }
            f.write(json.dumps(record) + "\n")
    
    return jsonl_path


# ==========================================
# Markers for conditional skipping
# ==========================================

try:
    from intseq_bert.analysis import analyze_mod_spectrum
    HAS_ANALYZE_MOD_SPECTRUM = True
except ImportError:
    HAS_ANALYZE_MOD_SPECTRUM = False

requires_analyze_mod_spectrum = pytest.mark.skipif(
    not HAS_ANALYZE_MOD_SPECTRUM,
    reason="analyze_mod_spectrum module not implemented yet"
)


# ==========================================
# compute_nig Tests
# ==========================================

class TestComputeNig:
    """Tests for compute_nig function."""
    
    @requires_analyze_mod_spectrum
    def test_perfect_prediction(self):
        """Test NIG = 1.0 when loss = 0."""
        from intseq_bert.analysis.analyze_mod_spectrum import compute_nig
        
        nig = compute_nig(ce_loss=0.0, modulus=10)
        assert nig == 1.0
    
    @requires_analyze_mod_spectrum
    def test_random_prediction(self):
        """Test NIG ≈ 0.0 when loss = log(m) (random baseline)."""
        from intseq_bert.analysis.analyze_mod_spectrum import compute_nig
        
        modulus = 10
        random_loss = np.log(modulus)
        nig = compute_nig(ce_loss=random_loss, modulus=modulus)
        assert abs(nig) < 1e-6  # Should be ~0
    
    @requires_analyze_mod_spectrum
    def test_worse_than_random(self):
        """Test NIG < 0 when loss > log(m)."""
        from intseq_bert.analysis.analyze_mod_spectrum import compute_nig
        
        modulus = 10
        bad_loss = np.log(modulus) * 2  # Twice the random baseline
        nig = compute_nig(ce_loss=bad_loss, modulus=modulus)
        assert nig < 0
    
    @requires_analyze_mod_spectrum
    def test_nig_range(self):
        """Test NIG is in expected range for typical losses."""
        from intseq_bert.analysis.analyze_mod_spectrum import compute_nig
        
        for m in [2, 10, 100]:
            for loss_ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
                loss = np.log(m) * loss_ratio
                nig = compute_nig(loss, m)
                assert 0.0 <= nig <= 1.0 or (loss_ratio == 1.0 and abs(nig) < 1e-6)


# ==========================================
# split_mod_logits Tests
# ==========================================

@requires_analyze_mod_spectrum
class TestSplitModLogits:
    """Tests for split_mod_logits helper function."""
    
    def test_split_count(self):
        """Test returns correct number of splits."""
        from intseq_bert.analysis.common import split_mod_logits
        
        total_classes = sum(config.MOD_RANGE)
        logits = torch.randn(10, 16, total_classes)
        
        splits = split_mod_logits(logits)
        
        assert len(splits) == len(config.MOD_RANGE)
    
    def test_split_shapes(self):
        """Test each split has correct shape."""
        from intseq_bert.analysis.common import split_mod_logits
        
        N, L = 10, 16
        total_classes = sum(config.MOD_RANGE)
        logits = torch.randn(N, L, total_classes)
        
        splits = split_mod_logits(logits)
        
        for i, m in enumerate(config.MOD_RANGE):
            assert splits[i].shape == (N, L, m)
    
    def test_split_2d_input(self):
        """Test with 2D input (L, sum(MOD_RANGE))."""
        from intseq_bert.analysis.common import split_mod_logits
        
        L = 16
        total_classes = sum(config.MOD_RANGE)
        logits = torch.randn(L, total_classes)
        
        splits = split_mod_logits(logits)
        
        assert len(splits) == len(config.MOD_RANGE)
        assert splits[0].shape == (L, 2)  # mod 2
        assert splits[-1].shape == (L, 101)  # mod 101


# ==========================================
# load_oeis_tags Tests
# ==========================================

@requires_analyze_mod_spectrum
class TestLoadOeisTags:
    """Tests for load_oeis_tags function."""
    
    def test_loads_tags_correctly(self, temp_jsonl_file, sample_oeis_ids):
        """Test loading tags from JSONL file."""
        from intseq_bert.analysis.analyze_mod_spectrum import load_oeis_tags
        
        id_to_tags = load_oeis_tags(str(temp_jsonl_file))
        
        assert isinstance(id_to_tags, dict)
        assert len(id_to_tags) == len(sample_oeis_ids)
        
        for oeis_id in sample_oeis_ids:
            assert oeis_id in id_to_tags
            assert isinstance(id_to_tags[oeis_id], list)
    
    def test_handles_missing_keywords(self, tmp_path):
        """Test handling records without keywords field."""
        from intseq_bert.analysis.analyze_mod_spectrum import load_oeis_tags
        
        jsonl_path = tmp_path / "no_keywords.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"oeis_id": "A000001", "values": [1, 2, 3]}) + "\n")
        
        id_to_tags = load_oeis_tags(str(jsonl_path))
        
        assert id_to_tags["A000001"] == []


# ==========================================
# tag_stratified_analysis Tests
# ==========================================

@requires_analyze_mod_spectrum
class TestTagStratifiedAnalysis:
    """Tests for tag_stratified_analysis_from_stats function."""
    
    def test_output_structure(self):
        """Test output DataFrame has expected structure."""
        from intseq_bert.analysis.analyze_mod_spectrum import tag_stratified_analysis_from_stats
        
        N = 50
        num_mods = len(config.MOD_RANGE)
        
        # Mock stats
        stats = {
            "loss_sum": torch.rand(N, num_mods),
            "acc_sum": torch.rand(N, num_mods) * 100,
            "counts": torch.ones(N) * 10,
            "oeis_ids": [f"A{i:06d}" for i in range(N)]
        }
        
        id_to_tags = {oid: ["core", "nice"] for oid in stats["oeis_ids"]}
        
        df = tag_stratified_analysis_from_stats(stats, id_to_tags)
        
        expected_cols = {"tag", "count", "overall_acc", "non_trivial_acc", "nig_score", "top_modulus"}
        assert set(df.columns) == expected_cols
    
    def test_filters_small_tags(self):
        """Test that tags with < MIN_TAG_SAMPLES are filtered out."""
        from intseq_bert.analysis.analyze_mod_spectrum import tag_stratified_analysis_from_stats
        
        N = config.MIN_TAG_SAMPLES - 1  # Less than minimum
        num_mods = len(config.MOD_RANGE)
        
        stats = {
            "loss_sum": torch.rand(N, num_mods),
            "acc_sum": torch.rand(N, num_mods) * 100,
            "counts": torch.ones(N) * 10,
            "oeis_ids": [f"A{i:06d}" for i in range(N)]
        }
        
        # All assigned to "rare_tag"
        id_to_tags = {oid: ["rare_tag"] for oid in stats["oeis_ids"]}
        
        df = tag_stratified_analysis_from_stats(stats, id_to_tags)
        
        # Should be empty since count < threshold
        assert len(df) == 0
    
    def test_sorted_by_nig(self):
        """Test results are sorted by nig_score descending."""
        from intseq_bert.analysis.analyze_mod_spectrum import tag_stratified_analysis_from_stats
        
        N = 50
        num_mods = len(config.MOD_RANGE)
        
        loss_sum = torch.zeros(N, num_mods)
        # Make tag_a have better (lower) loss than tag_b
        loss_sum[:25, :] = 0.5  # tag_a
        loss_sum[25:, :] = 2.0  # tag_b
        
        stats = {
            "loss_sum": loss_sum,
            "acc_sum": torch.rand(N, num_mods) * 100,
            "counts": torch.ones(N) * 10,
            "oeis_ids": [f"A{i:06d}" for i in range(N)]
        }
        
        id_to_tags = {}
        for i, oid in enumerate(stats["oeis_ids"]):
            if i < 25:
                id_to_tags[oid] = ["tag_a"]
            else:
                id_to_tags[oid] = ["tag_b"]
        
        df = tag_stratified_analysis_from_stats(stats, id_to_tags)
        
        if len(df) > 1:
            # Check sorted descending
            nig_scores = df["nig_score"].tolist()
            assert nig_scores == sorted(nig_scores, reverse=True)


# ==========================================
# INTERPRETATION_MAP Tests
# ==========================================

class TestInterpretationMap:
    """Tests for INTERPRETATION_MAP configuration."""
    
    @requires_analyze_mod_spectrum
    def test_interpretation_map_exists(self):
        """Test INTERPRETATION_MAP is defined."""
        from intseq_bert.analysis.analyze_mod_spectrum import INTERPRETATION_MAP
        
        assert isinstance(INTERPRETATION_MAP, dict)
        assert len(INTERPRETATION_MAP) > 0
    
    @requires_analyze_mod_spectrum
    def test_key_mods_have_interpretations(self):
        """Test key moduli have interpretations."""
        from intseq_bert.analysis.analyze_mod_spectrum import INTERPRETATION_MAP
        
        key_mods = [2, 3, 10, 100]
        for m in key_mods:
            assert m in INTERPRETATION_MAP, f"Mod {m} should have interpretation"
    
    @requires_analyze_mod_spectrum
    def test_get_interpretation_function(self):
        """Test get_interpretation function."""
        from intseq_bert.analysis.analyze_mod_spectrum import get_interpretation
        
        # Known interpretations
        assert "Parity" in get_interpretation(2)
        assert "Base-10" in get_interpretation(10)
        
        # Prime number in map
        assert "Prime" in get_interpretation(101)


# ==========================================
# StreamingEvaluator Tests
# ==========================================

@requires_analyze_mod_spectrum
class TestStreamingEvaluator:
    """Tests for StreamingEvaluator class."""
    
    def test_accumulation(
        self, sample_mod_logits, sample_mod_targets, sample_mask_map, sample_oeis_ids
    ):
        """Test accumulation of stats."""
        from intseq_bert.analysis.analyze_mod_spectrum import StreamingEvaluator
        
        # Prepare batch and preds
        batch = {
            "mod_labels": sample_mod_targets,
            "mask_matrix": sample_mask_map,
            "oeis_ids": sample_oeis_ids
        }
        preds = {"mod_logits": sample_mod_logits}
        
        evaluator = StreamingEvaluator()
        evaluator.process_batch(preds, batch)
        stats = evaluator.finalize()
        
        # Check shapes
        N = len(sample_oeis_ids)
        num_mods = len(config.MOD_RANGE)
        
        assert stats["loss_sum"].shape == (N, num_mods)
        assert stats["acc_sum"].shape == (N, num_mods)
        assert stats["counts"].shape == (N,)
        assert len(stats["oeis_ids"]) == N
    
    def test_correctness_simple_case(self):
        """Verify that StreamingEvaluator produces correct metric values."""
        from intseq_bert.analysis.analyze_mod_spectrum import compute_mod_metrics_from_stats
        
        # Manually create stats for a known simple case
        # 2 samples. 100 mods.
        # Sample 1: 10 counts. Mod 2 loss=5.0, acc=8.0
        # Sample 2: 10 counts. Mod 2 loss=15.0, acc=2.0
        
        N = 2
        n_mods = len(config.MOD_RANGE)
        
        loss_sum = torch.zeros(N, n_mods)
        acc_sum = torch.zeros(N, n_mods)
        counts = torch.tensor([10.0, 10.0])
        
        # Set values for Mod 2 (index 0)
        loss_sum[0, 0] = 5.0
        loss_sum[1, 0] = 15.0 # Total loss = 20.0, Mean loss = 20/20 = 1.0
        
        acc_sum[0, 0] = 8.0
        acc_sum[1, 0] = 2.0  # Total acc = 10.0, Mean acc = 10/20 = 0.5 (50%)
        
        stats = {
            "loss_sum": loss_sum,
            "acc_sum": acc_sum,
            "counts": counts,
            "oeis_ids": ["A1", "A2"]
        }
        
        df = compute_mod_metrics_from_stats(stats)
        
        # Find row for mod 2
        mod2_row = df[df["modulus"] == 2].iloc[0]
        
        assert abs(mod2_row["ce_loss"] - 1.0) < 1e-6
        assert abs(mod2_row["accuracy"] - 50.0) < 1e-6
        
        expected_nig = 1.0 - (1.0 / np.log(2))
        assert abs(mod2_row["nig_score"] - expected_nig) < 1e-6
    
    def test_empty_stats(self):
        """Test handling of empty stats."""
        from intseq_bert.analysis.analyze_mod_spectrum import compute_mod_metrics_from_stats
        
        stats = {
            "loss_sum": torch.empty(0, 100),
            "acc_sum": torch.empty(0, 100),
            "counts": torch.empty(0),
            "oeis_ids": []
        }
        df = compute_mod_metrics_from_stats(stats)
        assert len(df) == 0 or len(df.columns) == 4
