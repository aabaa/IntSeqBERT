"""
Tests for evaluate_final.py (Encoder-Decoder Evaluation Script).

Tests the key functions:
1. normalize_id
2. run_inference with proper encoder output (encoded_state)
3. load_models
4. setup_args
"""

import pytest
import torch
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from intseq_bert import evaluate_final
from intseq_bert.bert_model import IntSeqBERT
from intseq_bert.decoder_model import IntSeqDecoder


# ==========================================
# 1. normalize_id Tests
# ==========================================

class TestNormalizeId:
    """Tests for normalize_id function."""
    
    def test_already_normalized(self):
        """Test ID that's already in correct format."""
        assert evaluate_final.normalize_id("A000001") == "A000001"
    
    def test_without_prefix(self):
        """Test ID without A prefix."""
        assert evaluate_final.normalize_id("123") == "A000123"
    
    def test_with_prefix_short(self):
        """Test ID with A prefix but short number."""
        assert evaluate_final.normalize_id("A1") == "A000001"
    
    def test_integer_input(self):
        """Test integer input."""
        assert evaluate_final.normalize_id(42) == "A000042"


# ==========================================
# 2. run_inference Tests
# ==========================================

class TestRunInference:
    """Tests for run_inference function."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Create mock encoder that returns proper structure with encoded_state."""
        encoder = MagicMock()
        
        # Mock output: dict with 'encoded_state' (key used by IntSeqBERT)
        mock_output = {
            'encoded_state': torch.randn(1, 128, 512),
            'pred_mag': torch.randn(1, 128, 5)
        }
        encoder.return_value = mock_output
        encoder.eval = MagicMock(return_value=encoder)
        encoder.to = MagicMock(return_value=encoder)
        
        return encoder
    
    @pytest.fixture
    def mock_decoder(self):
        """Create mock decoder that returns predictions."""
        decoder = MagicMock()
        
        # Mock forward output (predictions dict)
        predictions = {
            'mag_mu': torch.tensor([1.5]),
            'mag_logvar': torch.tensor([-3.0]),
            'sign_logits': torch.tensor([[0.1, 0.1, 5.0]])  # Positive
        }
        for m in range(2, 102):
            predictions[f'mod{m}'] = torch.ones(1, m) / m
        
        decoder.return_value = predictions
        decoder.eval = MagicMock(return_value=decoder)
        decoder.to = MagicMock(return_value=decoder)
        
        # Mock beam_search_solve
        decoder.beam_search_solve = MagicMock(return_value=[
            (42, -1.5),
            (43, -2.0),
            (44, -2.5)
        ])
        
        return decoder
    
    def test_returns_expected_structure(self, mock_encoder, mock_decoder):
        """Test that run_inference returns dict with candidates and magnitude."""
        result = evaluate_final.run_inference(
            mock_encoder,
            mock_decoder,
            [1, 2, 3, 4, 5],
            "cpu",
            beam_width=20,
            top_k=5
        )
        
        assert "candidates" in result
        assert "predicted_magnitude" in result
    
    def test_candidates_format(self, mock_encoder, mock_decoder):
        """Test that candidates are list of tuples."""
        result = evaluate_final.run_inference(
            mock_encoder,
            mock_decoder,
            [1, 1, 2, 3, 5],
            "cpu",
            beam_width=20,
            top_k=5
        )
        
        candidates = result["candidates"]
        assert isinstance(candidates, list)
        if len(candidates) > 0:
            assert isinstance(candidates[0], tuple)
    
    def test_predicted_magnitude_is_positive(self, mock_encoder, mock_decoder):
        """Test that predicted magnitude is positive."""
        result = evaluate_final.run_inference(
            mock_encoder,
            mock_decoder,
            [1, 2, 4, 8, 16],
            "cpu",
            beam_width=20,
            top_k=5
        )
        
        assert result["predicted_magnitude"] > 0


# ==========================================
# 3. setup_args Tests
# ==========================================

class TestSetupArgs:
    """Tests for setup_args function."""
    
    def test_has_required_arguments(self):
        """Test that setup_args defines expected arguments."""
        with patch('sys.argv', ['evaluate_final.py', 
                                '--model_path', 'test.pt',
                                '--features_dir', '/tmp/features',
                                '--jsonl_path', '/tmp/data.jsonl']):
            args = evaluate_final.setup_args()
            
            assert hasattr(args, 'model_path')
            assert hasattr(args, 'decoder_path')
            assert hasattr(args, 'features_dir')
            assert hasattr(args, 'jsonl_path')
            assert hasattr(args, 'beam_width')
            assert hasattr(args, 'top_k')
            assert hasattr(args, 'device')
    
    def test_decoder_path_optional(self):
        """Test that decoder_path is optional (defaults to None)."""
        with patch('sys.argv', ['evaluate_final.py', 
                                '--model_path', 'test.pt',
                                '--features_dir', '/tmp/features',
                                '--jsonl_path', '/tmp/data.jsonl']):
            args = evaluate_final.setup_args()
            assert args.decoder_path is None


# ==========================================
# 4. load_test_sequences_direct Tests
# ==========================================

class TestLoadTestSequencesDirect:
    """Tests for load_test_sequences_direct function."""
    
    def test_loads_sequences_with_matching_features(self, tmp_path):
        """Test that only sequences with matching .pt files are loaded."""
        # Create features dir with some .pt files
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        (features_dir / "A000001.pt").touch()
        (features_dir / "A000003.pt").touch()
        
        # Create JSONL with 3 records (only 2 have matching features)
        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, 'w') as f:
            f.write('{"oeis_id": "A000001", "sequence": [1, 2, 3]}\n')
            f.write('{"oeis_id": "A000002", "sequence": [2, 4, 6]}\n')  # No matching .pt
            f.write('{"oeis_id": "A000003", "sequence": [3, 6, 9]}\n')
        
        # Create mock args
        class MockArgs:
            pass
        args = MockArgs()
        args.features_dir = str(features_dir)
        args.jsonl_path = str(jsonl_path)
        args.limit = None
        
        result = evaluate_final.load_test_sequences_direct(args)
        
        assert len(result) == 2
        ids = {r['oeis_id'] for r in result}
        assert ids == {"A000001", "A000003"}
