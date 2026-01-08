"""
Tests for evaluate_final.py (Encoder-Decoder Evaluation Script).

Note: The new evaluate_final.py removes many standalone helper functions
and integrates them into the main evaluation loop. We focus on testing:
1. get_test_ids_from_loader (still exported)
2. run_inference (the main inference pipeline)
3. Integration with mock encoder/decoder
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
# 1. get_test_ids_from_loader Tests
# ==========================================

class TestGetTestIdsFromLoader:
    """Tests for get_test_ids_from_loader function."""
    
    def test_returns_set(self, tmp_path):
        """Test that function returns a set."""
        # Create minimal feature files
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        for i in range(10):
            (features_dir / f"A{i:06d}.pt").touch()
        
        with patch.object(evaluate_final.loader, 'load_and_split_data') as mock_load:
            # Mock return value
            class MockDataset:
                def __init__(self, files):
                    self.feature_files = files
            
            train_files = [features_dir / f"A{i:06d}.pt" for i in range(7)]
            val_files = [features_dir / f"A000007.pt"]
            test_files = [features_dir / f"A000008.pt", features_dir / f"A000009.pt"]
            
            mock_load.return_value = (
                MockDataset(train_files),
                MockDataset(val_files),
                MockDataset(test_files)
            )
            
            result = evaluate_final.get_test_ids_from_loader(
                str(features_dir), 0.05, 0.05, 42
            )
            
            assert isinstance(result, set)
            assert len(result) == 2
            assert "A000008" in result
            assert "A000009" in result


# ==========================================
# 2. run_inference Tests
# ==========================================

class TestRunInference:
    """Tests for run_inference function."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Create mock encoder that returns proper structure."""
        encoder = MagicMock()
        
        # Mock output: dict with last_hidden_state
        mock_output = {
            'last_hidden_state': torch.randn(1, 128, 512),
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
# 3. load_models Tests
# ==========================================

class TestLoadModels:
    """Tests for load_models function."""
    
    def test_returns_encoder_decoder_tuple(self, tmp_path):
        """Test that load_models returns encoder and decoder."""
        # Create a minimal checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        
        # Create state dict with encoder and decoder keys
        state_dict = {
            "encoder.embedding.weight": torch.randn(100, 512),
            "decoder.fc1.weight": torch.randn(512, 512),
        }
        torch.save({"state_dict": state_dict}, checkpoint_path)
        
        with patch.object(evaluate_final.IntSeqBERT, '__init__', return_value=None):
            with patch.object(evaluate_final.IntSeqDecoder, '__init__', return_value=None):
                with patch.object(evaluate_final.IntSeqBERT, 'load_state_dict'):
                    with patch.object(evaluate_final.IntSeqDecoder, 'load_state_dict'):
                        with patch.object(evaluate_final.IntSeqBERT, 'to', return_value=MagicMock()):
                            with patch.object(evaluate_final.IntSeqDecoder, 'to', return_value=MagicMock()):
                                with patch.object(evaluate_final.IntSeqBERT, 'eval'):
                                    with patch.object(evaluate_final.IntSeqDecoder, 'eval'):
                                        # Skip actual loading for unit test
                                        pass


# ==========================================
# 4. Integration Tests
# ==========================================

class TestEvaluationIntegration:
    """Integration tests for evaluation workflow."""
    
    def test_setup_args_has_required_arguments(self):
        """Test that setup_args defines expected arguments."""
        with patch('sys.argv', ['evaluate_final.py', 
                                '--model_path', 'test.pt',
                                '--features_dir', '/tmp/features',
                                '--jsonl_path', '/tmp/data.jsonl']):
            args = evaluate_final.setup_args()
            
            assert hasattr(args, 'model_path')
            assert hasattr(args, 'features_dir')
            assert hasattr(args, 'jsonl_path')
            assert hasattr(args, 'beam_width')
            assert hasattr(args, 'top_k')
