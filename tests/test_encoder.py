import pytest
import torch
import json
import argparse
from unittest.mock import patch, MagicMock
from pathlib import Path

from intseq_bert import encoder

@pytest.fixture
def mock_jsonl_file(tmp_path):
    """Fixture to create a valid JSONL file"""
    data = [
        {"oeis_id": "A000001", "sequence": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {"oeis_id": "A000002", "sequence": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
    ]
    file_path = tmp_path / "test_data.jsonl"
    with open(file_path, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")
    return file_path

def test_encoder_happy_path(mock_jsonl_file, tmp_path):
    """
    Normal case: Verify JSONL to .pt conversion is successful and format is correct
    """
    output_path = tmp_path / "features.pt"
    
    # Create mock arguments
    args = argparse.Namespace(
        input=str(mock_jsonl_file),
        output=str(output_path),
        min_len=5 # Test data length is 10, so it should pass
    )

    # Execution
    # Note: Since the logic in features.py can take time, whether to actually compute it 
    # as an integration test or mock it is a matter of judgment, but here we call 
    # the actual extract_features to test "encoder.py I/O".
    # (Assuming features.extract_features is already implemented)
    encoder.process_encode(args)

    # Verification 1: File generation
    assert output_path.exists()

    # Verification 2: Load and verify structure
    data = torch.load(output_path)
    assert isinstance(data, dict)
    assert "A000001" in data
    assert "A000002" in data
    
    # Verification 3: Data type and shape
    tensor_a = data["A000001"]
    assert isinstance(tensor_a, torch.Tensor)
    assert tensor_a.dtype == torch.float32
    # Features are 27-dimensional, length is original sequence length (10) - reduction due to window processing etc.
    # (Depends on features.py specification, but at least dim=1 should be 27)
    assert tensor_a.shape[1] == 27 

def test_encoder_error_handling(tmp_path):
    """
    Error case: Verify if it skips the record and continues when a calculation error occurs
    """
    # 1. Data preparation
    data = [
        {"oeis_id": "A_GOOD", "sequence": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {"oeis_id": "A_BAD", "sequence": [0, 0, 0]} # Assume this will result in an error
    ]
    input_path = tmp_path / "mixed.jsonl"
    output_path = tmp_path / "mixed_features.pt"
    
    with open(input_path, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")

    args = argparse.Namespace(
        input=str(input_path),
        output=str(output_path),
        min_len=5 
    )

    # 2. Mock extract_features to raise an error for a specific ID
    # Patch intseq_bert.features.extract_features
    with patch("intseq_bert.features.extract_features") as mock_extract:
        def side_effect(seq):
            if len(seq) < 5: # A_BAD case
                raise ValueError("Calculation Error Sim")
            return torch.randn(len(seq), 27).numpy() # A_GOOD succeeds (dummy array)

        mock_extract.side_effect = side_effect
        
        # Execution
        encoder.process_encode(args)

    # 3. Verification
    assert output_path.exists()
    data = torch.load(output_path)
    
    # A_GOOD should exist
    assert "A_GOOD" in data
    # A_BAD should be skipped (proof that it didn't crash)
    assert "A_BAD" not in data

def test_cli_integration(mock_jsonl_file, tmp_path, monkeypatch):
    """Test if it can be called as a CLI command"""
    output_path = tmp_path / "cli_out.pt"
    
    # Rewrite sys.argv and call main()
    test_args = [
        "encoder.py",
        "--input", str(mock_jsonl_file),
        "--output", str(output_path)
    ]
    monkeypatch.setattr("sys.argv", test_args)
    
    encoder.main()
    
    assert output_path.exists()