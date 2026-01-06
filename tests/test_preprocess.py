"""
Tests for the OEIS preprocessing module.
Tests all pipeline steps including the new feature extraction.
"""

import pytest
import gzip
import json
import argparse
import torch
from pathlib import Path

# Modules to be tested
from intseq_bert import preprocess, schemas, converters


def create_gzipped_file(path: Path, content: str):
    """Helper function: Write string to a file with gzip compression."""
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        f.write(content)


# ==========================================
# 1. Test process_stripped
# ==========================================

class TestProcessStripped:
    """Tests for stripped.gz -> JSONL conversion."""
    
    def test_end_to_end(self, tmp_path):
        """Test full conversion flow from stripped.gz to jsonl."""
        stripped_data = """
# OEIS Stripped Data
A000001 ,1,2,3,4,5
A000002 ,1,2
A000003 ,10,20,30,40,50
        """.strip()
        
        input_gz = tmp_path / "stripped.gz"
        output_jsonl = tmp_path / "stripped.jsonl"
        create_gzipped_file(input_gz, stripped_data)
        
        args = argparse.Namespace(
            input=str(input_gz),
            output=str(output_jsonl),
            min_len=3  # Excludes A000002 (len=2)
        )
        
        preprocess.process_stripped(args)
        
        assert output_jsonl.exists()
        records = schemas.load_records(str(output_jsonl))
        assert len(records) == 2
        
        ids = [r.oeis_id for r in records]
        assert "A000001" in ids
        assert "A000003" in ids
        assert "A000002" not in ids
        
        rec1 = next(r for r in records if r.oeis_id == "A000001")
        assert rec1.sequence == [1, 2, 3, 4, 5]


# ==========================================
# 2. Test process_merge_names
# ==========================================

class TestProcessMergeNames:
    """Tests for merging names.gz into JSONL."""
    
    def test_end_to_end(self, tmp_path):
        """Test name merging flow."""
        records = [
            schemas.OEISRecord(oeis_id="A000001", sequence=[1, 2, 3]),
            schemas.OEISRecord(oeis_id="A000002", sequence=[4, 5, 6]),
            schemas.OEISRecord(oeis_id="A999999", sequence=[0])  # No name exists
        ]
        input_jsonl = tmp_path / "input.jsonl"
        schemas.save_records(records, str(input_jsonl))
        
        names_data = """
# OEIS Names
A000001 Name for Sequence One
A000002 Name for Sequence Two
        """.strip()
        names_gz = tmp_path / "names.gz"
        create_gzipped_file(names_gz, names_data)
        
        output_jsonl = tmp_path / "merged.jsonl"
        
        args = argparse.Namespace(
            input_jsonl=str(input_jsonl),
            input_names=str(names_gz),
            output=str(output_jsonl)
        )
        
        preprocess.process_merge_names(args)
        
        assert output_jsonl.exists()
        merged_records = schemas.load_records(str(output_jsonl))
        assert len(merged_records) == 3
        
        rec1 = next(r for r in merged_records if r.oeis_id == "A000001")
        assert rec1.name == "Name for Sequence One"
        
        rec2 = next(r for r in merged_records if r.oeis_id == "A000002")
        assert rec2.name == "Name for Sequence Two"
        
        rec9 = next(r for r in merged_records if r.oeis_id == "A999999")
        assert rec9.name == ""


# ==========================================
# 3. Test process_merge_metadata
# ==========================================

class TestProcessMergeMetadata:
    """Tests for merging .seq metadata into JSONL."""
    
    def test_end_to_end(self, tmp_path):
        """Test metadata merging from .seq files."""
        records = [
            schemas.OEISRecord(oeis_id="A000001", sequence=[1, 2, 3]),
            schemas.OEISRecord(oeis_id="A000002", sequence=[4, 5, 6]),
            schemas.OEISRecord(oeis_id="A000003", sequence=[7, 8, 9])  # No .seq file
        ]
        input_jsonl = tmp_path / "step2.jsonl"
        schemas.save_records(records, str(input_jsonl))
        
        # Create .seq directory structure
        seq_root = tmp_path / "seq"
        a000_dir = seq_root / "A000"
        a000_dir.mkdir(parents=True)
        
        # A000001.seq: Keywords and Offset
        seq1_content = """
%I A000001
%K A000001 nonn, core
%O A000001 1,5
        """.strip()
        (a000_dir / "A000001.seq").write_text(seq1_content, encoding='utf-8')
        
        # A000002.seq: Related (Cross-refs)
        seq2_content = """
%I A000002
%Y A000002 Cf. A000005, A000010.
        """.strip()
        (a000_dir / "A000002.seq").write_text(seq2_content, encoding='utf-8')
        
        output_jsonl = tmp_path / "final.jsonl"
        args = argparse.Namespace(
            input_jsonl=str(input_jsonl),
            seq_dir=str(seq_root),
            output=str(output_jsonl)
        )
        
        preprocess.process_merge_metadata(args)
        
        assert output_jsonl.exists()
        final_records = schemas.load_records(str(output_jsonl))
        assert len(final_records) == 3
        
        rec1 = next(r for r in final_records if r.oeis_id == "A000001")
        assert "nonn" in rec1.keywords
        assert "core" in rec1.keywords
        assert rec1.offset_a == 1
        
        rec2 = next(r for r in final_records if r.oeis_id == "A000002")
        assert "A000005" in rec2.related
        assert "A000010" in rec2.related
        
        rec3 = next(r for r in final_records if r.oeis_id == "A000003")
        assert rec3.keywords == []
        assert rec3.offset_a == 0


# ==========================================
# 4. Test process_features (NEW)
# ==========================================

class TestProcessFeatures:
    """Tests for feature extraction pipeline."""
    
    def test_feature_chunk_processing(self, tmp_path):
        """Test _process_feature_chunk helper function."""
        output_dir = tmp_path / "features"
        output_dir.mkdir()
        
        # Create test chunk
        chunk = [
            {'oeis_id': 'A000001', 'sequence': [1, 2, 3, 4, 5]},
            {'oeis_id': 'A000002', 'sequence': [10, 20, 30, 40, 50]},
            {'oeis_id': 'A000003', 'sequence': [1, 2]},  # Too short (< 5)
            {'oeis_id': None, 'sequence': [1, 2, 3, 4, 5]},  # No ID
        ]
        
        count = preprocess._process_feature_chunk(chunk, output_dir)
        
        # Only 2 valid records
        assert count == 2
        assert (output_dir / "A000001.pt").exists()
        assert (output_dir / "A000002.pt").exists()
        assert not (output_dir / "A000003.pt").exists()
    
    def test_feature_file_structure(self, tmp_path):
        """Test that saved .pt files have correct structure."""
        output_dir = tmp_path / "features"
        output_dir.mkdir()
        
        chunk = [{'oeis_id': 'A000042', 'sequence': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
        preprocess._process_feature_chunk(chunk, output_dir)
        
        data = torch.load(output_dir / "A000042.pt")
        
        # Check structure
        assert 'oeis_id' in data
        assert 'mag_features' in data
        assert 'mod_features' in data
        assert 'targets' in data
        
        # Check shapes
        assert data['oeis_id'] == 'A000042'
        assert data['mag_features'].shape == (10, 5)   # (SeqLen, 5)
        assert data['mod_features'].shape == (10, 200) # (SeqLen, 200)
        
        # Check targets
        assert 'mag' in data['targets']
        assert 'mod3' in data['targets']
        assert 'mod101' in data['targets']
    
        assert data['targets']['mod3'][0] == 1
        assert data['targets']['mod101'][0] == 1
    
    def test_process_features_end_to_end(self, tmp_path):
        """Test full feature extraction pipeline."""
        # Create input JSONL
        input_jsonl = tmp_path / "input.jsonl"
        with open(input_jsonl, 'w') as f:
            for i in range(5):
                record = {
                    'oeis_id': f'A{i:06d}',
                    'sequence': list(range(1, 11))  # 10 elements
                }
                f.write(json.dumps(record) + '\n')
        
        output_dir = tmp_path / "features"
        
        args = argparse.Namespace(
            input=str(input_jsonl),
            output_dir=str(output_dir),
            workers=1,  # Single worker for testing
            chunk_size=2
        )
        
        preprocess.process_features(args)
        
        # Verify output
        assert output_dir.exists()
        pt_files = list(output_dir.glob("*.pt"))
        assert len(pt_files) == 5
    
    def test_process_features_handles_errors(self, tmp_path):
        """Test that feature extraction handles invalid data gracefully."""
        input_jsonl = tmp_path / "input.jsonl"
        with open(input_jsonl, 'w') as f:
            # Valid record
            f.write(json.dumps({'oeis_id': 'A000001', 'sequence': [1, 2, 3, 4, 5]}) + '\n')
            # Invalid JSON (should be skipped)
            f.write("not valid json\n")
            # Empty sequence
            f.write(json.dumps({'oeis_id': 'A000002', 'sequence': []}) + '\n')
            # Too short
            f.write(json.dumps({'oeis_id': 'A000003', 'sequence': [1, 2]}) + '\n')
        
        output_dir = tmp_path / "features"
        
        args = argparse.Namespace(
            input=str(input_jsonl),
            output_dir=str(output_dir),
            workers=1,
            chunk_size=10
        )
        
        # Should not raise
        preprocess.process_features(args)
        
        # Only one valid file
        pt_files = list(output_dir.glob("*.pt"))
        assert len(pt_files) == 1
        assert (output_dir / "A000001.pt").exists()


# ==========================================
# 5. CLI Tests
# ==========================================

class TestCLI:
    """Tests for CLI argument parsing and execution."""
    
    def test_stripped_cli(self, tmp_path, monkeypatch):
        """Test CLI execution for stripped command."""
        input_gz = tmp_path / "data.gz"
        output_jsonl = tmp_path / "out.jsonl"
        
        create_gzipped_file(input_gz, "A001 ,1,2,3")
        
        test_args = [
            "preprocess.py", "stripped",
            "-i", str(input_gz),
            "-o", str(output_jsonl),
            "--min_len", "1"
        ]
        monkeypatch.setattr("sys.argv", test_args)
        
        preprocess.main()
        
        assert output_jsonl.exists()
        records = schemas.load_records(str(output_jsonl))
        assert len(records) == 1
        assert records[0].oeis_id == "A001"
    
    def test_features_cli(self, tmp_path, monkeypatch):
        """Test CLI execution for features command."""
        # Create input
        input_jsonl = tmp_path / "input.jsonl"
        with open(input_jsonl, 'w') as f:
            f.write(json.dumps({'oeis_id': 'A000001', 'sequence': [1, 2, 3, 4, 5]}) + '\n')
        
        output_dir = tmp_path / "features"
        
        test_args = [
            "preprocess.py", "features",
            "-i", str(input_jsonl),
            "-o", str(output_dir),
            "--workers", "1",
            "--chunk-size", "10"
        ]
        monkeypatch.setattr("sys.argv", test_args)
        
        preprocess.main()
        
        assert (output_dir / "A000001.pt").exists()


# ==========================================
# 6. Helper Function Tests
# ==========================================

class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_open_text_plain_file(self, tmp_path):
        """Test opening plain text file."""
        plain_file = tmp_path / "test.txt"
        plain_file.write_text("hello world")
        
        with preprocess._open_text(str(plain_file)) as f:
            content = f.read()
        
        assert content == "hello world"
    
    def test_open_text_gzipped_file(self, tmp_path):
        """Test opening gzipped file."""
        gz_file = tmp_path / "test.gz"
        create_gzipped_file(gz_file, "hello compressed")
        
        with preprocess._open_text(str(gz_file)) as f:
            content = f.read()
        
        assert content == "hello compressed"