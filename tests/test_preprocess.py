"""
Tests for preprocess.py module.

Covers:
1. Layer 1: Pure parsing functions
2. Layer 2: Worker and helper functions
3. Layer 3: Command handlers
4. CLI integration
"""

import pytest
import gzip
import argparse
import torch
from pathlib import Path

from intseq_bert import preprocess, schemas, config


# ==========================================
# Helper Functions
# ==========================================

def create_gzipped_file(path: Path, content: str):
    """Write string to a gzip file."""
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        f.write(content)


def create_jsonl_file(path: Path, records: list):
    """Create a JSONL file from OEISRecord list."""
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(rec.to_json_line() + '\n')


# ==========================================
# Layer 1: Pure Parsing Functions
# ==========================================

class TestParseStrippedLine:
    """Tests for _parse_stripped_line."""
    
    def test_valid_line(self):
        """Test parsing a valid stripped line."""
        line = "A000045 ,0,1,1,2,3,5,8"
        result = preprocess._parse_stripped_line(line)
        
        assert result is not None
        oeis_id, sequence = result
        assert oeis_id == "A000045"
        assert sequence == [0, 1, 1, 2, 3, 5, 8]
    
    def test_empty_line(self):
        """Test that empty line returns None."""
        assert preprocess._parse_stripped_line("") is None
    
    def test_comment_line(self):
        """Test that comment-like lines are handled."""
        # Note: comment lines don't have " ," so they fail validation
        result = preprocess._parse_stripped_line("# This is a comment")
        assert result is None
    
    def test_invalid_format(self):
        """Test that malformed lines return None."""
        assert preprocess._parse_stripped_line("A000001,1,2,3") is None  # Missing space
        assert preprocess._parse_stripped_line("B000001 ,1,2,3") is None  # Wrong prefix
    
    def test_non_integer_values(self):
        """Test that non-integer values return None."""
        result = preprocess._parse_stripped_line("A000001 ,1,abc,3")
        assert result is None
    
    def test_negative_numbers(self):
        """Test that negative numbers are parsed correctly."""
        line = "A000001 ,1,-2,3,-4"
        result = preprocess._parse_stripped_line(line)
        
        assert result is not None
        _, sequence = result
        assert sequence == [1, -2, 3, -4]


class TestParseNamesLine:
    """Tests for _parse_names_line."""
    
    def test_valid_line(self):
        """Test parsing a valid names line."""
        line = "A000045 Fibonacci numbers"
        result = preprocess._parse_names_line(line)
        
        assert result is not None
        oeis_id, name = result
        assert oeis_id == "A000045"
        assert name == "Fibonacci numbers"
    
    def test_empty_line(self):
        """Test that empty line returns None."""
        assert preprocess._parse_names_line("") is None
    
    def test_comment_line(self):
        """Test that comment lines return None."""
        assert preprocess._parse_names_line("# Comment line") is None
    
    def test_invalid_prefix(self):
        """Test that non-A lines return None."""
        assert preprocess._parse_names_line("B000001 Some name") is None
    
    def test_name_with_spaces(self):
        """Test that multi-word names are preserved."""
        line = "A000001 This is a long sequence name with many words"
        result = preprocess._parse_names_line(line)
        
        assert result is not None
        _, name = result
        assert name == "This is a long sequence name with many words"


class TestParseSeqContent:
    """Tests for _parse_seq_content."""
    
    def test_keywords_extraction(self):
        """Test extracting keywords from %K line."""
        lines = [
            "%I A000045",
            "%K A000045 nonn,core,easy",
            "%O A000045 0,2"
        ]
        result = preprocess._parse_seq_content(lines)
        
        assert result["keywords"] == ["nonn", "core", "easy"]
    
    def test_offset_extraction(self):
        """Test extracting offset from %O line."""
        lines = ["%O A000045 1,5"]
        result = preprocess._parse_seq_content(lines)
        
        assert result["offset_a"] == 1
    
    def test_empty_lines(self):
        """Test with empty input."""
        result = preprocess._parse_seq_content([])
        
        assert result["keywords"] == []
        assert result["offset_a"] == 0
    
    def test_partial_data(self):
        """Test with only some fields present."""
        lines = ["%K A000001 nonn"]
        result = preprocess._parse_seq_content(lines)
        
        assert result["keywords"] == ["nonn"]
        assert result["offset_a"] == 0  # Default


# ==========================================
# Layer 2: Worker & Helper Functions
# ==========================================

class TestLoadNamesMap:
    """Tests for _load_names_map."""
    
    def test_loads_names(self, tmp_path):
        """Test loading names from gzipped file."""
        names_data = """# OEIS Names
A000001 Name One
A000002 Name Two
A000003 Name Three
"""
        names_gz = tmp_path / "names.gz"
        create_gzipped_file(names_gz, names_data)
        
        result = preprocess._load_names_map(names_gz)
        
        assert len(result) == 3
        assert result["A000001"] == "Name One"
        assert result["A000002"] == "Name Two"
        assert result["A000003"] == "Name Three"


class TestScanSeqFiles:
    """Tests for _scan_seq_files."""
    
    def test_scans_files(self, tmp_path):
        """Test scanning directory for .seq files."""
        # Create directory structure
        seq_dir = tmp_path / "seq"
        a000_dir = seq_dir / "A000"
        a000_dir.mkdir(parents=True)
        
        (a000_dir / "A000001.seq").write_text("%I A000001")
        (a000_dir / "A000002.seq").write_text("%I A000002")
        (a000_dir / "readme.txt").write_text("Not a seq file")
        
        result = preprocess._scan_seq_files(seq_dir)
        
        assert len(result) == 2
        assert "A000001" in result
        assert "A000002" in result
    
    def test_ignores_non_a_files(self, tmp_path):
        """Test that non-A prefixed files are ignored."""
        seq_dir = tmp_path / "seq"
        seq_dir.mkdir()
        
        (seq_dir / "A000001.seq").write_text("")
        (seq_dir / "b000001.seq").write_text("")  # b-file, not A-file
        
        result = preprocess._scan_seq_files(seq_dir)
        
        assert len(result) == 1
        assert "A000001" in result


class TestWorkerExtractFeatures:
    """Tests for _worker_extract_features."""
    
    def test_extracts_features(self, tmp_path):
        """Test feature extraction from JSONL lines."""
        output_dir = tmp_path / "features"
        output_dir.mkdir()
        
        # Create valid JSONL lines (must have MIN_SEQUENCE_LENGTH elements)
        record = schemas.OEISRecord(
            oeis_id="A000001",
            sequence=list(range(1, config.MIN_SEQUENCE_LENGTH + 5))
        )
        lines = [record.to_json_line()]
        
        count = preprocess._worker_extract_features(lines, output_dir)
        
        assert count == 1
        assert (output_dir / "A000001.pt").exists()
    
    def test_skips_short_sequences(self, tmp_path):
        """Test that sequences shorter than MIN_SEQUENCE_LENGTH are skipped."""
        output_dir = tmp_path / "features"
        output_dir.mkdir()
        
        record = schemas.OEISRecord(
            oeis_id="A000001",
            sequence=[1, 2, 3]  # Too short
        )
        lines = [record.to_json_line()]
        
        count = preprocess._worker_extract_features(lines, output_dir)
        
        assert count == 0
        assert not (output_dir / "A000001.pt").exists()
    
    def test_saved_file_structure(self, tmp_path):
        """Test that saved .pt files have correct structure."""
        output_dir = tmp_path / "features"
        output_dir.mkdir()
        
        record = schemas.OEISRecord(
            oeis_id="A000042",
            sequence=list(range(1, 20))  # 19 elements
        )
        lines = [record.to_json_line()]
        preprocess._worker_extract_features(lines, output_dir)
        
        data = torch.load(output_dir / "A000042.pt")
        
        # Check keys (using config constants)
        assert config.KEY_OEIS_ID in data
        assert config.KEY_MAG_FEATURES in data
        assert config.KEY_MOD_FEATURES in data
        assert config.KEY_MOD_INTEGERS in data
        assert "numbers" in data  # Raw integer sequence for Vanilla Transformer
        
        # Check shapes
        L = 19
        assert data[config.KEY_OEIS_ID] == "A000042"
        assert data[config.KEY_MAG_FEATURES].shape == (L, config.MAG_RAW_DIM)
        assert data[config.KEY_MOD_FEATURES].shape == (L, config.MOD_FEATURE_DIM)
        assert data[config.KEY_MOD_INTEGERS].shape == (L, config.NUM_MODULI)
        
        # Check raw numbers
        assert data["numbers"] == list(range(1, 20))
    
    def test_numbers_truncated_to_max_length(self, tmp_path):
        """Test that numbers are truncated to MAX_SEQUENCE_LENGTH."""
        output_dir = tmp_path / "features"
        output_dir.mkdir()
        
        # Create sequence longer than MAX_SEQUENCE_LENGTH
        long_sequence = list(range(1, config.MAX_SEQUENCE_LENGTH + 100))
        record = schemas.OEISRecord(
            oeis_id="A000099",
            sequence=long_sequence
        )
        lines = [record.to_json_line()]
        preprocess._worker_extract_features(lines, output_dir)
        
        data = torch.load(output_dir / "A000099.pt")
        
        # Check that numbers are truncated to MAX_SEQUENCE_LENGTH
        assert len(data["numbers"]) == config.MAX_SEQUENCE_LENGTH
        assert data["numbers"] == long_sequence[:config.MAX_SEQUENCE_LENGTH]
        
        # Check consistency: numbers length should match feature tensor length
        assert len(data["numbers"]) == data[config.KEY_MAG_FEATURES].shape[0]


# ==========================================
# Layer 3: Command Handlers
# ==========================================

class TestCmdBuildJsonl:
    """Tests for cmd_build_jsonl."""
    
    def test_basic_build(self, tmp_path):
        """Test building JSONL from stripped.gz only."""
        stripped_data = """A000001 ,1,2,3,4,5
A000002 ,10,20,30
"""
        input_gz = tmp_path / "stripped.gz"
        output_jsonl = tmp_path / "data.jsonl"
        create_gzipped_file(input_gz, stripped_data)
        
        args = argparse.Namespace(
            stripped=str(input_gz),
            names=None,
            seq_dir=None,
            output=str(output_jsonl)
        )
        
        preprocess.cmd_build_jsonl(args)
        
        assert output_jsonl.exists()
        records = list(schemas.load_records(str(output_jsonl)))
        assert len(records) == 2
        
        rec1 = next(r for r in records if r.oeis_id == "A000001")
        assert rec1.sequence == [1, 2, 3, 4, 5]
    
    def test_with_names(self, tmp_path):
        """Test building JSONL with names.gz merge."""
        stripped_data = "A000001 ,1,2,3,4,5\n"
        names_data = "A000001 Fibonacci numbers\n"
        
        stripped_gz = tmp_path / "stripped.gz"
        names_gz = tmp_path / "names.gz"
        output_jsonl = tmp_path / "data.jsonl"
        
        create_gzipped_file(stripped_gz, stripped_data)
        create_gzipped_file(names_gz, names_data)
        
        args = argparse.Namespace(
            stripped=str(stripped_gz),
            names=str(names_gz),
            seq_dir=None,
            output=str(output_jsonl)
        )
        
        preprocess.cmd_build_jsonl(args)
        
        records = list(schemas.load_records(str(output_jsonl)))
        assert records[0].name == "Fibonacci numbers"


class TestCmdExtractFeatures:
    """Tests for cmd_extract_features."""
    
    def test_extraction(self, tmp_path):
        """Test feature extraction from JSONL."""
        # Create input
        input_jsonl = tmp_path / "data.jsonl"
        records = [
            schemas.OEISRecord(oeis_id=f"A{i:06d}", sequence=list(range(1, 20)))
            for i in range(5)
        ]
        create_jsonl_file(input_jsonl, records)
        
        output_dir = tmp_path / "features"
        
        args = argparse.Namespace(
            input=str(input_jsonl),
            output_dir=str(output_dir),
            workers=1,
            chunk_size=2
        )
        
        preprocess.cmd_extract_features(args)
        
        pt_files = list(output_dir.glob("*.pt"))
        assert len(pt_files) == 5


class TestCmdSplitDataset:
    """Tests for cmd_split_dataset."""
    
    def test_split_creation(self, tmp_path):
        """Test train/val/test split creation."""
        # Create JSONL with records
        jsonl_path = tmp_path / "data.jsonl"
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        records = []
        for i in range(100):
            rec = schemas.OEISRecord(oeis_id=f"A{i:06d}", sequence=[1, 2, 3])
            records.append(rec)
            (features_dir / f"A{i:06d}.pt").write_bytes(b"dummy")
        
        create_jsonl_file(jsonl_path, records)
        
        output_dir = tmp_path / "splits"
        
        args = argparse.Namespace(
            jsonl=str(jsonl_path),
            features_dir=str(features_dir),
            output_dir=str(output_dir),
            include_tags=None,
            exclude_tags=None
        )
        
        preprocess.cmd_split_dataset(args)
        
        assert (output_dir / "train.txt").exists()
        assert (output_dir / "val.txt").exists()
        assert (output_dir / "test.txt").exists()
        
        # Check total IDs match input
        all_ids = set()
        for split_file in ["train.txt", "val.txt", "test.txt"]:
            with open(output_dir / split_file) as f:
                ids = [line.strip() for line in f if line.strip()]
                all_ids.update(ids)
        
        assert len(all_ids) == 100
    
    def test_include_tags_filter(self, tmp_path):
        """Test filtering by include tags."""
        jsonl_path = tmp_path / "data.jsonl"
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        # Create records with different tags
        records = [
            schemas.OEISRecord(oeis_id="A000001", sequence=[1], keywords=["core", "easy"]),
            schemas.OEISRecord(oeis_id="A000002", sequence=[1], keywords=["nonn"]),
            schemas.OEISRecord(oeis_id="A000003", sequence=[1], keywords=["core"]),
        ]
        for rec in records:
            (features_dir / f"{rec.oeis_id}.pt").write_bytes(b"dummy")
        
        create_jsonl_file(jsonl_path, records)
        
        output_dir = tmp_path / "splits"
        
        args = argparse.Namespace(
            jsonl=str(jsonl_path),
            features_dir=str(features_dir),
            output_dir=str(output_dir),
            include_tags="core",
            exclude_tags=None
        )
        
        preprocess.cmd_split_dataset(args)
        
        # Only A000001 and A000003 should be included (have 'core')
        all_ids = set()
        for split_file in ["train.txt", "val.txt", "test.txt"]:
            with open(output_dir / split_file) as f:
                ids = [line.strip() for line in f if line.strip()]
                all_ids.update(ids)
        
        assert len(all_ids) == 2
        assert "A000001" in all_ids
        assert "A000003" in all_ids
        assert "A000002" not in all_ids
    
    def test_exclude_tags_filter(self, tmp_path):
        """Test filtering by exclude tags."""
        jsonl_path = tmp_path / "data.jsonl"
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        records = [
            schemas.OEISRecord(oeis_id="A000001", sequence=[1], keywords=["nonn"]),
            schemas.OEISRecord(oeis_id="A000002", sequence=[1], keywords=["base"]),
            schemas.OEISRecord(oeis_id="A000003", sequence=[1], keywords=["cons"]),
        ]
        for rec in records:
            (features_dir / f"{rec.oeis_id}.pt").write_bytes(b"dummy")
        
        create_jsonl_file(jsonl_path, records)
        
        output_dir = tmp_path / "splits"
        
        args = argparse.Namespace(
            jsonl=str(jsonl_path),
            features_dir=str(features_dir),
            output_dir=str(output_dir),
            include_tags=None,
            exclude_tags="base,cons"
        )
        
        preprocess.cmd_split_dataset(args)
        
        # Only A000001 should remain
        all_ids = set()
        for split_file in ["train.txt", "val.txt", "test.txt"]:
            with open(output_dir / split_file) as f:
                ids = [line.strip() for line in f if line.strip()]
                all_ids.update(ids)
        
        assert len(all_ids) == 1
        assert "A000001" in all_ids
    
    def test_deterministic_split(self, tmp_path):
        """Test that splits are deterministic with same seed."""
        jsonl_path = tmp_path / "data.jsonl"
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        records = []
        for i in range(50):
            rec = schemas.OEISRecord(oeis_id=f"A{i:06d}", sequence=[1, 2, 3])
            records.append(rec)
            (features_dir / f"A{i:06d}.pt").write_bytes(b"dummy")
        
        create_jsonl_file(jsonl_path, records)
        
        # First split
        output1 = tmp_path / "split1"
        args1 = argparse.Namespace(
            jsonl=str(jsonl_path),
            features_dir=str(features_dir),
            output_dir=str(output1),
            include_tags=None,
            exclude_tags=None
        )
        preprocess.cmd_split_dataset(args1)
        
        # Second split
        output2 = tmp_path / "split2"
        args2 = argparse.Namespace(
            jsonl=str(jsonl_path),
            features_dir=str(features_dir),
            output_dir=str(output2),
            include_tags=None,
            exclude_tags=None
        )
        preprocess.cmd_split_dataset(args2)
        
        # Compare
        with open(output1 / "train.txt") as f1, open(output2 / "train.txt") as f2:
            assert f1.read() == f2.read()


# ==========================================
# CLI Integration Tests
# ==========================================

class TestCLI:
    """Tests for CLI argument parsing."""
    
    def test_build_jsonl_cli(self, tmp_path, monkeypatch):
        """Test build-jsonl command via CLI."""
        stripped_gz = tmp_path / "stripped.gz"
        output = tmp_path / "out.jsonl"
        
        create_gzipped_file(stripped_gz, "A000001 ,1,2,3,4,5\n")
        
        monkeypatch.setattr("sys.argv", [
            "preprocess.py", "build-jsonl",
            "--stripped", str(stripped_gz),
            "-o", str(output)
        ])
        
        preprocess.main()
        
        assert output.exists()
    
    def test_extract_features_cli(self, tmp_path, monkeypatch):
        """Test extract-features command via CLI."""
        input_jsonl = tmp_path / "data.jsonl"
        records = [schemas.OEISRecord(oeis_id="A000001", sequence=list(range(1, 20)))]
        create_jsonl_file(input_jsonl, records)
        
        output_dir = tmp_path / "features"
        
        monkeypatch.setattr("sys.argv", [
            "preprocess.py", "extract-features",
            "-i", str(input_jsonl),
            "-o", str(output_dir),
            "--workers", "1"
        ])
        
        preprocess.main()
        
        assert (output_dir / "A000001.pt").exists()
    
    def test_split_dataset_cli(self, tmp_path, monkeypatch):
        """Test split-dataset command via CLI."""
        jsonl_path = tmp_path / "data.jsonl"
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        records = []
        for i in range(20):
            rec = schemas.OEISRecord(oeis_id=f"A{i:06d}", sequence=[1, 2, 3])
            records.append(rec)
            (features_dir / f"A{i:06d}.pt").write_bytes(b"dummy")
        
        create_jsonl_file(jsonl_path, records)
        
        output_dir = tmp_path / "splits"
        
        monkeypatch.setattr("sys.argv", [
            "preprocess.py", "split-dataset",
            "-j", str(jsonl_path),
            "-f", str(features_dir),
            "-o", str(output_dir)
        ])
        
        preprocess.main()
        
        assert (output_dir / "train.txt").exists()

        assert (output_dir / "train.txt").exists()