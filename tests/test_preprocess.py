import pytest
import gzip
import json
import argparse
from pathlib import Path

# Modules to be tested
from intseq_bert import preprocess, schemas

def create_gzipped_file(path: Path, content: str):
    """Helper function: Write string to a file with gzip compression"""
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        f.write(content)

def test_process_stripped_end_to_end(tmp_path):
    """
    Test for the conversion flow from stripped.gz to jsonl
    """
    # 1. Create mock data (stripped.gz)
    # A001: Normal
    # A002: Too short (should be excluded by min_len)
    # A003: Normal (does not contain large numbers)
    stripped_data = """
# OEIS Stripped Data
A000001 ,1,2,3,4,5
A000002 ,1,2
A000003 ,10,20,30,40,50
    """.strip()
    
    input_gz = tmp_path / "stripped.gz"
    output_jsonl = tmp_path / "stripped.jsonl"
    create_gzipped_file(input_gz, stripped_data)
    
    # 2. Create mock arguments
    args = argparse.Namespace(
        input=str(input_gz),
        output=str(output_jsonl),
        min_len=3  # Set to exclude A000002 (len=2)
    )
    
    # 3. Execution
    preprocess.process_stripped(args)
    
    # 4. Verification
    assert output_jsonl.exists()
    
    records = schemas.load_records(str(output_jsonl))
    assert len(records) == 2
    
    # Verify IDs
    ids = [r.oeis_id for r in records]
    assert "A000001" in ids
    assert "A000003" in ids
    assert "A000002" not in ids # Ensure it is excluded
    
    # Verify content
    rec1 = next(r for r in records if r.oeis_id == "A000001")
    assert rec1.sequence == [1, 2, 3, 4, 5]

def test_process_merge_names_end_to_end(tmp_path):
    """
    Test for the flow of merging names.gz information into JSONL
    """
    # 1. Create input JSONL (without names)
    records = [
        schemas.OEISRecord(oeis_id="A000001", sequence=[1, 2, 3]),
        schemas.OEISRecord(oeis_id="A000002", sequence=[4, 5, 6]),
        schemas.OEISRecord(oeis_id="A999999", sequence=[0]) # Case where no name definition exists
    ]
    input_jsonl = tmp_path / "input.jsonl"
    schemas.save_records(records, str(input_jsonl))
    
    # 2. Create names.gz
    names_data = """
# OEIS Names
A000001 Name for Sequence One
A000002 Name for Sequence Two
    """.strip()
    names_gz = tmp_path / "names.gz"
    create_gzipped_file(names_gz, names_data)
    
    # 3. Output path
    output_jsonl = tmp_path / "merged.jsonl"
    
    # 4. Mock arguments
    args = argparse.Namespace(
        input_jsonl=str(input_jsonl),
        input_names=str(names_gz),
        output=str(output_jsonl)
    )
    
    # 5. Execution
    preprocess.process_merge_names(args)
    
    # 6. Verification
    assert output_jsonl.exists()
    merged_records = schemas.load_records(str(output_jsonl))
    assert len(merged_records) == 3
    
    # Check if names are merged
    rec1 = next(r for r in merged_records if r.oeis_id == "A000001")
    assert rec1.name == "Name for Sequence One"
    
    rec2 = next(r for r in merged_records if r.oeis_id == "A000002")
    assert rec2.name == "Name for Sequence Two"
    
    # Check if name remains an empty string when not found
    rec9 = next(r for r in merged_records if r.oeis_id == "A999999")
    assert rec9.name == ""

def test_cli_execution_stripped(tmp_path, monkeypatch):
    """
    Execution test including argument parsing via main() function
    (Simulates behavior when called from the command line)
    """
    input_gz = tmp_path / "data.gz"
    output_jsonl = tmp_path / "out.jsonl"
    
    create_gzipped_file(input_gz, "A001 ,1,2,3")
    
    # Overwrite sys.argv and execute
    test_args = [
        "preprocess.py", "stripped",
        "-i", str(input_gz),
        "-o", str(output_jsonl),
        "--min_len", "1"
    ]
    monkeypatch.setattr("sys.argv", test_args)
    
    # Verify it finishes without errors
    preprocess.main()
    
    assert output_jsonl.exists()
    records = schemas.load_records(str(output_jsonl))
    assert len(records) == 1
    assert records[0].oeis_id == "A001"