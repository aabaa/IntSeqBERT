import pytest
from pathlib import Path

# Import the module as a namespace
from intseq_bert import schemas
# Import the class directly (as per convention)
from intseq_bert.schemas import OEISRecord

def test_oeis_record_serialization():
    # Create a dummy record
    record = OEISRecord(
        oeis_id="A000001",
        sequence=[0, 1, 1, 2, 3, 5],
        name="Test Sequence",
        offset_a=0,
        keywords=["core", "easy"],
        related=["A000002"]
    )
    
    # To JSON Line
    json_str = record.to_json_line()
    assert '"oeis_id": "A000001"' in json_str
    assert '"sequence": [0, 1, 1, 2, 3, 5]' in json_str
    
    # From JSON Line
    record_back = OEISRecord.from_json_line(json_str)
    assert record_back == record
    assert record_back.sequence == [0, 1, 1, 2, 3, 5]

def test_file_io(tmp_path):
    # Prepare multiple records
    records = [
        OEISRecord("A001", [1, 2], name="Seq 1"),
        OEISRecord("A002", [3, 4], name="Seq 2", keywords=["test"])
    ]
    
    file_path = tmp_path / "test_data.jsonl"
    
    # Save using module namespace
    schemas.save_records(records, str(file_path))
    
    # Check if file exists
    assert file_path.exists()
    
    # Load using module namespace
    loaded = schemas.load_records(str(file_path))
    
    assert len(loaded) == 2
    assert loaded[0].oeis_id == "A001"
    assert loaded[1].keywords == ["test"]
    assert loaded[0].sequence == [1, 2]

def test_empty_defaults():
    # Test if defaults work correctly (empty lists vs None)
    data = {"oeis_id": "A999", "sequence": [1]}
    record = OEISRecord.from_dict(data)
    
    assert record.keywords == []
    assert record.related == []
    assert record.offset_a == 0
    assert record.metadata == {}