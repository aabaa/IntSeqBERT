import pytest
import io
from intseq_bert import converters, schemas

def test_stripped_converter():
    # Mock data stream
    data = "A001 ,1,2,3,4,5\nA002 ,10,20\n" # A002 is short (len=2)
    f = io.StringIO(data)
    
    converter = converters.StrippedConverter(min_len=3)
    records = list(converter.parse(f))
    
    assert len(records) == 1
    assert records[0].oeis_id == "A001"
    assert records[0].sequence == [1, 2, 3, 4, 5]

def test_stripped_converter_filter_large():
    # Test large value filtering
    data = "A001 ,1,2,1000\n"
    f = io.StringIO(data)
    
    # Threshold = 100
    converter = converters.StrippedConverter(min_len=1, max_val_threshold=100)
    records = list(converter.parse(f))
    
    # Should be filtered out because 1000 > 100
    assert len(records) == 0

def test_names_converter():
    data = """
# Comment
A001 Sequence One
A002 Sequence Two
    """.strip()
    f = io.StringIO(data)
    
    converter = converters.NamesConverter()
    names = dict(converter.parse(f))
    
    assert len(names) == 2
    assert names["A001"] == "Sequence One"
    assert names["A002"] == "Sequence Two"

def test_seq_metadata_converter():
    # Mock .seq file content conforming to actual format
    # Note: %T, %S, %N lines are ignored by the current parser logic but present in file
    data = """
%I A000001 M0098 N0035
%S A000001 0,1,1,1,2
%N A000001 Number of groups.
%O A000001 0,5
%K A000001 nonn, core, nice
%Y A000001 Cf. A000002, A000005.
%Y A000001 A000001 is self reference.
    """.strip()
    f = io.StringIO(data)
    
    converter = converters.SeqMetadataConverter()
    # Pass expected_id="A000001" to verify ID checking logic
    meta = converter.parse(f, expected_id="A000001")
    
    assert meta["offset_a"] == 0
    
    # Keywords check
    assert "nonn" in meta["keywords"]
    assert "core" in meta["keywords"]
    
    # Related check
    assert "A000002" in meta["related"]
    assert "A000005" in meta["related"]
    # Self reference should be excluded if logic implemented (optional but good practice)
    # The current regex logic just finds A-numbers. 
    # If we want to strictly exclude self, we can filter it. 
    # The implementation I provided above does filter `expected_id`.
    assert "A000001" not in meta["related"]

def test_seq_metadata_converter_mismatch():
    """Test that lines with wrong ID are skipped."""
    data = """
%K A000001 correct keyword
%K A000002 wrong keyword
    """.strip()
    f = io.StringIO(data)
    
    converter = converters.SeqMetadataConverter()
    meta = converter.parse(f, expected_id="A000001")
    
    assert "correct keyword" in meta["keywords"]
    assert "wrong keyword" not in meta["keywords"]
