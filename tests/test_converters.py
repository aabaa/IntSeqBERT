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