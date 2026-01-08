"""
Tests for schemas.py (OEISRecord Dataclass and IO Helpers).

Covers:
1. Required field validation (oeis_id, sequence)
2. Sequence type handling (list, string, invalid)
3. No implicit ID normalization
4. Mutable default avoidance
5. Serialization round-trip
6. File IO
"""

import pytest
from pathlib import Path

from intseq_bert import schemas
from intseq_bert.schemas import OEISRecord


# ==========================================
# 1. Required Field Validation (Fail Fast)
# ==========================================

class TestRequiredFieldValidation:
    """Tests for strict validation of required fields."""
    
    def test_missing_oeis_id_raises_value_error(self):
        """Test that missing oeis_id raises ValueError."""
        data = {"sequence": [1, 2, 3]}
        with pytest.raises(ValueError, match="Missing required key: 'oeis_id'"):
            OEISRecord.from_dict(data)
    
    def test_missing_sequence_raises_value_error(self):
        """Test that missing sequence raises ValueError."""
        data = {"oeis_id": "A000001"}
        with pytest.raises(ValueError, match="Missing required key: 'sequence'"):
            OEISRecord.from_dict(data)
    
    def test_legacy_id_key_not_supported(self):
        """Test that legacy 'id' key is not accepted."""
        data = {"id": "A000001", "sequence": [1, 2, 3]}
        with pytest.raises(ValueError, match="Missing required key: 'oeis_id'"):
            OEISRecord.from_dict(data)


# ==========================================
# 2. Sequence Type Handling
# ==========================================

class TestSequenceTypeHandling:
    """Tests for sequence type conversion and validation."""
    
    def test_list_sequence_accepted(self):
        """Test that list sequence is accepted as-is."""
        data = {"oeis_id": "A001", "sequence": [1, 2, 3]}
        record = OEISRecord.from_dict(data)
        assert record.sequence == [1, 2, 3]
        assert isinstance(record.sequence, list)
    
    def test_string_sequence_parsed(self):
        """Test that CSV-style string sequence is parsed."""
        data = {"oeis_id": "A002", "sequence": "1, 2, 3, 4, 5"}
        record = OEISRecord.from_dict(data)
        assert record.sequence == [1, 2, 3, 4, 5]
    
    def test_bracketed_string_sequence_parsed(self):
        """Test that bracketed string sequence is parsed."""
        data = {"oeis_id": "A003", "sequence": "[1, 2, 3]"}
        record = OEISRecord.from_dict(data)
        assert record.sequence == [1, 2, 3]
    
    def test_empty_string_sequence_returns_empty_list(self):
        """Test that empty string sequence returns empty list."""
        data = {"oeis_id": "A004", "sequence": ""}
        record = OEISRecord.from_dict(data)
        assert record.sequence == []
    
    def test_none_sequence_raises_type_error(self):
        """Test that None sequence raises TypeError."""
        data = {"oeis_id": "A005", "sequence": None}
        with pytest.raises(TypeError, match="Invalid type for 'sequence'"):
            OEISRecord.from_dict(data)
    
    def test_invalid_sequence_type_raises_type_error(self):
        """Test that invalid sequence type raises TypeError."""
        data = {"oeis_id": "A006", "sequence": 12345}
        with pytest.raises(TypeError, match="Invalid type for 'sequence'"):
            OEISRecord.from_dict(data)
    
    def test_malformed_string_sequence_raises_value_error(self):
        """Test that malformed string sequence raises ValueError."""
        data = {"oeis_id": "A007", "sequence": "1, 2, not_a_number"}
        with pytest.raises(ValueError, match="Malformed sequence string"):
            OEISRecord.from_dict(data)


# ==========================================
# 3. No Implicit ID Normalization
# ==========================================

class TestNoImplicitNormalization:
    """Tests that ID values are preserved exactly as given."""
    
    def test_id_preserved_exactly(self):
        """Test that ID is not modified."""
        data = {"oeis_id": "A000001", "sequence": [1]}
        record = OEISRecord.from_dict(data)
        assert record.oeis_id == "A000001"
    
    def test_short_id_not_zero_padded(self):
        """Test that short IDs are NOT zero-padded."""
        data = {"oeis_id": "A1", "sequence": [1]}
        record = OEISRecord.from_dict(data)
        assert record.oeis_id == "A1"  # NOT "A000001"
    
    def test_numeric_id_not_prefixed(self):
        """Test that numeric string IDs are NOT given A prefix."""
        data = {"oeis_id": "123", "sequence": [1]}
        record = OEISRecord.from_dict(data)
        assert record.oeis_id == "123"  # NOT "A000123"
    
    def test_integer_id_converted_to_string_only(self):
        """Test that integer ID is converted to string without modification."""
        data = {"oeis_id": 42, "sequence": [1]}
        record = OEISRecord.from_dict(data)
        assert record.oeis_id == "42"  # Just str(), no normalization


# ==========================================
# 4. Mutable Default Avoidance
# ==========================================

class TestMutableDefaultAvoidance:
    """Tests that mutable defaults are not shared between instances."""
    
    def test_keywords_not_shared(self):
        """Test that keywords list is not shared between instances."""
        record1 = OEISRecord(oeis_id="A1", sequence=[1])
        record2 = OEISRecord(oeis_id="A2", sequence=[2])
        
        record1.keywords.append("test")
        
        assert record1.keywords == ["test"]
        assert record2.keywords == []  # Should NOT be affected
    
    def test_metadata_not_shared(self):
        """Test that metadata dict is not shared between instances."""
        record1 = OEISRecord(oeis_id="A1", sequence=[1])
        record2 = OEISRecord(oeis_id="A2", sequence=[2])
        
        record1.metadata["key"] = "value"
        
        assert record1.metadata == {"key": "value"}
        assert record2.metadata == {}  # Should NOT be affected
    
    def test_from_dict_defaults_not_shared(self):
        """Test that from_dict defaults are not shared."""
        data1 = {"oeis_id": "A1", "sequence": [1]}
        data2 = {"oeis_id": "A2", "sequence": [2]}
        
        r1 = OEISRecord.from_dict(data1)
        r2 = OEISRecord.from_dict(data2)
        
        r1.keywords.append("modified")
        
        assert r2.keywords == []


# ==========================================
# 5. Serialization Round-Trip
# ==========================================

class TestSerializationRoundTrip:
    """Tests for JSON serialization and deserialization."""
    
    def test_to_dict_contains_all_fields(self):
        """Test that to_dict includes all fields."""
        record = OEISRecord(
            oeis_id="A000001",
            sequence=[1, 2, 3],
            name="Test",
            offset_a=1,
            keywords=["core"],
            related=["A000002"],
            metadata={"source": "test"}
        )
        d = record.to_dict()
        
        assert d["oeis_id"] == "A000001"
        assert d["sequence"] == [1, 2, 3]
        assert d["name"] == "Test"
        assert d["offset_a"] == 1
        assert d["keywords"] == ["core"]
        assert d["related"] == ["A000002"]
        assert d["metadata"] == {"source": "test"}
    
    def test_json_line_round_trip(self):
        """Test that JSON line serialization is reversible."""
        original = OEISRecord(
            oeis_id="A000001",
            sequence=[0, 1, 1, 2, 3, 5],
            name="Fibonacci",
            keywords=["core", "easy"]
        )
        
        json_str = original.to_json_line()
        restored = OEISRecord.from_json_line(json_str)
        
        assert restored == original
    
    def test_str_representation(self):
        """Test __str__ method."""
        record = OEISRecord(oeis_id="A001", sequence=[1, 2, 3, 4, 5, 6, 7], name="Test")
        s = str(record)
        assert "A001" in s
        assert "Test" in s


# ==========================================
# 6. File IO
# ==========================================

class TestFileIO:
    """Tests for save_records and load_records functions."""
    
    def test_save_and_load_round_trip(self, tmp_path):
        """Test that save and load are reversible."""
        records = [
            OEISRecord("A001", [1, 2], name="Seq 1"),
            OEISRecord("A002", [3, 4], name="Seq 2", keywords=["test"])
        ]
        
        file_path = tmp_path / "test_data.jsonl"
        schemas.save_records(records, str(file_path))
        
        assert file_path.exists()
        
        loaded = schemas.load_records(str(file_path))
        
        assert len(loaded) == 2
        assert loaded[0].oeis_id == "A001"
        assert loaded[1].keywords == ["test"]
    
    def test_load_nonexistent_file_returns_empty(self, tmp_path):
        """Test that loading nonexistent file returns empty list."""
        result = schemas.load_records(str(tmp_path / "nonexistent.jsonl"))
        assert result == []
    
    def test_load_propagates_validation_errors(self, tmp_path):
        """Test that load_records propagates validation errors."""
        file_path = tmp_path / "invalid.jsonl"
        with open(file_path, 'w') as f:
            f.write('{"sequence": [1, 2, 3]}\n')  # Missing oeis_id
        
        with pytest.raises(ValueError, match="Missing required key: 'oeis_id'"):
            schemas.load_records(str(file_path))