import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from pathlib import Path

@dataclass
class OEISRecord:
    """
    A unified data structure representing a single OEIS sequence.
    Optimized for JSON Lines serialization.
    """
    oeis_id: str
    sequence: List[int]
    name: str = ""
    offset_a: int = 0
    keywords: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)
    
    # Placeholder for extra metadata (e.g., author, references)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the record to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OEISRecord':
        """Creates an instance from a dictionary."""
        return cls(
            oeis_id=data.get("oeis_id", ""),
            sequence=data.get("sequence", []),
            name=data.get("name", ""),
            offset_a=data.get("offset_a", 0),
            keywords=data.get("keywords", []),
            related=data.get("related", []),
            metadata=data.get("metadata", {})
        )

    def to_json_line(self) -> str:
        """Serializes to a single line JSON string (ensure_ascii=False for UTF-8)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json_line(cls, line: str) -> 'OEISRecord':
        """Parses a single line JSON string."""
        return cls.from_dict(json.loads(line))

    def __str__(self):
        """Human-readable short representation for debugging."""
        seq_preview = str(self.sequence[:5]) + "..." if len(self.sequence) > 5 else str(self.sequence)
        return f"[{self.oeis_id}] {self.name} (Offset:{self.offset_a}) Seq:{seq_preview}"


# --- IO Helper Functions ---

def save_records(records: List[OEISRecord], filepath: str):
    """
    Saves a list of OEISRecord objects to a JSONL file.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(record.to_json_line() + '\n')

def load_records(filepath: str) -> List[OEISRecord]:
    """
    Loads a list of OEISRecord objects from a JSONL file.
    """
    records = []
    path = Path(filepath)
    if not path.exists():
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(OEISRecord.from_json_line(line))
    return records