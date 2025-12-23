import gzip
import logging
from pathlib import Path
from typing import Iterator, Tuple, Optional, Union, TextIO, Dict, Any
import re
import io

# Import schemas
from . import schemas

logger = logging.getLogger(__name__)

class StrippedConverter:
    """
    Parses OEIS 'stripped' format and yields OEISRecord objects.
    Focuses solely on parsing logic and data validation.
    """
    def __init__(self, min_len: int = 2, max_val_threshold: int = 10**100):
        self.min_len = min_len
        self.max_val_threshold = max_val_threshold

    def parse(self, file_handle: TextIO) -> Iterator[schemas.OEISRecord]:
        """
        Yields valid OEISRecord objects from a file-like object.
        """
        for line in file_handle:
            # Format: "A000001 ,1,2,5,..."
            if len(line) < 10: 
                continue
            
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            
            oeis_id = parts[0].strip()
            
            seq = []
            is_valid = True
            
            try:
                for x_str in parts[1:]:
                    x_clean = x_str.strip()
                    if not x_clean: continue
                    
                    val = int(x_clean)
                    if abs(val) > self.max_val_threshold:
                        is_valid = False
                        break
                    seq.append(val)
            except ValueError:
                is_valid = False
            
            if is_valid and len(seq) >= self.min_len:
                yield schemas.OEISRecord(oeis_id=oeis_id, sequence=seq)


class NamesConverter:
    """
    Parses OEIS 'names' format and yields (id, name) tuples.
    """
    def parse(self, file_handle: TextIO) -> Iterator[Tuple[str, str]]:
        """
        Yields (oeis_id, name) tuples.
        """
        for line in file_handle:
            if line.startswith("#") or not line.strip():
                continue
            
            # Format: "A000001 Number of groups of order n."
            # Split only on the first space
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
                
            oeis_id = parts[0].strip()
            name = parts[1].strip()
            
            yield oeis_id, name


class SeqMetadataConverter:
    """
    Parses OEIS '.seq' format to extract rich metadata.
    Format: %Tag OEIS_ID Content
    """
    def __init__(self):
        # Regex to extract cross-references (A followed by 6 digits)
        self.re_id = re.compile(r'A\d{6}')

    def parse(self, file_handle: TextIO, expected_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Parses a .seq file and returns a dictionary of metadata updates.
        Args:
            file_handle: Open file object.
            expected_id: If provided, ensures lines verify against this ID.
        """
        metadata = {
            "keywords": [],
            "offset_a": 0,
            "related": []
        }
        related_ids = set()
        
        for line in file_handle:
            line = line.strip()
            if len(line) < 10: continue # Minimum length for "%T A000000 X"
            
            # Line structure: "%K A000001 nonn,core,nice"
            # Split into 3 parts: Tag, ID, Content
            parts = line.split(" ", 2)
            if len(parts) < 3: 
                continue
                
            tag, line_id, content = parts[0], parts[1], parts[2]
            
            # Verify ID consistency
            if expected_id and line_id != expected_id:
                # Skip lines belonging to other IDs (shouldn't happen in valid .seq files)
                continue
            
            if tag == "%K":
                # Keywords: comma separated
                keys = [k.strip() for k in content.split(",") if k.strip()]
                metadata["keywords"] = keys
                
            elif tag == "%O":
                # Offset: "0,5" -> start_index=0
                # Content might be "0,5"
                offset_parts = content.split(",")
                if len(offset_parts) >= 1:
                    try:
                        metadata["offset_a"] = int(offset_parts[0])
                    except ValueError:
                        pass 
                        
            elif tag == "%Y":
                # Cross-refs: "Cf. A000002, A000005."
                # Extract all Axxxxxx patterns
                found = self.re_id.findall(content)
                # Exclude self-reference if present
                if expected_id and expected_id in found:
                    found.remove(expected_id)
                related_ids.update(found)
                
            # Future: %F (Formula), etc.

        metadata["related"] = sorted(list(related_ids))
        return metadata