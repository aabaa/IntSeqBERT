import gzip
import logging
from pathlib import Path
from typing import Iterator, Tuple, Optional, Union, TextIO

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
