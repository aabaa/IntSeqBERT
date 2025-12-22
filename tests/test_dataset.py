# tests/test_dataset.py
import torch
import pytest
from intseq_bert.dataset import OEISDataset

def test_dataset_initialization():
    ds = OEISDataset() # dummy data
    assert len(ds) > 0
    seq = ds[0]
    assert isinstance(seq, list)
    assert isinstance(seq[0], int)

def test_feature_extraction_dimensions():
    ds = OEISDataset()
    seq = [1, 2, 3, 4, 5]
    features = ds.process_seq(seq)
    
    assert isinstance(features, torch.Tensor)
    # Shape check: [SeqLen, 24]
    assert features.shape == (5, 24)

def test_feature_correctness():
    ds = OEISDataset()
    # Typical sequence: [0, 1, 2, 4, -3]
    # 0: Zero flag
    # 1: Unit
    # 2: Prime, Even
    # 4: Square, Power of 2
    # -3: Negative, Prime(abs)
    seq = [0, 1, 2, 4, -3]
    feats = ds.process_seq(seq)
    
    # 24 dimensions:
    # 0-4: Analytic (Log, Diff1, Diff2, Sign, Dir)
    # 5-14: Algebraic (Mods)
    # 15-17: Valuations
    # 18: IsZero
    # 19: IsSqFree
    # 20: IsPrime
    # 21: IsSquare
    # 22: Pop
    # 23: DigitSum

    # Check 0
    f0 = feats[0]
    assert f0[18] == 1.0 # IsZero
    assert f0[20] == 0.0 # IsPrime (0 is not prime)
    assert f0[0] == 0.0  # Log(0+1) -> 0 ? or Log(0) handling in utils returns 0

    # Check 2 (Prime)
    f2 = feats[2] 
    assert f2[20] == 1.0 # IsPrime
    assert f2[21] == 0.0 # IsSquare

    # Check 4 (Square)
    f4 = feats[3]
    assert f4[20] == 0.0 # IsPrime
    assert f4[21] == 1.0 # IsSquare
    
    # Check -3 (Negative)
    f_neg = feats[4]
    assert f_neg[3] == -1.0 # Sign
    assert f_neg[20] == 1.0 # IsPrime(abs) -> True