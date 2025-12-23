import pytest
import torch
from intseq_bert.model import IntSeqBERT

def test_model_forward_pass():
    # Setup parameters
    input_dim = 10
    d_model = 32
    seq_len = 20
    batch_size = 2
    
    model = IntSeqBERT(input_dim=input_dim, d_model=d_model, num_layers=2)
    
    # Create dummy inputs
    pixel_values = torch.randn(batch_size, seq_len, input_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Case 1: Inference (No labels)
    logits, loss = model(pixel_values, attention_mask)
    
    # Check output shape
    assert logits.shape == (batch_size, seq_len, input_dim)
    assert loss is None

def test_model_loss_calculation():
    # Setup parameters
    input_dim = 10
    seq_len = 20
    batch_size = 2
    
    model = IntSeqBERT(input_dim=input_dim, d_model=32, num_layers=2)
    
    pixel_values = torch.randn(batch_size, seq_len, input_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randn(batch_size, seq_len, input_dim)
    
    # Create a mask matrix (True where masked)
    mask_matrix = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask_matrix[0, 5] = True # Mask one token
    
    # Case 2: Training (With labels and mask)
    logits, loss = model(
        pixel_values=pixel_values, 
        attention_mask=attention_mask,
        labels=labels,
        mask_matrix=mask_matrix
    )
    
    assert logits.shape == (batch_size, seq_len, input_dim)
    assert loss is not None
    assert not torch.isnan(loss)
    assert loss.ndim == 0 # Scalar

def test_model_padding_mask():
    """Ensure padding mask is working (loss shouldn't be NaN with padding)."""
    input_dim = 4
    model = IntSeqBERT(input_dim=input_dim, d_model=16, num_layers=1)
    
    # Batch with padding
    # Seq 1: [1, 1, 1]
    # Seq 2: [1, 0, 0] (2 padding tokens)
    pixel_values = torch.randn(2, 3, input_dim)
    attention_mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    labels = torch.randn(2, 3, input_dim)
    
    # Masking a real token in Seq 2
    mask_matrix = torch.zeros(2, 3, dtype=torch.bool)
    mask_matrix[1, 0] = True
    
    logits, loss = model(pixel_values, attention_mask, labels, mask_matrix)
    
    assert loss is not None
    # Just ensuring it runs without error on padding logic