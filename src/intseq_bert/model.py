import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

class IntSeqBERT(nn.Module):
    """
    BERT-like Encoder for Integer Sequences.
    Predicts masked feature vectors (Regression).
    """
    def __init__(
        self,
        input_dim: int = 27,      # Feature dimension (from features.py)
        d_model: int = 128,       # Internal dimension
        nhead: int = 4,           # Number of attention heads
        num_layers: int = 4,      # Number of transformer layers
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 2048       # Maximum sequence length supported
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Input Projection (Instead of Word Embeddings)
        # Projects continuous 27-dim vector to d_model-dim
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding (Learnable)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # 3. Transformer Encoder Backbone
        # batch_first=True makes input [Batch, Seq, Dim]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            norm_first=True # Pre-LN is generally more stable
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Prediction Head (Regression)
        # Projects hidden state back to original feature dimension
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, input_dim)
        )

        # Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights like BERT."""
        if isinstance(module, nn.Linear):
            # Slightly different from standard BERT, adapted for regression stability
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, 
        pixel_values: torch.Tensor,        # [Batch, SeqLen, InputDim] (Masked Input)
        attention_mask: torch.Tensor,      # [Batch, SeqLen] (1=Real, 0=Pad)
        labels: Optional[torch.Tensor] = None, # [Batch, SeqLen, InputDim] (Ground Truth)
        mask_matrix: Optional[torch.Tensor] = None # [Batch, SeqLen] (Boolean: True=Masked)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, seq_len, _ = pixel_values.shape
        device = pixel_values.device

        # --- A. Embeddings ---
        # Project inputs
        x = self.input_projection(pixel_values) # [Batch, SeqLen, d_model]
        
        # Add Position Embeddings
        positions = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0) # [1, SeqLen]
        x = x + self.pos_embedding(positions)
        
        # --- B. Encoder ---
        # PyTorch Transformer expects src_key_padding_mask as True for PADDING tokens.
        # Our attention_mask is 1 for REAL, 0 for PAD. So we invert it.
        # (attention_mask == 0) -> True (is padding)
        key_padding_mask = (attention_mask == 0) 
        
        hidden_states = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        
        # --- C. Prediction ---
        logits = self.prediction_head(hidden_states) # [Batch, SeqLen, InputDim]
        
        # --- D. Loss Calculation ---
        loss = None
        if labels is not None:
            # MSE Loss
            loss_fct = nn.MSELoss(reduction='none')
            loss_per_element = loss_fct(logits, labels) # [Batch, SeqLen, InputDim]
            
            # Reduce over feature dimension -> [Batch, SeqLen]
            loss_per_token = loss_per_element.mean(dim=-1)
            
            if mask_matrix is not None:
                # Only calculate loss on MASKED tokens
                # Apply mask (boolean)
                masked_loss = loss_per_token[mask_matrix]
                if masked_loss.numel() > 0:
                    loss = masked_loss.mean()
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                # If no mask matrix provided, calculate over all non-padding tokens
                active_loss = loss_per_token * attention_mask
                loss = active_loss.sum() / (attention_mask.sum() + 1e-6)

        return logits, loss