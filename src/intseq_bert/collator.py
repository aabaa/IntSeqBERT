import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class IntSeqCollator:
    """
    Collator for IntSeqBERT.
    Handles padding and dynamic masking for continuous feature vectors.
    """
    feature_dim: int = 27
    mask_prob: float = 0.15
    
    def __call__(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            examples: List of tensors, each shape [SeqLen, FeatureDim]
        Returns:
            Dict with keys:
            - inputs: Masked input tensor [Batch, MaxLen, FeatureDim]
            - attention_mask: 1 for real tokens, 0 for pad [Batch, MaxLen]
            - labels: Original input tensor (for loss calc) [Batch, MaxLen, FeatureDim]
            - mask_matrix: Boolean mask indicating which positions are masked [Batch, MaxLen]
        """
        # 1. Padding
        # batch_first=True -> [Batch, MaxLen, FeatureDim]
        padded_inputs = pad_sequence(examples, batch_first=True, padding_value=0.0)
        
        batch_size, max_len, _ = padded_inputs.shape
        
        # 2. Create Attention Mask (based on original lengths)
        # Initialize with zeros (padding)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, seq in enumerate(examples):
            length = seq.shape[0]
            attention_mask[i, :length] = 1
            
        # 3. Dynamic Masking Logic
        # Create a mask for masking (True = should be masked)
        # Probability check
        probability_matrix = torch.full((batch_size, max_len), self.mask_prob)
        mask_matrix = torch.bernoulli(probability_matrix).bool()
        
        # Constraint: Do NOT mask padding tokens
        # mask_matrix should be False where attention_mask is 0
        mask_matrix = mask_matrix & (attention_mask.bool())
        
        # 4. Apply Mask to Inputs
        # Create a copy for labels (Ground Truth)
        labels = padded_inputs.clone()
        
        # Clone inputs to create masked version
        inputs = padded_inputs.clone()
        
        # Apply zero masking (Vector-wise)
        # We mask the entire feature vector [D] for the selected positions
        # mask_matrix is [Batch, MaxLen], inputs is [Batch, MaxLen, Dim]
        # We need to broadcast mask to the last dimension
        inputs[mask_matrix] = 0.0
        
        return {
            "inputs": inputs,
            "attention_mask": attention_mask,
            "labels": labels,
            "mask_matrix": mask_matrix
        }