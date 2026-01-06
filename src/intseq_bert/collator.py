import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import List, Dict, Any, Union

@dataclass
class DualStreamCollator:
    """
    Collator for Dual Stream Architecture.
    Handles padding and masking for both Magnitude and Mod Spectrum streams.
    Also batches the training targets.
    """
    mask_prob: float = 0.15
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Args:
            batch: List of dicts from DualStreamDataset.
                   Each dict contains:
                   - 'mag_features': (L, 5)
                   - 'mod_features': (L, 200)
                   - 'targets': Dict[str, Tensor] (L,)
        
        Returns:
            Dict with keys:
            - mag_inputs: Masked magnitude features (B, L, 5)
            - mod_inputs: Masked mod features (B, L, 200)
            - attention_mask: (B, L)
            - mask_matrix: (B, L) Boolean mask
            - mag_labels: Original unmasked magnitude features (for BERT pretraining loss)
            - mod_labels: Original unmasked mod features (for BERT pretraining loss)
            - targets: Dict of batched target tensors (for Decoder training)
        """
        # 1. Unpack data streams
        mag_list = [item['mag_features'] for item in batch]
        mod_list = [item['mod_features'] for item in batch]
        
        # 2. Basic Padding (Inputs)
        # batch_first=True -> [Batch, MaxLen, FeatureDim]
        mag_padded = pad_sequence(mag_list, batch_first=True, padding_value=0.0)
        mod_padded = pad_sequence(mod_list, batch_first=True, padding_value=0.0)
        
        batch_size = mag_padded.size(0)
        max_len = mag_padded.size(1)
        
        # 3. Create Attention Mask
        # Based on actual lengths of sequences
        lengths = torch.tensor([len(x) for x in mag_list])
        # (B, L) range matrix compared against lengths
        attention_mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        attention_mask = attention_mask.long()
        
        # 4. Dynamic Masking Logic
        # Create a mask for masking (True = should be masked)
        prob_matrix = torch.full((batch_size, max_len), self.mask_prob)
        # Do NOT mask padding tokens (where attention_mask is 0)
        prob_matrix[attention_mask == 0] = 0.0
        mask_matrix = torch.bernoulli(prob_matrix).bool()
        
        # 5. Apply Mask to Inputs
        # Clone for labels (Ground Truth for reconstruction pre-training)
        mag_labels = mag_padded.clone()
        mod_labels = mod_padded.clone()
        
        # Apply zero masking
        # Need to broadcast mask (B, L) to (B, L, D)
        mag_inputs = mag_padded.clone()
        mag_inputs[mask_matrix] = 0.0
        
        mod_inputs = mod_padded.clone()
        mod_inputs[mask_matrix] = 0.0
        
        # 6. Process Targets (Batched Dictionary)
        # We need to pad each target type (mod2, mod3, ..., mag)
        batched_targets = {}
        if len(batch) > 0 and 'targets' in batch[0]:
            target_keys = batch[0]['targets'].keys()
            
            for key in target_keys:
                t_list = [item['targets'][key] for item in batch]
                
                # Determine padding value
                if torch.is_floating_point(t_list[0]):
                    # Regression targets (e.g. 'mag') -> Pad with 0.0
                    pad_val = 0.0
                else:
                    # Classification targets (e.g. 'modX') -> Pad with -100 (Ignore Index)
                    pad_val = -100
                
                batched_targets[key] = pad_sequence(t_list, batch_first=True, padding_value=pad_val)
        
        return {
            "mag_inputs": mag_inputs,
            "mod_inputs": mod_inputs,
            "attention_mask": attention_mask,
            "mask_matrix": mask_matrix,
            "mag_labels": mag_labels,
            "mod_labels": mod_labels,
            "targets": batched_targets
        }
