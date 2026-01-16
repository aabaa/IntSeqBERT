
import torch
import torch.nn.functional as F

def test_filtering_logic():
    print("Testing filtering logic...")
    
    # 1. Setup Dummy Data
    B, L = 2, 5
    # mag_labels: [log, s+, s-, s0]
    # Batch 0: partial padding at end
    # Batch 1: middle zero considered valid by mask
    mag_labels = torch.zeros(B, L, 4)
    mag_labels[0, :, 0] = torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0]) # 0.0 is padding
    mag_labels[1, :, 0] = torch.tensor([10.0, 0.0, 5.0, 0.0, 0.0]) # Index 1 is zero
    
    # Mask Matrix: True where we want to predict
    mask_matrix = torch.tensor([
        [True, True, True, False, False],  # Index 2 is 0.0 but Masked=True (Simulating the bug)
        [True, True, True, False, False]   # Index 1 is 0.0 but Masked=True
    ], dtype=torch.bool)
    
    batch = {
        "mag_labels": mag_labels,
        "mask_matrix": mask_matrix
    }
    
    print("Initial Mask Matrix:")
    print(batch["mask_matrix"])
    
    # 2. Apply Logic from collect_predictions
    gt_mag = mag_labels[:, :, 0]  # Extract log value
    
    # Filter out padding (0.00) from mask
    is_valid_value = (gt_mag.abs() > 1e-6)
    
    print(f"\nis_valid_value:\n{is_valid_value}")
    
    batch["mask_matrix"] = batch["mask_matrix"] & is_valid_value
    
    print(f"\nUpdated Mask Matrix:")
    print(batch["mask_matrix"])
    
    # 3. Verify
    # Batch 0, Index 2 should be False (was True, gt=0)
    assert batch["mask_matrix"][0, 2] == False, "Failed to filter Batch 0 Index 2"
    # Batch 1, Index 1 should be False (was True, gt=0)
    assert batch["mask_matrix"][1, 1] == False, "Failed to filter Batch 1 Index 1"
    
    print("\nSUCCESS: Filtering logic works as expected.")

def test_extract_worst_k():
    print("\nTesting extract_worst_k logic...")
    gt = torch.tensor([[100.0, 0.0, 50.0]])
    pred = torch.tensor([[10.0, 10.0, 10.0]])
    mask = torch.tensor([[True, False, True]]) # Middle is 0.0, filtered out
    
    errors = (gt - pred).abs() # [90, 10, 40]
    
    if mask is not None:
        errors = errors * mask.float() # [90, 0, 40]
        
    print(f"Errors after masking:\n{errors}")
    
    assert errors[0, 1] == 0.0, "Masked error is not zero"
    
    flat_errors = errors.flatten()
    topk = torch.topk(flat_errors, k=2)
    
    print(f"Top 2 indices: {topk.indices}")
    # Should be 0 (90) and 2 (40). NOT 1.
    
    assert 1 not in topk.indices, "Masked error was selected in Top-K"
    print("SUCCESS: Selection logic works.")

if __name__ == "__main__":
    test_filtering_logic()
    test_extract_worst_k()
