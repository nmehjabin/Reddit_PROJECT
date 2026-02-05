import pandas as pd

# ============================================================================
# CONVERT MULTI-CLASS LABELS TO BINARY
# Mapping: '1' = 1 (burnout), everything else = 0 (no burnout)
# ============================================================================

# Load the original ground truth data
ground_truth_path = 'test_data_with_labels.csv'
ground_truth = pd.read_csv(ground_truth_path)

print("="*70)
print("CONVERTING LABELS TO BINARY")
print("="*70)

print(f"\nOriginal data:")
print(f"  Total samples: {len(ground_truth)}")
print(f"\nOriginal label distribution:")
print(ground_truth['label'].value_counts().sort_index())

# Conversion function
def convert_to_binary(label):
    """
    Convert label to binary:
    - '1' → 1 (burnout)
    - Everything else → 0 (no burnout)
    """
    label_str = str(label).strip()
    
    # If it's exactly '1', keep as 1
    if label_str == '1':
        return 1
    # Everything else becomes 0
    else:
        return 0

# Apply conversion
ground_truth_binary = ground_truth.copy()
ground_truth_binary['label'] = ground_truth_binary['label'].apply(convert_to_binary)

print(f"\nAfter conversion:")
print(f"\nNew label distribution:")
print(ground_truth_binary['label'].value_counts().sort_index())

# Verify conversion
print(f"\nConversion details:")
original_counts = ground_truth['label'].value_counts()
for orig_label in sorted(ground_truth['label'].unique()):
    new_label = convert_to_binary(orig_label)
    count = original_counts[orig_label]
    print(f"  '{orig_label}' → {new_label} ({count} samples)")

# Save the binary version
output_path = 'test_data_with_labels_BINARY.csv'
ground_truth_binary.to_csv(output_path, index=False)

print(f"\n" + "="*70)
print(" SUCCESS!")
print("="*70)
print(f"Binary ground truth saved to:")
print(f"  {output_path}")
print(f"\nFinal distribution:")
print(f"  Label 0 (no burnout): {(ground_truth_binary['label'] == 0).sum()} samples")
print(f"  Label 1 (burnout):    {(ground_truth_binary['label'] == 1).sum()} samples")
print(f"  Total:                {len(ground_truth_binary)} samples")

