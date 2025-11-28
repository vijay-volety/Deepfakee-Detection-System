#!/usr/bin/env python3
"""
Script to verify archive dataset integration and test the data loading
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from preprocessing import DatasetBuilder, DeepfakeDataset, DataAugmentation
from torch.utils.data import DataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_archive_dataset():
    """Verify that the archive dataset is properly integrated."""
    print("Verifying archive dataset integration...")
    
    # Test DatasetBuilder
    builder = DatasetBuilder("./data")
    
    # Test building archive dataset
    print("\n1. Testing archive dataset building...")
    train_data = builder.build_archive_dataset('train')
    val_data = builder.build_archive_dataset('val')
    test_data = builder.build_archive_dataset('test')
    
    print(f"   Train samples: {len(train_data)}")
    print(f"   Validation samples: {len(val_data)}")
    print(f"   Test samples: {len(test_data)}")
    
    if len(train_data) > 0:
        print(f"   Sample train item: {train_data[0]}")
    
    # Test DeepfakeDataset with archive data
    print("\n2. Testing DeepfakeDataset with archive data...")
    if len(train_data) > 0:
        train_dataset = DeepfakeDataset(
            train_data[:5],  # Just test with first 5 samples
            transform=DataAugmentation.get_validation_transforms(),
            sequence_length=16,
            is_training=False
        )
        
        print(f"   Created dataset with {len(train_dataset)} samples")
        
        # Test data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0  # Set to 0 for easier debugging
        )
        
        print("\n3. Testing data loading...")
        for i, batch in enumerate(train_loader):
            print(f"   Batch {i}:")
            print(f"     Frames shape: {batch['frames'].shape}")
            print(f"     Labels: {batch['label']}")
            print(f"     File paths: {batch['video_path']}")
            if i >= 2:  # Just test first 3 batches
                break
        
        print("\n4. Verification complete!")
        print("   ✓ Archive dataset integration successful")
        return True
    else:
        print("   ✗ No archive data found!")
        return False

if __name__ == "__main__":
    verify_archive_dataset()