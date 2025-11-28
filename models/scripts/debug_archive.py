#!/usr/bin/env python3
"""
Debug script for archive dataset
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from preprocessing import DatasetBuilder
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_archive_dataset():
    """Debug the archive dataset building process."""
    print("Debugging archive dataset...")
    
    # Test DatasetBuilder
    builder = DatasetBuilder("./data")
    
    # Check if paths exist
    archive_base = Path("archive/Dataset")
    print(f"Archive base path exists: {archive_base.exists()}")
    
    if archive_base.exists():
        for split in ['Train', 'Validation', 'Test']:
            split_path = archive_base / split
            print(f"  {split} path exists: {split_path.exists()}")
            
            if split_path.exists():
                real_path = split_path / "Real"
                fake_path = split_path / "Fake"
                print(f"    {split}/Real exists: {real_path.exists()}")
                print(f"    {split}/Fake exists: {fake_path.exists()}")
                
                if real_path.exists():
                    real_count = sum(1 for _ in real_path.glob("*.*"))
                    print(f"    {split}/Real files: {real_count}")
                
                if fake_path.exists():
                    fake_count = sum(1 for _ in fake_path.glob("*.*"))
                    print(f"    {split}/Fake files: {fake_count}")
    
    # Test building archive dataset
    print("\nBuilding archive dataset...")
    train_data = builder.build_archive_dataset('train')
    print(f"Train data samples: {len(train_data)}")
    
    if train_data:
        print("First few samples:")
        for i, sample in enumerate(train_data[:3]):
            print(f"  {i+1}. {sample}")

if __name__ == "__main__":
    debug_archive_dataset()