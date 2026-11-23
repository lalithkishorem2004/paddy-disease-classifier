"""
Data Preparation Script for Paddy Leaf Disease Classifier.
Filters the raw dataset to the 5 target classes, performs a stratified split
(80% train, 10% val, 10% test), and copies files into the processed directory.
Also prints a summary table of class distributions.
"""

import os
import sys
import shutil
import random
from pathlib import Path

# Add parent directory of this script to system path to ensure config is importable
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

import config

def prepare_data():
    # Set random seed for reproducibility
    random.seed(config.RANDOM_SEED)
    
    print("Starting data preparation...")
    print(f"Source images directory: {config.RAW_IMAGES_DIR}")
    print(f"Target processed directory: {config.FIELD_PROCESSED_DIR}")
    
    # Check if raw images directory exists
    if not config.RAW_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Raw images directory not found at: {config.RAW_IMAGES_DIR}. "
            "Please ensure the Kaggle dataset is downloaded and relocated correctly."
        )

    # Initialize statistics dict
    stats = {}
    
    # Process each target class
    for class_name in config.TARGET_CLASSES:
        class_src_dir = config.RAW_IMAGES_DIR / class_name
        if not class_src_dir.exists():
            print(f"Warning: Directory for class '{class_name}' not found at {class_src_dir}. Skipping.")
            continue
            
        # Get all image paths (filtering for jpeg/jpg)
        images = list(class_src_dir.glob("*.jpg")) + list(class_src_dir.glob("*.jpeg"))
        # Sort to ensure determinism before shuffling
        images.sort(key=lambda x: x.name)
        # Shuffle
        random.shuffle(images)
        
        n_images = len(images)
        if n_images == 0:
            print(f"Warning: No images found for class '{class_name}'.")
            continue
            
        # Calculate split sizes
        n_train = int(n_images * config.SPLIT_RATIOS["train"])
        n_val = int(n_images * config.SPLIT_RATIOS["val"])
        
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }
        
        # Save split counts for summary
        stats[class_name] = {
            "total": n_images,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"])
        }
        
        # Copy files to destination splits
        for split_name, split_files in splits.items():
            split_dest_dir = config.FIELD_PROCESSED_DIR / split_name / class_name
            # Create folder if it doesn't exist
            split_dest_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in split_files:
                dest_path = split_dest_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                
    # Print summary table
    print_summary(stats)

def print_summary(stats):
    print("\n" + "="*70)
    print("                   Paddy Dataset Splitting Summary")
    print("="*70)
    print(f"{'Class Name':<25} | {'Total':<8} | {'Train':<8} | {'Val':<8} | {'Test':<8}")
    print("-"*70)
    
    totals = {"total": 0, "train": 0, "val": 0, "test": 0}
    for class_name, counts in stats.items():
        print(f"{class_name:<25} | {counts['total']:<8} | {counts['train']:<8} | {counts['val']:<8} | {counts['test']:<8}")
        for key in totals:
            totals[key] += counts[key]
            
    print("-"*70)
    print(f"{'Total (All Classes)':<25} | {totals['total']:<8} | {totals['train']:<8} | {totals['val']:<8} | {totals['test']:<8}")
    print("="*70 + "\n")

if __name__ == "__main__":
    prepare_data()
