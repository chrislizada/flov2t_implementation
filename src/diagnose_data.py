#!/usr/bin/env python3
"""
Diagnostic script to check CICIDS data loading and identify issues
"""
import numpy as np
import argparse
from load_cicids_data import load_cicids_data
from config import FLoV2TConfig

def diagnose_dataset(data_path, dataset_name, max_samples=100):
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC: Loading {dataset_name} from {data_path}")
    print(f"{'='*80}\n")
    
    config = FLoV2TConfig()
    
    try:
        data, labels = load_cicids_data(
            data_path=data_path,
            dataset=dataset_name,
            max_samples_per_file=max_samples
        )
        
        print(f"\n{'='*80}")
        print(f"DATA LOADED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Total samples: {len(data)}")
        print(f"Data shape: {data.shape}")
        print(f"Data dtype: {data.dtype}")
        print(f"Data range: [{data.min()}, {data.max()}]")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique labels: {np.unique(labels)}")
        
        print(f"\n{'='*80}")
        print(f"CLASS DISTRIBUTION")
        print(f"{'='*80}")
        for label in np.unique(labels):
            count = np.sum(labels == label)
            percentage = 100 * count / len(labels)
            class_name = config.ATTACK_CATEGORIES.get(label, f"Unknown-{label}")
            print(f"Class {label} ({class_name}): {count} samples ({percentage:.2f}%)")
        
        print(f"\n{'='*80}")
        print(f"DATA QUALITY CHECKS")
        print(f"{'='*80}")
        
        # Check for empty/zero images
        zero_images = np.sum(np.all(data == 0, axis=(1, 2, 3)))
        print(f"Empty images (all zeros): {zero_images} ({100*zero_images/len(data):.2f}%)")
        
        # Check image statistics
        print(f"\nImage statistics (per channel):")
        for ch in range(3):
            ch_data = data[:, ch, :, :]
            print(f"  Channel {ch}: mean={ch_data.mean():.2f}, std={ch_data.std():.2f}")
        
        # Check if all channels are identical (grayscale issue)
        identical_channels = np.all(data[:, 0] == data[:, 1]) and np.all(data[:, 1] == data[:, 2])
        if identical_channels:
            print(f"\n⚠️  WARNING: All RGB channels are identical (grayscale data)")
        
        # Check label distribution per class
        print(f"\n{'='*80}")
        print(f"IMBALANCE ANALYSIS")
        print(f"{'='*80}")
        class_counts = [np.sum(labels == i) for i in range(config.NUM_CLASSES)]
        max_count = max([c for c in class_counts if c > 0])
        min_count = min([c for c in class_counts if c > 0])
        print(f"Imbalance ratio: {max_count/min_count:.1f}:1")
        print(f"Most common class: {max_count} samples")
        print(f"Least common class: {min_count} samples")
        
        if max_count / min_count > 100:
            print(f"\n⚠️  SEVERE CLASS IMBALANCE DETECTED")
            print(f"   Recommendation: Use class-weighted loss function")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR LOADING DATA")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diagnose CICIDS data loading')
    parser.add_argument('--data_path', type=str, required=True, help='Path to PCAP directory')
    parser.add_argument('--dataset', type=str, default='CICIDS2017', choices=['CICIDS2017', 'CICIDS2018'])
    parser.add_argument('--max_samples', type=int, default=100, help='Max samples per PCAP file')
    
    args = parser.parse_args()
    
    success = diagnose_dataset(args.data_path, args.dataset, args.max_samples)
    
    if success:
        print(f"\n✓ Diagnosis complete - data looks ready for training")
    else:
        print(f"\n✗ Diagnosis failed - check errors above")
