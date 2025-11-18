import os
import numpy as np
import argparse
import json
from typing import Tuple
from load_cicids_data import load_cicids_data
from config import FLoV2TConfig

def create_test_set(data_path: str, dataset: str, output_dir: str, 
                   test_ratio: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract test samples from CICIDS dataset for final evaluation.
    This is separate from the 80-20 train/test split used during training.
    
    Args:
        data_path: Path to PCAP files
        dataset: 'CICIDS2017' or 'CICIDS2018'
        output_dir: Directory to save test set
        test_ratio: Ratio of data to use for testing (default: 0.1 = 10%)
        seed: Random seed for reproducibility
    
    Returns:
        test_data, test_labels
    """
    
    np.random.seed(seed)
    
    print(f"\n{'='*80}")
    print(f"Creating Test Set for {dataset}")
    print(f"{'='*80}")
    print(f"Test ratio: {test_ratio*100:.0f}%")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    print(f"{'='*80}\n")
    
    # Load full dataset
    data, labels = load_cicids_data(data_path=data_path, dataset=dataset)
    
    total_samples = len(data)
    test_size = int(total_samples * test_ratio)
    
    print(f"\n{'='*80}")
    print(f"Dataset Statistics")
    print(f"{'='*80}")
    print(f"Total samples: {total_samples}")
    print(f"Test samples (10%): {test_size}")
    print(f"{'='*80}\n")
    
    # Stratified sampling to maintain class distribution
    config = FLoV2TConfig()
    test_indices = []
    
    print("Performing stratified sampling to maintain class distribution...")
    print(f"{'='*80}")
    
    for class_id in range(config.NUM_CLASSES):
        class_indices = np.where(labels == class_id)[0]
        class_count = len(class_indices)
        
        if class_count == 0:
            print(f"Class {class_id} ({config.ATTACK_CATEGORIES[class_id]:20s}): No samples")
            continue
        
        class_test_size = max(1, int(class_count * test_ratio))
        class_test_indices = np.random.choice(class_indices, size=class_test_size, replace=False)
        test_indices.extend(class_test_indices)
        
        print(f"Class {class_id} ({config.ATTACK_CATEGORIES[class_id]:20s}): {class_count:6d} total → {class_test_size:5d} test")
    
    print(f"{'='*80}\n")
    
    test_indices = np.array(test_indices)
    np.random.shuffle(test_indices)
    
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save test set
    test_data_path = os.path.join(output_dir, f'{dataset}_test_data.npy')
    test_labels_path = os.path.join(output_dir, f'{dataset}_test_labels.npy')
    
    np.save(test_data_path, test_data)
    np.save(test_labels_path, test_labels)
    
    print(f"{'='*80}")
    print(f"Test Set Saved")
    print(f"{'='*80}")
    print(f"Data: {test_data_path}")
    print(f"  Shape: {test_data.shape}")
    print(f"  Size: {test_data.nbytes / (1024**2):.2f} MB")
    print(f"\nLabels: {test_labels_path}")
    print(f"  Shape: {test_labels.shape}")
    print(f"  Size: {test_labels.nbytes / (1024**2):.2f} MB")
    print(f"{'='*80}\n")
    
    # Save metadata
    metadata = {
        'dataset': dataset,
        'total_samples': int(total_samples),
        'test_samples': int(len(test_data)),
        'test_ratio': test_ratio,
        'seed': seed,
        'data_shape': test_data.shape,
        'labels_shape': test_labels.shape,
        'class_distribution': {}
    }
    
    print("Test Set Class Distribution:")
    print(f"{'='*80}")
    unique, counts = np.unique(test_labels, return_counts=True)
    for class_id, count in zip(unique, counts):
        class_name = config.ATTACK_CATEGORIES[class_id]
        percentage = (count / len(test_labels)) * 100
        metadata['class_distribution'][int(class_id)] = {
            'name': class_name,
            'count': int(count),
            'percentage': float(percentage)
        }
        print(f"Class {class_id} ({class_name:20s}): {count:5d} samples ({percentage:5.2f}%)")
    print(f"{'='*80}\n")
    
    metadata_path = os.path.join(output_dir, f'{dataset}_test_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}\n")
    
    print(f"{'='*80}")
    print(f"Test Set Creation Complete")
    print(f"{'='*80}")
    print(f"\nTo load the test set:")
    print(f"  test_data = np.load('{test_data_path}')")
    print(f"  test_labels = np.load('{test_labels_path}')")
    print(f"\nTo use for evaluation:")
    print(f"  python src/evaluate_deployment.py --test_data {output_dir}")
    print(f"{'='*80}\n")
    
    return test_data, test_labels

def main():
    parser = argparse.ArgumentParser(description='Create test set from CICIDS dataset')
    parser.add_argument('--data_path', type=str, required=True, help='Path to PCAP files')
    parser.add_argument('--dataset', type=str, default='CICIDS2017', 
                       choices=['CICIDS2017', 'CICIDS2018'], help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='./test_data', 
                       help='Output directory for test set')
    parser.add_argument('--test_ratio', type=float, default=0.1, 
                       help='Ratio of data for testing (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    create_test_set(
        data_path=args.data_path,
        dataset=args.dataset,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
