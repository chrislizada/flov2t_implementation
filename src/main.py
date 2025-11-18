import torch
import numpy as np
import argparse
from data_preprocessing import TrafficPreprocessor
from models import FLoV2TModel
from federated_trainer import Client, FederatedServer
from config import FLoV2TConfig
from utils import (set_seed, create_iid_partition, create_non_iid_partition, 
                   split_train_test, count_parameters, print_data_distribution,
                   create_imbalanced_iid_partition)
from load_cicids_data import load_cicids_data

def main():
    parser = argparse.ArgumentParser(description='FLoV2T: Federated Learning for AIoT Malicious Traffic Classification')
    parser.add_argument('--num_clients', type=int, default=3, choices=[3, 5], help='Number of federated clients')
    parser.add_argument('--scenario', type=str, default='iid', choices=['iid', 'non-iid'], help='Data distribution scenario')
    parser.add_argument('--dataset', type=str, default='CICIDS2017', choices=['CICIDS2017', 'CICIDS2018'], help='Dataset to use')
    parser.add_argument('--num_rounds', type=int, default=20, help='Number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=5, help='Local training epochs per round')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--aggregation', type=str, default='rgpa', choices=['rgpa', 'fedavg'], help='Aggregation method')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_path', type=str, default=None, help='Path to PCAP data directory')
    parser.add_argument('--max_samples_per_file', type=int, default=None, help='Max samples per PCAP file (for testing)')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config = FLoV2TConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("FLoV2T: Federated Learning for AIoT Malicious Traffic Classification")
    print("="*80)
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Scenario: {args.scenario.upper()}")
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Aggregation: {args.aggregation.upper()}")
    print(f"  LoRA Rank: {config.LORA_RANK}, Alpha: {config.LORA_ALPHA}")
    print("="*80 + "\n")
    
    if args.data_path:
        try:
            data, labels = load_cicids_data(
                data_path=args.data_path,
                dataset=args.dataset,
                max_samples_per_file=args.max_samples_per_file
            )
        except Exception as e:
            print(f"Error loading PCAP data: {e}")
            print("Falling back to dummy data for testing\n")
            data = np.random.randint(0, 256, (1000, 3, 224, 224), dtype=np.uint8)
            labels = np.random.randint(0, config.NUM_CLASSES, 1000)
    else:
        print("No --data_path specified. Using dummy data for demonstration")
        print("To use real PCAP data, run with: --data_path /path/to/pcap/directory\n")
        data = np.random.randint(0, 256, (1000, 3, 224, 224), dtype=np.uint8)
        labels = np.random.randint(0, config.NUM_CLASSES, 1000)
    
    train_data, train_labels, test_data, test_labels = split_train_test(
        data, labels, test_ratio=0.2
    )
    
    if args.scenario == 'iid':
        print(f"=== IID Scenario with {args.num_clients} clients ===")
        print("Using imbalanced class distribution as per Table 1 in paper\n")
        client_datasets = create_imbalanced_iid_partition(
            train_data, train_labels, args.num_clients
        )
    else:
        print(f"=== Non-IID Scenario with {args.num_clients} clients ===")
        print(f"Using class distribution from Table 2 in paper\n")
        client_class_distribution = config.get_non_iid_distribution(args.dataset, args.num_clients)
        client_datasets = create_non_iid_partition(
            train_data, train_labels, args.num_clients, client_class_distribution
        )
    
    print_data_distribution(client_datasets, config.NUM_CLASSES)
    
    clients = []
    for i, (client_data, client_labels) in enumerate(client_datasets):
        client = Client(
            client_id=i,
            data=client_data,
            labels=client_labels,
            num_classes=config.NUM_CLASSES,
            rank=config.LORA_RANK,
            alpha=config.LORA_ALPHA,
            device=device
        )
        clients.append(client)
    
    total_params, trainable_params = count_parameters(clients[0].model)
    print(f"\n{'='*80}")
    print(f"Model Architecture: ViT-tiny with LoRA")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameter reduction: {total_params/trainable_params:.2f}x")
    print(f"  Target reduction (paper): ~64x")
    print(f"{'='*80}\n")
    
    server = FederatedServer(
        num_classes=config.NUM_CLASSES,
        rank=config.LORA_RANK,
        alpha=config.LORA_ALPHA,
        aggregation_method=args.aggregation,
        lambda_reg=config.LAMBDA_REG,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    print(f"{'='*80}")
    print(f"Starting Federated Training")
    print(f"{'='*80}")
    print(f"Training Configuration:")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Local epochs: {args.local_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {config.WEIGHT_DECAY}")
    if args.aggregation == 'rgpa':
        print(f"  Aggregation: RGPA (λ={config.LAMBDA_REG})")
    else:
        print(f"  Aggregation: FedAvg")
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    print(f"{'='*80}\n")
    
    history = server.train_federated(
        clients=clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        test_data=test_data,
        test_labels=test_labels,
        early_stopping_patience=5,
        save_every=5
    )
    
    print(f"\n{'='*80}")
    print(f"Training Complete")
    print(f"{'='*80}")
    print(f"Best Performance:")
    print(f"  Accuracy: {server.best_accuracy:.4f} ({server.best_accuracy*100:.2f}%)")
    print(f"  F1 Score: {server.best_f1:.4f} ({server.best_f1*100:.2f}%)")
    print(f"\nFinal Round Performance:")
    print(f"  Accuracy: {history['accuracy'][-1]:.4f} ({history['accuracy'][-1]*100:.2f}%)")
    print(f"  Precision: {history['precision'][-1]:.4f} ({history['precision'][-1]*100:.2f}%)")
    print(f"  Recall: {history['recall'][-1]:.4f} ({history['recall'][-1]*100:.2f}%)")
    print(f"  F1 Score: {history['f1'][-1]:.4f} ({history['f1'][-1]*100:.2f}%)")
    print(f"\nPaper Baseline (IID): Accuracy=97.26%, F1=96.99%")
    print(f"Paper Baseline (Non-IID): Accuracy=96.17%, F1=95.81%")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
