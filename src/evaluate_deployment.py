#!/usr/bin/env python3
import torch
import numpy as np
import argparse
import os
from models import FLoV2TModel
from deployment_monitor import DeploymentMonitor
from config import FLoV2TConfig

def load_model(checkpoint_path: str, device: str) -> FLoV2TModel:
    """Load trained FLoV2T model from checkpoint"""
    config = FLoV2TConfig()
    
    model = FLoV2TModel(
        num_classes=config.NUM_CLASSES,
        rank=config.LORA_RANK,
        alpha=config.LORA_ALPHA
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    if 'round' in checkpoint:
        print(f"  Training round: {checkpoint['round']}")
    if 'best_accuracy' in checkpoint:
        print(f"  Best accuracy: {checkpoint['best_accuracy']*100:.2f}%")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Evaluate FLoV2T deployment on IoT devices')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--test_data', type=str, required=True, 
                       help='Path to test data directory or .npy file')
    parser.add_argument('--test_labels', type=str, default=None, 
                       help='Path to test labels .npy file (auto-detected if not provided)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default=None, 
                       help='Device to use (cuda/cpu, auto-detected if not provided)')
    parser.add_argument('--log_dir', type=str, default='./deployment_logs', 
                       help='Directory to save logs')
    parser.add_argument('--monitor_interval', type=float, default=1.0, 
                       help='Background monitoring interval in seconds')
    
    args = parser.parse_args()
    
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\n{'='*80}")
    print(f"FLoV2T Deployment Evaluation")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Test data: {args.test_data}")
    print(f"Batch size: {args.batch_size}")
    print(f"Log directory: {args.log_dir}")
    print(f"{'='*80}\n")
    
    print("Loading model...")
    model = load_model(args.model, device)
    
    print("\nLoading test data...")
    if os.path.isdir(args.test_data):
        test_files = [f for f in os.listdir(args.test_data) if f.endswith('_test_data.npy')]
        if not test_files:
            raise ValueError(f"No test data files found in {args.test_data}")
        
        test_data_path = os.path.join(args.test_data, test_files[0])
        
        if args.test_labels:
            test_labels_path = args.test_labels
        else:
            test_labels_path = test_data_path.replace('_test_data.npy', '_test_labels.npy')
    else:
        test_data_path = args.test_data
        if args.test_labels:
            test_labels_path = args.test_labels
        else:
            test_labels_path = test_data_path.replace('_data.npy', '_labels.npy')
    
    test_data = np.load(test_data_path)
    test_labels = np.load(test_labels_path)
    
    print(f"Test data loaded: {test_data.shape}")
    print(f"Test labels loaded: {test_labels.shape}")
    
    print("\nInitializing deployment monitor...")
    monitor = DeploymentMonitor(model, device=device, log_dir=args.log_dir)
    
    print("\nStarting background monitoring...")
    monitor.start_background_monitoring(interval=args.monitor_interval)
    
    print(f"\n{'='*80}")
    print("Running Deployment Evaluation")
    print(f"{'='*80}\n")
    
    print("1. Measuring Inference Latency...")
    print("-" * 80)
    latency_metrics = monitor.measure_inference_latency(
        test_data, 
        batch_size=args.batch_size
    )
    
    print(f"\n2. Evaluating Accuracy Metrics...")
    print("-" * 80)
    accuracy_metrics = monitor.evaluate_accuracy(
        test_data, 
        test_labels, 
        batch_size=args.batch_size
    )
    
    print(f"\n3. Stopping background monitoring...")
    print("-" * 80)
    monitor.stop_background_monitoring()
    
    print(f"\n4. Generating deployment report...")
    print("-" * 80)
    report = monitor.generate_report()
    
    print(f"\n5. Generating confusion matrix...")
    print("-" * 80)
    monitor.plot_confusion_matrix()
    
    print(f"\n{'='*80}")
    print("Deployment Evaluation Complete")
    print(f"{'='*80}")
    print(f"\nResults saved to: {args.log_dir}")
    print(f"\nKey Metrics:")
    print(f"  Accuracy: {accuracy_metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score: {accuracy_metrics['f1_score']*100:.2f}%")
    print(f"  Avg Latency: {latency_metrics['avg_latency_ms']:.2f} ms")
    print(f"  P99 Latency: {latency_metrics['p99_latency_ms']:.2f} ms")
    print(f"  Throughput: {latency_metrics['throughput_samples_per_sec']:.2f} samples/sec")
    
    summary = report['summary']
    if 'memory' in summary:
        print(f"  Peak RAM: {summary['memory']['peak_ram_gb']:.2f} GB")
    if 'computation' in summary:
        print(f"  Avg CPU: {summary['computation']['avg_cpu_percent']:.1f}%")
    
    print(f"{'='*80}\n")
    
    print("IoT Device Suitability Assessment:")
    print("-" * 80)
    assess_iot_suitability(summary, latency_metrics, accuracy_metrics)
    print("-" * 80)

def assess_iot_suitability(summary: dict, latency: dict, accuracy: dict):
    """Assess if the model is suitable for IoT deployment"""
    
    score = 0
    max_score = 5
    
    if 'memory' in summary:
        ram_gb = summary['memory']['peak_ram_gb']
        if ram_gb < 2:
            print(f"[PASS] Memory: {ram_gb:.2f} GB - EXCELLENT (works on edge devices)")
            score += 1
        elif ram_gb < 4:
            print(f"[PASS] Memory: {ram_gb:.2f} GB - GOOD (works on standard IoT)")
            score += 1
        else:
            print(f"[WARN] Memory: {ram_gb:.2f} GB - HIGH (needs powerful device)")
    
    avg_latency = latency['avg_latency_ms']
    if avg_latency < 100:
        print(f"[PASS] Latency: {avg_latency:.2f} ms - EXCELLENT (real-time capable)")
        score += 1
    elif avg_latency < 500:
        print(f"[PASS] Latency: {avg_latency:.2f} ms - GOOD (near real-time)")
        score += 1
    else:
        print(f"[WARN] Latency: {avg_latency:.2f} ms - SLOW (optimization needed)")
    
    p99_latency = latency['p99_latency_ms']
    if p99_latency < 200:
        print(f"[PASS] P99 Latency: {p99_latency:.2f} ms - EXCELLENT (consistent)")
        score += 1
    elif p99_latency < 1000:
        print(f"[PASS] P99 Latency: {p99_latency:.2f} ms - ACCEPTABLE")
    else:
        print(f"[WARN] P99 Latency: {p99_latency:.2f} ms - INCONSISTENT")
    
    acc = accuracy['accuracy']
    if acc > 0.95:
        print(f"[PASS] Accuracy: {acc*100:.2f}% - EXCELLENT (production ready)")
        score += 1
    elif acc > 0.90:
        print(f"[PASS] Accuracy: {acc*100:.2f}% - GOOD (acceptable)")
        score += 1
    else:
        print(f"[WARN] Accuracy: {acc*100:.2f}% - LOW (more training needed)")
    
    throughput = latency['throughput_samples_per_sec']
    if throughput > 10:
        print(f"[PASS] Throughput: {throughput:.2f} samples/sec - HIGH")
        score += 1
    elif throughput > 5:
        print(f"[PASS] Throughput: {throughput:.2f} samples/sec - MODERATE")
    else:
        print(f"[WARN] Throughput: {throughput:.2f} samples/sec - LOW")
    
    print(f"\nOverall Score: {score}/{max_score}")
    
    if score >= 4:
        print("[PASS] HIGHLY SUITABLE for IoT/Edge deployment")
    elif score >= 3:
        print("[PASS] SUITABLE for IoT deployment")
    elif score >= 2:
        print("[WARN] MARGINAL - optimization recommended")
    else:
        print("[FAIL] NOT SUITABLE - significant improvements needed")

if __name__ == "__main__":
    main()
