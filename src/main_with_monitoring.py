import torch
import numpy as np
from data_preprocessing import TrafficPreprocessor
from models import FLoV2TModel
from federated_trainer import Client, FederatedServer
from config import FLoV2TConfig
from utils import set_seed, create_iid_partition, split_train_test
from benchmark import PerformanceMonitor, ModelProfiler

def main_with_monitoring():
    set_seed(42)
    
    config = FLoV2TConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("FLoV2T: Federated Learning for AIoT Malicious Traffic Classification")
    print("="*80 + "\n")
    
    monitor = PerformanceMonitor(log_file='flov2t_performance.json')
    monitor.start_monitoring()
    
    print("=== PHASE 1: System Information ===")
    snapshot = monitor.record_snapshot(phase='system_init')
    print(f"CPU Cores: {snapshot['cpu_percent']}")
    print(f"Total RAM: {snapshot['memory']['total_gb']:.2f} GB")
    if snapshot['gpu']:
        if 'gpu_name' in snapshot['gpu']:
            print(f"GPU: {snapshot['gpu']['gpu_name']}")
        if 'gpu_memory_total_mb' in snapshot['gpu']:
            print(f"GPU Memory: {snapshot['gpu']['gpu_memory_total_mb']:.0f} MB")
    
    print("\n=== PHASE 2: Model Profiling ===")
    model = FLoV2TModel(num_classes=config.NUM_CLASSES, rank=config.LORA_RANK, alpha=config.LORA_ALPHA)
    model_profile = ModelProfiler.profile_model(model, device=device)
    ModelProfiler.print_profile(model_profile)
    
    snapshot = monitor.record_snapshot(
        phase='model_loaded', 
        additional_info={'model_profile': model_profile}
    )
    
    print("\n=== PHASE 3: Data Loading ===")
    print("Loading dummy data (replace with actual PCAP preprocessing)...")
    
    dummy_data = np.random.randint(0, 256, (1000, 256, 14, 14), dtype=np.uint8)
    dummy_labels = np.random.randint(0, config.NUM_CLASSES, 1000)
    
    train_data, train_labels, test_data, test_labels = split_train_test(
        dummy_data, dummy_labels, test_ratio=0.2
    )
    
    snapshot = monitor.record_snapshot(phase='data_loaded')
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Memory after data load: {snapshot['memory']['used_gb']:.2f} GB")
    
    print("\n=== PHASE 4: Client Setup ===")
    num_clients = 3
    client_datasets = create_iid_partition(train_data, train_labels, num_clients)
    
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
        print(f"Client {i}: {len(client_data)} samples")
    
    snapshot = monitor.record_snapshot(phase='clients_created')
    
    print("\n=== PHASE 5: Federated Training ===")
    server = FederatedServer(
        num_classes=config.NUM_CLASSES,
        rank=config.LORA_RANK,
        alpha=config.LORA_ALPHA,
        aggregation_method='rgpa',
        lambda_reg=config.LAMBDA_REG,
        device=device
    )
    
    print(f"Rounds: {config.NUM_ROUNDS}")
    print(f"Local epochs: {config.LOCAL_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    
    for round_num in range(config.NUM_ROUNDS):
        print(f"\n--- Round {round_num + 1}/{config.NUM_ROUNDS} ---")
        
        round_start = monitor.record_snapshot(
            phase='round_start',
            epoch=round_num
        )
        
        client_states = []
        client_data_sizes = []
        
        for client_idx, client in enumerate(clients):
            global_state = server.get_global_state()
            client.set_global_model(global_state)
            
            client_state = client.train(
                epochs=config.LOCAL_EPOCHS,
                batch_size=config.BATCH_SIZE,
                lr=config.LEARNING_RATE
            )
            
            client_states.append(client_state)
            client_data_sizes.append(client.data_size)
            
            client_snapshot = monitor.record_snapshot(
                phase='client_training',
                epoch=round_num,
                additional_info={'client_id': client_idx}
            )
        
        server.aggregate_client_updates(client_states, client_data_sizes)
        
        global_state = server.get_global_state()
        clients[0].set_global_model(global_state)
        acc, prec, rec, f1 = clients[0].evaluate(test_data, test_labels, batch_size=config.BATCH_SIZE)
        
        round_end = monitor.record_snapshot(
            phase='round_end',
            epoch=round_num,
            additional_info={
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            }
        )
        
        print(f"Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(f"CPU: {round_end['cpu_percent']:.1f}%, RAM: {round_end['memory']['used_gb']:.2f} GB", end='')
        if round_end['gpu'] and 'gpu_memory_allocated_gb' in round_end['gpu']:
            print(f", GPU Mem: {round_end['gpu']['gpu_memory_allocated_gb']:.2f} GB")
        else:
            print()
    
    print("\n=== PHASE 6: Final Evaluation ===")
    final_snapshot = monitor.record_snapshot(phase='training_complete')
    
    monitor.save_log()
    monitor.print_summary()
    
    print("\n=== Lightweight Device Suitability Analysis ===")
    analyze_lightweight_suitability(model_profile, monitor.metrics)
    
    print("\nPerformance logs saved to: flov2t_performance.json")
    print("Model profile saved to: model_profile.json")

def analyze_lightweight_suitability(model_profile, metrics):
    print("\nDevice Requirements Analysis:")
    print("-" * 60)
    
    trainable_params_k = model_profile['parameters']['trainable_params_K']
    total_params_m = model_profile['parameters']['total_params_M']
    reduction = model_profile['parameters']['parameter_reduction']
    
    print(f"\n✓ Parameter Efficiency:")
    print(f"  Trainable: {trainable_params_k:.2f}K (vs {total_params_m:.2f}M full model)")
    print(f"  Reduction: {reduction:.2f}x")
    if trainable_params_k < 500:
        print(f"  Status: ✓ EXCELLENT - Suitable for edge devices")
    elif trainable_params_k < 1000:
        print(f"  Status: ✓ GOOD - Suitable for most IoT devices")
    else:
        print(f"  Status: ⚠ MODERATE - May need optimization")
    
    model_size_mb = model_profile['model_size']['total_size_mb']
    print(f"\n✓ Model Size:")
    print(f"  Total: {model_size_mb:.2f} MB")
    if model_size_mb < 100:
        print(f"  Status: ✓ EXCELLENT - Fits in constrained memory")
    elif model_size_mb < 500:
        print(f"  Status: ✓ GOOD - Acceptable for edge deployment")
    else:
        print(f"  Status: ⚠ LARGE - May require compression")
    
    avg_inference_ms = model_profile['inference']['avg_inference_time_ms']
    throughput = model_profile['inference']['throughput_samples_per_sec']
    print(f"\n✓ Inference Performance:")
    print(f"  Latency: {avg_inference_ms:.2f} ms/sample")
    print(f"  Throughput: {throughput:.2f} samples/sec")
    if avg_inference_ms < 100:
        print(f"  Status: ✓ EXCELLENT - Real-time capable")
    elif avg_inference_ms < 500:
        print(f"  Status: ✓ GOOD - Near real-time")
    else:
        print(f"  Status: ⚠ SLOW - May need optimization")
    
    if metrics:
        peak_ram_gb = max([m['memory']['used_gb'] for m in metrics])
        print(f"\n✓ Memory Usage:")
        print(f"  Peak RAM: {peak_ram_gb:.2f} GB")
        if peak_ram_gb < 2:
            print(f"  Status: ✓ EXCELLENT - Runs on low-end devices")
        elif peak_ram_gb < 4:
            print(f"  Status: ✓ GOOD - Standard IoT device capable")
        elif peak_ram_gb < 8:
            print(f"  Status: ⚠ MODERATE - Needs mid-range device")
        else:
            print(f"  Status: ⚠ HIGH - Requires powerful device")
        
        gpu_metrics = [m['gpu'] for m in metrics if m['gpu'] and 'gpu_memory_allocated_gb' in m['gpu']]
        if gpu_metrics:
            peak_gpu_gb = max([g['gpu_memory_allocated_gb'] for g in gpu_metrics])
            print(f"\n✓ GPU Memory:")
            print(f"  Peak: {peak_gpu_gb:.2f} GB")
            if peak_gpu_gb < 2:
                print(f"  Status: ✓ EXCELLENT - Works on integrated GPUs")
            elif peak_gpu_gb < 4:
                print(f"  Status: ✓ GOOD - Consumer GPU capable")
            else:
                print(f"  Status: ⚠ MODERATE - Needs dedicated GPU")
    
    avg_training_ms = model_profile['training']['avg_training_time_ms']
    print(f"\n✓ Training Performance:")
    print(f"  Time per batch: {avg_training_ms:.2f} ms")
    if avg_training_ms < 500:
        print(f"  Status: ✓ EXCELLENT - Fast training")
    elif avg_training_ms < 1000:
        print(f"  Status: ✓ GOOD - Acceptable training speed")
    else:
        print(f"  Status: ⚠ SLOW - Consider GPU acceleration")
    
    print("\n" + "="*60)
    print("OVERALL ASSESSMENT:")
    
    score = 0
    if trainable_params_k < 500: score += 1
    if model_size_mb < 100: score += 1
    if avg_inference_ms < 100: score += 1
    if metrics and peak_ram_gb < 4: score += 1
    if avg_training_ms < 500: score += 1
    
    if score >= 4:
        print("✓ HIGHLY SUITABLE for lightweight AIoT edge devices")
    elif score >= 3:
        print("✓ SUITABLE for most AIoT edge devices")
    elif score >= 2:
        print("⚠ MODERATELY SUITABLE - optimization recommended")
    else:
        print("⚠ NOT IDEAL for resource-constrained devices")
    print("="*60)

if __name__ == "__main__":
    main_with_monitoring()
