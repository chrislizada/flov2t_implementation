# FLoV2T Deployment and Evaluation Guide

Complete guide for creating test sets, training models, and evaluating deployment metrics on IoT devices.

## Overview

This implementation includes comprehensive monitoring for real-world IoT deployment:

- **Memory Monitoring**: Peak RAM, swap, GPU memory
- **Computation Metrics**: CPU utilization, inference time per sample
- **Energy Consumption**: Power monitoring (if hardware supports)
- **Communication Overhead**: Bytes transmitted per federated round
- **Latency Measurements**: End-to-end detection latency, P50/P95/P99
- **Accuracy Metrics**: Precision, Recall, F1-score, Confusion Matrix

## Quick Start

### Step 1: Create Test Sets (10% for final evaluation)

```bash
# Create test sets for CICIDS2017
python src/create_test_set.py --data_path ./data/CICIDS2017 --dataset CICIDS2017

# Create test sets for CICIDS2018
python src/create_test_set.py --data_path ./data/CICIDS2018 --dataset CICIDS2018
```

This creates a **separate 10% test set** for final deployment evaluation, distinct from the 80-20 train/test split used during training.

**Output:**
```
./test_data/
├── CICIDS2017_test_data.npy
├── CICIDS2017_test_labels.npy
├── CICIDS2017_test_metadata.json
├── CICIDS2018_test_data.npy
├── CICIDS2018_test_labels.npy
└── CICIDS2018_test_metadata.json
```

### Step 2: Train the Model

```bash
# Train on CICIDS2017 (IID scenario, 3 clients)
python src/main.py \
    --data_path ./data/CICIDS2017 \
    --dataset CICIDS2017 \
    --num_clients 3 \
    --scenario iid \
    --checkpoint_dir ./checkpoints/cicids2017_iid

# Train on CICIDS2018 (IID scenario, 3 clients)
python src/main.py \
    --data_path ./data/CICIDS2018 \
    --dataset CICIDS2018 \
    --num_clients 3 \
    --scenario iid \
    --checkpoint_dir ./checkpoints/cicids2018_iid
```

**Output:**
```
./checkpoints/
├── cicids2017_iid/
│   ├── best_model.pt
│   ├── checkpoint_round_5.pt
│   ├── checkpoint_round_10.pt
│   └── ...
└── cicids2018_iid/
    └── ...
```

### Step 3: Evaluate Deployment

```bash
# Evaluate CICIDS2017 model
python src/evaluate_deployment.py \
    --model ./checkpoints/cicids2017_iid/best_model.pt \
    --test_data ./test_data/CICIDS2017_test_data.npy

# Evaluate CICIDS2018 model
python src/evaluate_deployment.py \
    --model ./checkpoints/cicids2018_iid/best_model.pt \
    --test_data ./test_data/CICIDS2018_test_data.npy
```

**Output:**
```
./deployment_logs/
├── deployment_report_20250118_143022.json
└── confusion_matrix_20250118_143022.png
```

## Detailed Usage

### Creating Test Sets

The test set creation script performs **stratified sampling** to maintain class distribution:

```bash
python src/create_test_set.py \
    --data_path ./data/CICIDS2017 \
    --dataset CICIDS2017 \
    --output_dir ./test_data \
    --test_ratio 0.1 \
    --seed 42
```

**Options:**
- `--data_path`: Path to PCAP files
- `--dataset`: CICIDS2017 or CICIDS2018
- `--output_dir`: Where to save test set (default: ./test_data)
- `--test_ratio`: Percentage for testing (default: 0.1 = 10%)
- `--seed`: Random seed for reproducibility

**Why separate test set?**
- Training uses 80% train / 20% validation split
- This 10% test set is for **final deployment evaluation only**
- Prevents data leakage and overfitting
- Simulates real-world unseen data

### Training Models

#### IID Scenario (3 clients)
```bash
python src/main.py \
    --data_path ./data/CICIDS2017 \
    --dataset CICIDS2017 \
    --num_clients 3 \
    --scenario iid \
    --num_rounds 20 \
    --local_epochs 5 \
    --batch_size 32 \
    --lr 0.001 \
    --aggregation rgpa \
    --checkpoint_dir ./checkpoints/cicids2017_iid
```

#### Non-IID Scenario (5 clients)
```bash
python src/main.py \
    --data_path ./data/CICIDS2017 \
    --dataset CICIDS2017 \
    --num_clients 5 \
    --scenario non-iid \
    --num_rounds 20 \
    --aggregation rgpa \
    --checkpoint_dir ./checkpoints/cicids2017_noniid
```

**Key Parameters:**
- `--num_clients`: 3 or 5 (as per paper)
- `--scenario`: iid or non-iid
- `--aggregation`: rgpa (FLoV2T) or fedavg (baseline)
- `--checkpoint_dir`: Where to save model checkpoints

### Deployment Evaluation

Full deployment monitoring with all metrics:

```bash
python src/evaluate_deployment.py \
    --model ./checkpoints/cicids2017_iid/best_model.pt \
    --test_data ./test_data/CICIDS2017_test_data.npy \
    --test_labels ./test_data/CICIDS2017_test_labels.npy \
    --batch_size 32 \
    --device cuda \
    --log_dir ./deployment_logs \
    --monitor_interval 1.0
```

**Options:**
- `--model`: Path to trained checkpoint
- `--test_data`: Path to test data (.npy file or directory)
- `--test_labels`: Path to test labels (auto-detected if not provided)
- `--batch_size`: Batch size for inference (default: 32)
- `--device`: cuda or cpu (auto-detected if not provided)
- `--log_dir`: Where to save logs (default: ./deployment_logs)
- `--monitor_interval`: Background monitoring interval in seconds

## Metrics Collected

### 1. Memory Metrics
- **Peak RAM Usage**: Maximum RAM consumed during inference
- **Average RAM Usage**: Mean RAM over time
- **Swap Usage**: Swap memory utilization
- **GPU Memory** (if available): Allocated, reserved, peak

### 2. Computation Metrics
- **CPU Utilization**: Average and peak CPU percentage
- **CPU Frequency**: Current processor frequency
- **GPU Utilization** (if available): GPU load percentage
- **GPU Temperature**: Thermal monitoring

### 3. Latency Metrics
- **Average Latency**: Mean inference time per sample
- **Standard Deviation**: Latency variance
- **P50/P95/P99**: Percentile latencies for SLA monitoring
- **Throughput**: Samples processed per second

### 4. Accuracy Metrics
- **Overall Accuracy**: Correct predictions / total predictions
- **Precision**: Per-class and weighted average
- **Recall**: Per-class and weighted average
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation saved as PNG

### 5. Communication Metrics
- **Model Size**: Bytes transmitted per federated round
- **Total Sent/Received**: Network I/O during evaluation
- **Federated Overhead**: Communication cost of LoRA vs full model

### 6. Energy Metrics (if pyJoules installed)
- **Package Power**: CPU package power consumption
- **DRAM Power**: Memory power consumption
- **Total Energy**: Joules consumed during evaluation

## Expected Results

### Paper Baseline (CICIDS2017, IID)
- Accuracy: 97.26%
- F1 Score: 96.99%
- Inference Time: ~50-100ms
- Peak RAM: ~4.5 GB
- GPU Memory: ~2.09 GB

### Paper Baseline (CICIDS2017, Non-IID)
- Accuracy: 96.17%
- F1 Score: 95.81%

### IoT Suitability Criteria
| Metric | Excellent | Good | Marginal |
|--------|-----------|------|----------|
| Peak RAM | < 2 GB | < 4 GB | < 8 GB |
| Latency | < 100 ms | < 500 ms | < 1000 ms |
| P99 Latency | < 200 ms | < 1000 ms | > 1000 ms |
| Accuracy | > 95% | > 90% | > 85% |
| Throughput | > 10 samples/s | > 5 samples/s | > 1 samples/s |

## Output Files

### Deployment Report (JSON)
```json
{
  "timestamp": "2025-01-18T14:30:22",
  "duration_sec": 45.67,
  "device": "cuda",
  "metrics": {
    "memory": [...],
    "computation": [...],
    "latency": [...],
    "accuracy": [...],
    "communication": [...]
  },
  "summary": {
    "memory": {
      "peak_ram_gb": 3.2,
      "avg_ram_mb": 2800
    },
    "latency": {
      "avg_ms": 45.2,
      "p95_ms": 67.8,
      "p99_ms": 89.1
    },
    "accuracy": {
      "accuracy": 0.9726,
      "f1_score": 0.9699
    }
  }
}
```

### Confusion Matrix (PNG)
Visual heatmap showing prediction accuracy per attack class.

## Example Workflow

### Complete Training and Evaluation Pipeline

```bash
# 1. Create test sets (10% of data)
python src/create_test_set.py --data_path ./data/CICIDS2017 --dataset CICIDS2017

# 2. Train on CICIDS2017 (IID scenario)
python src/main.py \
    --data_path ./data/CICIDS2017 \
    --dataset CICIDS2017 \
    --num_clients 3 \
    --scenario iid \
    --checkpoint_dir ./checkpoints/cicids2017_iid

# 3. Evaluate deployment on test set
python src/evaluate_deployment.py \
    --model ./checkpoints/cicids2017_iid/best_model.pt \
    --test_data ./test_data/CICIDS2017_test_data.npy

# 4. Check results
cat ./deployment_logs/deployment_report_*.json
```

### Custom Training Configuration

```bash
# Train with more rounds for better convergence
python src/main.py \
    --data_path ./data/CICIDS2017 \
    --dataset CICIDS2017 \
    --num_clients 3 \
    --scenario iid \
    --num_rounds 50 \
    --local_epochs 10 \
    --batch_size 64 \
    --lr 0.0005 \
    --aggregation rgpa \
    --checkpoint_dir ./checkpoints/cicids2017_extended
```

### Limited Data Testing

```bash
# Quick test with limited samples
python src/main.py \
    --data_path ./data/CICIDS2017 \
    --dataset CICIDS2017 \
    --max_samples_per_file 100 \
    --num_rounds 5 \
    --checkpoint_dir ./checkpoints/test
```

## IoT Device Testing

### On Edge Devices (Raspberry Pi, Jetson Nano)

```bash
# CPU-only inference with smaller batch size
python src/evaluate_deployment.py \
    --model ./best_model.pt \
    --test_data ./test_data/CICIDS2017_test_data.npy \
    --batch_size 8 \
    --device cpu
```

### On GPU Devices (Jetson Xavier, EC2 g4dn)

```bash
# GPU inference with larger batch size
python src/evaluate_deployment.py \
    --model ./best_model.pt \
    --test_data ./test_data/CICIDS2017_test_data.npy \
    --batch_size 64 \
    --device cuda
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 8

# Or use CPU
--device cpu
```

### Slow Inference
```bash
# Use GPU if available
--device cuda

# Increase batch size
--batch_size 64
```

### Missing Test Data
```bash
# Recreate test set
./scripts/create_test_sets.sh
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{zeng2025flov2t,
  title={FLoV2T: A fine-grained malicious traffic classification method based on federated learning for AIoT},
  author={Zeng, Fanyi and Xu, Chen and Man, Dapeng and Jiang, Junhui and Yang, Wu},
  journal={Computer Communications},
  volume={242},
  pages={108288},
  year={2025},
  publisher={Elsevier}
}
```
