# FLoV2T Implementation

**FLoV2T: A fine-grained malicious traffic classification method based on federated learning for AIoT**

This is a complete implementation of the FLoV2T system described in the paper by Zeng et al. (2025).

## Overview

FLoV2T addresses three key challenges in AIoT security:
1. **Suboptimal Classification Accuracy** - Uses Vision Transformer (ViT) with raw traffic visualization
2. **Limited Terminal Resources** - Implements LoRA for efficient fine-tuning (64x parameter reduction)
3. **Non-IID Data Distribution** - Employs Regularized Global Parameter Aggregation (RGPA)

## Architecture

The system consists of three core components:

### 1. Raw Traffic Feature Extraction (RTFE)
- **Pcap2Flow**: Divides network traffic into flows
- **Packet2Patch**: Converts packets to 16×16 patches (20B network header + 20B transport header + 216B payload)
- **Flow2Image**: Creates 224×224 images from 196 patches
- **ViT Model**: Pre-trained Vision Transformer for feature extraction

### 2. Local Fine-tuning with LoRA (LFL)
- Low-Rank Adaptation with rank=4, alpha=8
- Freezes pre-trained ViT weights
- Only trains low-rank matrices A and B
- Reduces trainable parameters from 21.67M to 336.8K

### 3. Regularized Global Parameter Aggregation (RGPA)
- Weighted averaging based on client data sizes
- Regularization term: λ=0.1
- Mitigates bias from non-IID data distributions

## Project Structure

```
flov2t_implementation/
├── src/
│   ├── data_preprocessing.py      # Pcap2Flow, Packet2Patch, Flow2Image
│   ├── models.py                  # ViT-tiny with LoRA implementation
│   ├── federated_aggregation.py   # RGPA and FedAvg aggregators
│   ├── federated_trainer.py       # Client and Server training logic
│   ├── load_cicids_data.py        # CICIDS2017/2018 PCAP data loader
│   ├── create_test_set.py         # Create 10% test set for evaluation
│   ├── deployment_monitor.py      # IoT deployment monitoring
│   ├── evaluate_deployment.py     # Deployment evaluation script
│   ├── config.py                  # Configuration parameters
│   ├── utils.py                   # Helper functions
│   ├── main.py                    # Main training script
│   └── test_*.py                  # Unit tests
├── requirements.txt
├── README.md
├── PCAP_USAGE.md                  # Guide for using PCAP data
├── DEPLOYMENT_GUIDE.md            # Deployment and evaluation guide
└── BENCHMARK.md                   # Performance benchmarking guide
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Training with CICIDS2017/2018 PCAP Data

```bash
# Train on CICIDS2017 (IID scenario, 3 clients)
python src/main.py \
    --data_path ./data/CICIDS2017 \
    --dataset CICIDS2017 \
    --num_clients 3 \
    --scenario iid \
    --num_rounds 20 \
    --checkpoint_dir ./checkpoints/cicids2017_iid

# Train on CICIDS2018 (Non-IID scenario, 5 clients)
python src/main.py \
    --data_path ./data/CICIDS2018 \
    --dataset CICIDS2018 \
    --num_clients 5 \
    --scenario non-iid \
    --num_rounds 20 \
    --checkpoint_dir ./checkpoints/cicids2018_noniid
```

### 2. Create Test Set (10% for final evaluation)

```bash
# Create separate test set for deployment evaluation
python src/create_test_set.py \
    --data_path ./data/CICIDS2017 \
    --dataset CICIDS2017 \
    --output_dir ./test_data \
    --test_ratio 0.1
```

### 3. Evaluate Deployment on IoT Devices

```bash
# Comprehensive deployment evaluation
python src/evaluate_deployment.py \
    --model ./checkpoints/cicids2017_iid/best_model.pt \
    --test_data ./test_data/CICIDS2017_test_data.npy \
    --batch_size 32 \
    --log_dir ./deployment_logs
```

This measures:
- Memory: Peak RAM, swap, GPU memory
- Computation: CPU utilization, inference time
- Latency: End-to-end detection time (P50/P95/P99)
- Accuracy: Precision, Recall, F1-score
- Communication: Bytes transmitted per round

## Detailed Guides

- **[PCAP_USAGE.md](PCAP_USAGE.md)** - How to use CICIDS2017/2018 PCAP files
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Complete deployment and evaluation guide
- **[BENCHMARK.md](BENCHMARK.md)** - Performance benchmarking guide

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_PACKETS` | 196 | Maximum packets per flow |
| `PATCH_SIZE` | 16 | Patch dimension (16×16) |
| `NUM_CLASSES` | 8 | Number of attack categories |
| `LORA_RANK` | 4 | LoRA rank |
| `LORA_ALPHA` | 8 | LoRA alpha scaling |
| `LAMBDA_REG` | 0.1 | RGPA regularization coefficient |
| `NUM_ROUNDS` | 20 | Federated training rounds |
| `LOCAL_EPOCHS` | 5 | Local training epochs per round |
| `BATCH_SIZE` | 32 | Training batch size |
| `LEARNING_RATE` | 0.001 | AdamW learning rate |

## Attack Categories

The system classifies 8 types of malicious traffic:

0. Botnet
1. DoS-Slowloris
2. DoS-Goldeneye
3. DoS-Hulk
4. SSH-BruteForce
5. Web-SQL
6. Web-XSS
7. Web-Bruteforce

## Results (from Paper)

### IID Scenario
- **Accuracy**: 97.26%
- **F1 Score**: 96.99%
- **Improvement over baseline**: +10.94% accuracy, +11.47% F1

### Non-IID Scenario
- **Accuracy**: 96.17%
- **F1 Score**: 95.81%
- **Parameter reduction**: ~64× (21.67M → 336.8K)

## Citation

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

## License

This implementation is for research and educational purposes.

## Key Features

- **Automatic PCAP Processing**: Automatically detects attack types from PCAP filenames
- **CICIDS Support**: Built-in support for CICIDS2017 and CICIDS2018 datasets
- **ViT-tiny Model**: Uses lightweight ViT-tiny (192 hidden size) for edge devices
- **LoRA Fine-tuning**: 64x parameter reduction (21.67M → 336.8K trainable)
- **RGPA Aggregation**: Handles non-IID data distributions
- **Comprehensive Monitoring**: Memory, CPU, GPU, latency, accuracy tracking
- **IoT Deployment Ready**: Optimized for resource-constrained devices

## PCAP File Naming

Place PCAP files in `./data/CICIDS2017/` or `./data/CICIDS2018/` with keywords:

| Attack Type | Keywords | Label |
|-------------|----------|-------|
| Botnet | `botnet`, `bot` | 0 |
| DoS-Slowloris | `slowloris`, `slow` | 1 |
| DoS-GoldenEye | `goldeneye`, `golden` | 2 |
| DoS-Hulk | `hulk` | 3 |
| SSH-BruteForce | `ssh`, `patator`, `ftp` | 4 |
| Web-SQL | `sql` | 5 |
| Web-XSS | `xss` | 6 |
| Web-Bruteforce | `bruteforce`, `brute` | 7 |

Example: `botnet_capture.pcap`, `dos_slowloris_attack.pcap`

## Expected Performance

| Metric | Target | Description |
|--------|--------|-------------|
| Accuracy | 97.26% | IID scenario (paper baseline) |
| F1 Score | 96.99% | IID scenario (paper baseline) |
| Parameters | 336.8K | Trainable (64x reduction) |
| Inference | <100ms | Real-time capable |
| Memory | <4GB | Edge device compatible |
| Model Size | ~83MB | Lightweight for IoT |

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Scapy (for PCAP processing)
- CICIDS2017/2018 datasets in PCAP format
- GPU recommended (CUDA support) but CPU works
- 8GB+ RAM for training (4GB for inference)
