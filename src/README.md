# FLoV2T Implementation - Source Code

## Overview

This directory contains the complete implementation of FLoV2T (Federated Learning with LoRA and ViT for malicious traffic classification in AIoT).

## Files

### Core Implementation

- **`config.py`** - Configuration parameters and constants
  - Attack categories, hyperparameters, non-IID distributions
  - Paper-specified distributions for CICIDS2017/2018

- **`data_preprocessing.py`** - Raw traffic preprocessing pipeline
  - `Pcap2Flow`: Split PCAP files into network flows
  - `Packet2Patch`: Convert packets to 16×16 patches (256 bytes)
  - `Flow2Image`: Create 224×224 RGB images from 196 patches
  - Handshake packet detection for intelligent padding

- **`models.py`** - Neural network models
  - `LoRALayer`: Low-rank adaptation layer
  - `LoRALinear`: LoRA-enhanced linear layer
  - `ViTWithLoRA`: Vision Transformer with LoRA (ViT-tiny)
  - `FLoV2TModel`: Complete FLoV2T model

- **`federated_aggregation.py`** - Parameter aggregation strategies
  - `FedAvgAggregator`: Standard federated averaging
  - `RegularizedGlobalAggregator`: RGPA with λ regularization

- **`federated_trainer.py`** - Federated learning training loop
  - `Client`: Client-side training and evaluation
  - `FederatedServer`: Server-side aggregation and orchestration
  - Checkpointing, early stopping, best model tracking

- **`utils.py`** - Utility functions
  - Data partitioning (IID/non-IID)
  - Metrics calculation
  - Model parameter counting

- **`main.py`** - Main training script with CLI

### Testing

- **`test_preprocessing.py`** - Tests for data preprocessing
- **`test_models.py`** - Tests for model architecture
- **`test_aggregation.py`** - Tests for aggregation methods

## Quick Start

### Basic Usage

```bash
# IID scenario with 3 clients
python main.py --num_clients 3 --scenario iid

# Non-IID scenario with 5 clients
python main.py --num_clients 5 --scenario non-iid --dataset CICIDS2017

# Custom configuration
python main.py --num_clients 3 --scenario non-iid --num_rounds 30 --batch_size 64
```

### Command Line Arguments

```
--num_clients      Number of clients (3 or 5)
--scenario         Data distribution (iid or non-iid)
--dataset          Dataset name (CICIDS2017 or CICIDS2018)
--num_rounds       Number of federated rounds (default: 20)
--local_epochs     Local training epochs (default: 5)
--batch_size       Batch size (default: 32)
--lr               Learning rate (default: 0.001)
--aggregation      Aggregation method (rgpa or fedavg)
--checkpoint_dir   Directory for checkpoints (default: ./checkpoints)
--seed             Random seed (default: 42)
--data_path        Path to PCAP data directory
```

## Running Tests

```bash
# Run all tests
python -m pytest test_*.py -v

# Run specific test file
python test_preprocessing.py
python test_models.py
python test_aggregation.py
```

## Architecture

### Data Flow

```
PCAP Files → Pcap2Flow → Packet2Patch → Flow2Image → 224×224 RGB Images
                                                              ↓
                                                         ViT-tiny + LoRA
                                                              ↓
                                                      8-class Classification
```

### Federated Learning Flow

```
Round 1-N:
  1. Server broadcasts global model to clients
  2. Each client trains locally with LoRA
  3. Clients send LoRA parameters (A, B matrices) to server
  4. Server aggregates using RGPA
  5. Evaluate on test set
  6. Save checkpoint if best performance
```

## Key Parameters (from Paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max Packets | 196 | Maximum packets per flow |
| Patch Size | 16×16 | Patch dimensions (256 bytes) |
| Image Size | 224×224 | Final image size |
| LoRA Rank | 4 | Low-rank dimension |
| LoRA Alpha | 8 | Scaling factor |
| Lambda (λ) | 0.1 | RGPA regularization |
| Learning Rate | 0.001 | AdamW learning rate |
| Weight Decay | 0.01 | L2 regularization |

## Performance Targets (from Paper)

### IID Scenario
- Accuracy: 97.26%
- F1 Score: 96.99%
- Parameter Reduction: ~64×

### Non-IID Scenario
- Accuracy: 96.17%
- F1 Score: 95.81%

## Attack Categories

```python
0: 'Botnet'
1: 'DoS-Slowloris'
2: 'DoS-Goldeneye'
3: 'DoS-Hulk'
4: 'SSH-BruteForce'
5: 'Web-SQL'
6: 'Web-XSS'
7: 'Web-Bruteforce'
```

## Implementation Notes

### Fixed Issues
1. ✅ Correct 224×224 image generation from 196 patches
2. ✅ ViT-tiny instead of ViT-base
3. ✅ Proper RGB channel handling
4. ✅ LoRA applied to all attention + FFN layers
5. ✅ Paper-specified non-IID distributions
6. ✅ Handshake packet detection for padding
7. ✅ Checkpointing and early stopping

### Current Limitations
- Uses dummy data (replace with actual PCAP files)
- No baseline implementations (DeepFed, FELIDS, SSFL-IDS)
- Limited visualization (add training curves, confusion matrices)

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
