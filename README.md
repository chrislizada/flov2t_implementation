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
EdgeFedIDS/
├── src/
│   ├── data_preprocessing.py      # Pcap2Flow, Packet2Patch, Flow2Image
│   ├── models.py                  # ViT with LoRA implementation
│   ├── federated_aggregation.py   # RGPA and FedAvg aggregators
│   ├── federated_trainer.py       # Client and Server training logic
│   ├── config.py                  # Configuration parameters
│   ├── utils.py                   # Helper functions
│   └── main.py                    # Main training script
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training

```python
from src.main import main
main()
```

### Custom Configuration

```python
from src.config import FLoV2TConfig
from src.federated_trainer import Client, FederatedServer
from src.data_preprocessing import TrafficPreprocessor

# Configure parameters
config = FLoV2TConfig()
config.NUM_ROUNDS = 20
config.LOCAL_EPOCHS = 5
config.LORA_RANK = 4
config.LAMBDA_REG = 0.1

# Preprocess PCAP files
preprocessor = TrafficPreprocessor()
images = preprocessor.preprocess_pcap('path/to/file.pcap')

# Train federated model
server = FederatedServer(
    num_classes=8,
    rank=4,
    alpha=8,
    aggregation_method='rgpa',
    lambda_reg=0.1
)
```

### Data Preprocessing

```python
from src.data_preprocessing import TrafficPreprocessor

preprocessor = TrafficPreprocessor(max_packets=196, patch_size=16)

# Process single PCAP file
images = preprocessor.preprocess_pcap('traffic.pcap')

# Process dataset directory
labels_dict = {
    'botnet_traffic.pcap': 0,
    'dos_slowloris.pcap': 1,
    # ...
}
data, labels = preprocessor.preprocess_dataset('pcap_dir/', labels_dict)
```

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

## Notes

- Requires CICIDS2017 or CICIDS2018 datasets in PCAP format
- GPU recommended for training (CUDA support)
- Pre-trained ViT model downloaded automatically from HuggingFace
- Supports both IID and non-IID data distributions
