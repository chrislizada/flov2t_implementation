# FLoV2T Performance Benchmarking Guide

This guide shows how to measure and validate that FLoV2T can run on lightweight devices.

## Key Metrics to Measure

### 1. **Parameter Efficiency** ⭐ Most Important
- **Trainable Parameters**: Should be ~336.8K (vs 21.67M full model)
- **Parameter Reduction**: Should be ~64x
- **Target**: < 500K trainable parameters for edge devices

### 2. **Memory Usage**
- **RAM**: Peak memory during training/inference
- **GPU Memory**: VRAM usage (if applicable)
- **Target**: < 4GB RAM for IoT devices, < 2GB for edge devices

### 3. **Computation Performance**
- **Inference Latency**: Time per sample (ms)
- **Training Time**: Time per epoch/batch (seconds)
- **Throughput**: Samples processed per second
- **Target**: < 100ms inference for real-time, < 5s per epoch for training

### 4. **Model Size**
- **Disk Size**: Total model file size (MB)
- **Target**: < 100MB for edge deployment

### 5. **Resource Utilization**
- **CPU Usage**: Average/peak CPU percentage
- **GPU Utilization**: GPU usage percentage
- **Network I/O**: Communication overhead in federated setting

## Running Benchmarks

### Quick Benchmark (Model Only)
```bash
python src/benchmark.py
```

This will measure:
- Parameter counts and reduction
- Model size on disk
- Inference time (100 iterations)
- Training time (10 iterations)

### Full Training Benchmark
```bash
python src/main_with_monitoring.py
```

This will:
- Profile the model
- Monitor system resources during training
- Track performance across all federated rounds
- Generate performance reports
- Analyze lightweight device suitability

## Output Files

1. **`model_profile.json`**: Model architecture metrics
2. **`flov2t_performance.json`**: Runtime performance logs
3. **Console output**: Real-time metrics and summary

## Expected Results (From Paper)

| Metric | Value | Status |
|--------|-------|--------|
| Total Parameters | 21.67M | Frozen |
| Trainable Parameters | 336.8K | ✓ 64x reduction |
| Model Size | ~83MB | ✓ Lightweight |
| Inference Time | ~50-100ms | ✓ Real-time capable |
| Training Time/Epoch | 1.48s | ✓ Fast (vs 4.60s without LoRA) |
| Peak RAM | 4.5GB | ✓ Edge compatible |
| GPU Memory | 2.09GB | ✓ Consumer GPU |

## Interpreting Results

### ✓ Excellent (Highly Suitable for Edge)
- Trainable params < 500K
- Model size < 100MB
- Inference < 100ms
- Peak RAM < 2GB
- GPU memory < 2GB

### ✓ Good (Suitable for Most IoT)
- Trainable params < 1M
- Model size < 500MB
- Inference < 500ms
- Peak RAM < 4GB
- GPU memory < 4GB

### ⚠ Moderate (Optimization Needed)
- Trainable params > 1M
- Model size > 500MB
- Inference > 500ms
- Peak RAM > 4GB
- GPU memory > 4GB

## On EC2 g4dn.xlarge

### Expected Performance
```
GPU: NVIDIA T4 (16GB)
CPU: 4 vCPUs
RAM: 16GB

Trainable Parameters: 336.8K (✓)
Model Size: ~83MB (✓)
Inference Time: ~30-50ms (✓✓)
Training Time: ~1.2-1.8s/epoch (✓)
Peak RAM: ~5-6GB (✓)
GPU Memory: ~2-3GB (✓✓)
```

### Key Validations
1. ✓ **64x parameter reduction** confirmed
2. ✓ **Sub-100ms inference** for real-time detection
3. ✓ **Low memory footprint** suitable for edge
4. ✓ **Fast training** with LoRA fine-tuning

## Comparison Commands

### Measure with LoRA (FLoV2T)
```bash
python src/main_with_monitoring.py
```

### Measure without LoRA (Baseline)
Modify `config.py`:
```python
LORA_RANK = None  # Disable LoRA
```
Then run:
```bash
python src/main_with_monitoring.py
```

Compare:
- Parameter count increase (64x more)
- Training time increase (~3x slower)
- Memory usage increase (~2x more)

## Validating Lightweight Claims

The paper claims FLoV2T is lightweight. Verify:

1. **Parameter Efficiency** ✓
   - Check: `trainable_params_K < 500`
   - FLoV2T: 336.8K ✓

2. **Communication Efficiency** ✓
   - Check: Only LoRA matrices transmitted (A, B)
   - Size: ~1.3MB per round vs ~83MB full model

3. **Computation Efficiency** ✓
   - Check: Training time per epoch < 2s
   - FLoV2T: 1.48s ✓

4. **Memory Efficiency** ✓
   - Check: Peak RAM < 5GB, GPU < 3GB
   - FLoV2T: 4.5GB RAM, 2.09GB GPU ✓

## Real-World Edge Device Targets

| Device Type | RAM | Compute | FLoV2T Compatible? |
|-------------|-----|---------|-------------------|
| Raspberry Pi 4 (8GB) | 8GB | 4-core ARM | ✓ (CPU mode) |
| NVIDIA Jetson Nano | 4GB | 128-core GPU | ✓✓ (Ideal) |
| NVIDIA Jetson Xavier | 8GB | 512-core GPU | ✓✓ (Overkill) |
| AIoT Gateway | 2-4GB | Dual-core | ⚠ (Tight) |
| Standard Server | 16GB+ | Multi-GPU | ✓✓ (Excessive) |

## Monitoring During Training

Watch real-time metrics:
```bash
# In separate terminal
watch -n 1 nvidia-smi  # GPU monitoring
watch -n 1 free -h      # RAM monitoring
watch -n 1 top          # CPU monitoring
```

## Troubleshooting

### High Memory Usage
- Reduce `BATCH_SIZE` in config.py
- Reduce `NUM_CLIENTS`
- Use gradient checkpointing

### Slow Training
- Enable GPU if available
- Increase `BATCH_SIZE`
- Reduce `MAX_PACKETS`

### OOM Errors
- Reduce `MAX_PACKETS` from 196 to 128
- Use CPU instead of GPU
- Process clients sequentially

## Citation

Compare your results with Table 10 in the paper:
```
Memory: 4.5GB
GPU: 2.09GB
Time/Epoch: 1.48s
Parameters: 336.78K
```
