import torch
import psutil
import time
import GPUtil
import numpy as np
from datetime import datetime
import json

class PerformanceMonitor:
    def __init__(self, log_file='performance_log.json'):
        self.log_file = log_file
        self.metrics = []
        self.start_time = None
        
    def get_cpu_usage(self):
        return psutil.cpu_percent(interval=1)
    
    def get_memory_usage(self):
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent': memory.percent
        }
    
    def get_gpu_usage(self):
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'gpu_id': gpu.id,
                    'gpu_name': gpu.name,
                    'gpu_load_percent': gpu.load * 100,
                    'gpu_memory_used_mb': gpu.memoryUsed,
                    'gpu_memory_total_mb': gpu.memoryTotal,
                    'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'gpu_temperature': gpu.temperature
                }
        except:
            pass
        
        if torch.cuda.is_available():
            return {
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated(0) / (1024**3),
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved(0) / (1024**3),
                'gpu_memory_max_allocated_gb': torch.cuda.max_memory_allocated(0) / (1024**3)
            }
        return None
    
    def get_disk_usage(self):
        disk = psutil.disk_usage('/')
        return {
            'total_gb': disk.total / (1024**3),
            'used_gb': disk.used / (1024**3),
            'free_gb': disk.free / (1024**3),
            'percent': disk.percent
        }
    
    def get_network_io(self):
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent_mb': net_io.bytes_sent / (1024**2),
            'bytes_recv_mb': net_io.bytes_recv / (1024**2),
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
    
    def start_monitoring(self):
        self.start_time = time.time()
        
    def record_snapshot(self, phase='', epoch=None, additional_info=None):
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_time_sec': time.time() - self.start_time if self.start_time else 0,
            'phase': phase,
            'epoch': epoch,
            'cpu_percent': self.get_cpu_usage(),
            'memory': self.get_memory_usage(),
            'gpu': self.get_gpu_usage(),
            'disk': self.get_disk_usage(),
            'network': self.get_network_io()
        }
        
        if additional_info:
            snapshot.update(additional_info)
        
        self.metrics.append(snapshot)
        return snapshot
    
    def save_log(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def print_summary(self):
        if not self.metrics:
            print("No metrics recorded")
            return
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        cpu_usage = [m['cpu_percent'] for m in self.metrics]
        print(f"\nCPU Usage:")
        print(f"  Average: {np.mean(cpu_usage):.2f}%")
        print(f"  Max: {np.max(cpu_usage):.2f}%")
        print(f"  Min: {np.min(cpu_usage):.2f}%")
        
        mem_usage = [m['memory']['used_gb'] for m in self.metrics]
        print(f"\nRAM Usage:")
        print(f"  Average: {np.mean(mem_usage):.2f} GB")
        print(f"  Max: {np.max(mem_usage):.2f} GB")
        print(f"  Min: {np.min(mem_usage):.2f} GB")
        
        if self.metrics[0]['gpu']:
            if 'gpu_memory_allocated_gb' in self.metrics[0]['gpu']:
                gpu_mem = [m['gpu']['gpu_memory_allocated_gb'] for m in self.metrics if m['gpu']]
                print(f"\nGPU Memory (Allocated):")
                print(f"  Average: {np.mean(gpu_mem):.2f} GB")
                print(f"  Max: {np.max(gpu_mem):.2f} GB")
            
            if 'gpu_load_percent' in self.metrics[0]['gpu']:
                gpu_load = [m['gpu']['gpu_load_percent'] for m in self.metrics if m['gpu'] and 'gpu_load_percent' in m['gpu']]
                if gpu_load:
                    print(f"\nGPU Utilization:")
                    print(f"  Average: {np.mean(gpu_load):.2f}%")
                    print(f"  Max: {np.max(gpu_load):.2f}%")
        
        total_time = self.metrics[-1]['elapsed_time_sec']
        print(f"\nTotal Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        print("="*60 + "\n")

class ModelProfiler:
    @staticmethod
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'total_params_M': total_params / 1e6,
            'trainable_params_K': trainable_params / 1e3,
            'parameter_reduction': total_params / trainable_params if trainable_params > 0 else 0
        }
    
    @staticmethod
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size_mb = (param_size + buffer_size) / (1024**2)
        
        return {
            'param_size_mb': param_size / (1024**2),
            'buffer_size_mb': buffer_size / (1024**2),
            'total_size_mb': total_size_mb
        }
    
    @staticmethod
    def measure_inference_time(model, input_tensor, num_iterations=100, warmup=10):
        model.eval()
        
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(input_tensor)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times = []
        for _ in range(num_iterations):
            start = time.time()
            with torch.no_grad():
                _ = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - start)
        
        return {
            'avg_inference_time_ms': np.mean(times) * 1000,
            'std_inference_time_ms': np.std(times) * 1000,
            'min_inference_time_ms': np.min(times) * 1000,
            'max_inference_time_ms': np.max(times) * 1000,
            'throughput_samples_per_sec': 1.0 / np.mean(times)
        }
    
    @staticmethod
    def measure_training_time(model, input_tensor, target_tensor, num_iterations=10):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        times = []
        for _ in range(num_iterations):
            start = time.time()
            
            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            times.append(time.time() - start)
        
        return {
            'avg_training_time_ms': np.mean(times) * 1000,
            'std_training_time_ms': np.std(times) * 1000,
            'min_training_time_ms': np.min(times) * 1000,
            'max_training_time_ms': np.max(times) * 1000
        }
    
    @staticmethod
    def profile_model(model, input_shape=(1, 256, 224, 224), num_classes=8, device='cuda'):
        model = model.to(device)
        input_tensor = torch.randn(input_shape).to(device)
        target_tensor = torch.randint(0, num_classes, (input_shape[0],)).to(device)
        
        param_info = ModelProfiler.count_parameters(model)
        size_info = ModelProfiler.get_model_size(model)
        inference_info = ModelProfiler.measure_inference_time(model, input_tensor)
        training_info = ModelProfiler.measure_training_time(model, input_tensor, target_tensor)
        
        profile = {
            'parameters': param_info,
            'model_size': size_info,
            'inference': inference_info,
            'training': training_info
        }
        
        return profile
    
    @staticmethod
    def print_profile(profile):
        print("\n" + "="*60)
        print("MODEL PROFILE")
        print("="*60)
        
        print("\nParameters:")
        print(f"  Total: {profile['parameters']['total_params']:,} ({profile['parameters']['total_params_M']:.2f}M)")
        print(f"  Trainable: {profile['parameters']['trainable_params']:,} ({profile['parameters']['trainable_params_K']:.2f}K)")
        print(f"  Frozen: {profile['parameters']['frozen_params']:,}")
        print(f"  Reduction: {profile['parameters']['parameter_reduction']:.2f}x")
        
        print("\nModel Size:")
        print(f"  Total: {profile['model_size']['total_size_mb']:.2f} MB")
        print(f"  Parameters: {profile['model_size']['param_size_mb']:.2f} MB")
        print(f"  Buffers: {profile['model_size']['buffer_size_mb']:.2f} MB")
        
        print("\nInference Performance:")
        print(f"  Average Time: {profile['inference']['avg_inference_time_ms']:.2f} ms")
        print(f"  Std Dev: {profile['inference']['std_inference_time_ms']:.2f} ms")
        print(f"  Throughput: {profile['inference']['throughput_samples_per_sec']:.2f} samples/sec")
        
        print("\nTraining Performance:")
        print(f"  Average Time: {profile['training']['avg_training_time_ms']:.2f} ms")
        print(f"  Std Dev: {profile['training']['std_training_time_ms']:.2f} ms")
        
        print("="*60 + "\n")

def benchmark_flov2t():
    from models import FLoV2TModel
    
    print("Benchmarking FLoV2T Model...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = FLoV2TModel(num_classes=8, rank=4, alpha=8)
    
    profile = ModelProfiler.profile_model(model, device=device)
    ModelProfiler.print_profile(profile)
    
    with open('model_profile.json', 'w') as f:
        json.dump(profile, f, indent=2)
    
    print("Profile saved to model_profile.json")

if __name__ == "__main__":
    benchmark_flov2t()
