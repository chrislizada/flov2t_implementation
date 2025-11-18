import torch
import numpy as np
import psutil
import time
import os
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import GPUtil
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

try:
    from pyJoules.energy_meter import EnergyMeter
    from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain
    ENERGY_AVAILABLE = True
except:
    ENERGY_AVAILABLE = False

class DeploymentMonitor:
    """
    Comprehensive monitoring for FLoV2T deployed on IoT/Edge devices.
    
    Monitors:
    - Memory: RAM, swap, GPU memory
    - Computation: CPU usage, inference time
    - Energy: Power consumption (if available)
    - Communication: Bytes transmitted
    - Latency: End-to-end detection time
    - Accuracy: Precision, Recall, F1, Confusion Matrix
    """
    
    def __init__(self, model, device: str = 'cpu', log_dir: str = './deployment_logs'):
        self.model = model
        self.device = device
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = {
            'memory': [],
            'computation': [],
            'energy': [],
            'communication': [],
            'latency': [],
            'accuracy': []
        }
        
        self.start_time = time.time()
        self.monitoring = False
        self.monitor_thread = None
        
        self.initial_net_io = psutil.net_io_counters()
        self.bytes_sent_baseline = self.initial_net_io.bytes_sent
        self.bytes_recv_baseline = self.initial_net_io.bytes_recv
        
        print(f"Deployment Monitor initialized")
        print(f"Device: {device}")
        print(f"Log directory: {log_dir}")
        print(f"Energy monitoring: {'Enabled' if ENERGY_AVAILABLE else 'Disabled (install pyJoules)'}")
        print(f"GPU monitoring: {'Enabled' if GPU_AVAILABLE and torch.cuda.is_available() else 'Disabled'}")
    
    def start_background_monitoring(self, interval: float = 1.0):
        """Start background thread for continuous monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        print(f"\nBackground monitoring started (interval: {interval}s)")
    
    def stop_background_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Background monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop"""
        while self.monitoring:
            self._record_memory()
            self._record_computation()
            time.sleep(interval)
    
    def _record_memory(self) -> Dict:
        """Record memory usage"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_info = {
            'timestamp': time.time() - self.start_time,
            'ram_used_mb': memory.used / (1024**2),
            'ram_available_mb': memory.available / (1024**2),
            'ram_percent': memory.percent,
            'ram_peak_mb': memory.used / (1024**2),
            'swap_used_mb': swap.used / (1024**2),
            'swap_percent': swap.percent
        }
        
        if torch.cuda.is_available():
            memory_info['gpu_allocated_mb'] = torch.cuda.memory_allocated(0) / (1024**2)
            memory_info['gpu_reserved_mb'] = torch.cuda.memory_reserved(0) / (1024**2)
            memory_info['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated(0) / (1024**2)
        
        self.metrics['memory'].append(memory_info)
        return memory_info
    
    def _record_computation(self) -> Dict:
        """Record CPU/GPU computation metrics"""
        comp_info = {
            'timestamp': time.time() - self.start_time,
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }
        
        if GPU_AVAILABLE and torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    comp_info['gpu_load_percent'] = gpu.load * 100
                    comp_info['gpu_temp_celsius'] = gpu.temperature
            except:
                pass
        
        self.metrics['computation'].append(comp_info)
        return comp_info
    
    def _record_energy(self, duration: float) -> Dict:
        """Record energy consumption"""
        if not ENERGY_AVAILABLE:
            return {'available': False}
        
        try:
            energy_info = {
                'timestamp': time.time() - self.start_time,
                'duration_sec': duration,
                'available': True
            }
            self.metrics['energy'].append(energy_info)
            return energy_info
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _record_communication(self, bytes_sent: int = 0, bytes_received: int = 0) -> Dict:
        """Record communication overhead"""
        net_io = psutil.net_io_counters()
        
        comm_info = {
            'timestamp': time.time() - self.start_time,
            'bytes_sent': bytes_sent,
            'bytes_received': bytes_received,
            'total_bytes_sent': net_io.bytes_sent - self.bytes_sent_baseline,
            'total_bytes_recv': net_io.bytes_recv - self.bytes_recv_baseline,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
        
        self.metrics['communication'].append(comm_info)
        return comm_info
    
    def measure_inference_latency(self, data: np.ndarray, batch_size: int = 1, 
                                  warmup: int = 10) -> Dict:
        """
        Measure end-to-end inference latency
        
        Args:
            data: Input data (N, C, H, W)
            batch_size: Batch size for inference
            warmup: Number of warmup iterations
        
        Returns:
            Latency metrics
        """
        self.model.eval()
        
        if len(data) < batch_size:
            batch_size = len(data)
        
        test_batch = torch.tensor(data[:batch_size]).float().to(self.device)
        if test_batch.dtype == torch.uint8:
            test_batch = test_batch.float() / 255.0
        
        print(f"\nMeasuring inference latency (batch_size={batch_size}, warmup={warmup})...")
        
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model(test_batch)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        latencies = []
        for i in range(0, min(100, len(data)), batch_size):
            batch = torch.tensor(data[i:i+batch_size]).float().to(self.device)
            if batch.dtype == torch.uint8:
                batch = batch.float() / 255.0
            
            start = time.time()
            with torch.no_grad():
                _ = self.model(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.time() - start)
        
        latency_info = {
            'timestamp': time.time() - self.start_time,
            'batch_size': batch_size,
            'avg_latency_ms': np.mean(latencies) * 1000,
            'std_latency_ms': np.std(latencies) * 1000,
            'min_latency_ms': np.min(latencies) * 1000,
            'max_latency_ms': np.max(latencies) * 1000,
            'p50_latency_ms': np.percentile(latencies, 50) * 1000,
            'p95_latency_ms': np.percentile(latencies, 95) * 1000,
            'p99_latency_ms': np.percentile(latencies, 99) * 1000,
            'throughput_samples_per_sec': batch_size / np.mean(latencies)
        }
        
        self.metrics['latency'].append(latency_info)
        
        print(f"Latency Results:")
        print(f"  Average: {latency_info['avg_latency_ms']:.2f} ms")
        print(f"  Std Dev: {latency_info['std_latency_ms']:.2f} ms")
        print(f"  P50: {latency_info['p50_latency_ms']:.2f} ms")
        print(f"  P95: {latency_info['p95_latency_ms']:.2f} ms")
        print(f"  P99: {latency_info['p99_latency_ms']:.2f} ms")
        print(f"  Throughput: {latency_info['throughput_samples_per_sec']:.2f} samples/sec")
        
        return latency_info
    
    def evaluate_accuracy(self, data: np.ndarray, labels: np.ndarray, 
                         batch_size: int = 32) -> Dict:
        """
        Evaluate model accuracy with comprehensive metrics
        
        Args:
            data: Test data (N, C, H, W)
            labels: Ground truth labels
            batch_size: Batch size for evaluation
        
        Returns:
            Accuracy metrics
        """
        self.model.eval()
        
        print(f"\nEvaluating accuracy on {len(data)} samples...")
        
        all_predictions = []
        all_labels = []
        inference_times = []
        
        mem_before = self._record_memory()
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            batch_tensor = torch.tensor(batch_data).float().to(self.device)
            if batch_tensor.dtype == torch.uint8:
                batch_tensor = batch_tensor.float() / 255.0
            
            start = time.time()
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            inference_times.append(time.time() - start)
            
            all_predictions.extend(predictions)
            all_labels.extend(batch_labels)
        
        mem_after = self._record_memory()
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        per_class_precision = precision_score(all_labels, all_predictions, average=None, zero_division=0)
        per_class_recall = recall_score(all_labels, all_predictions, average=None, zero_division=0)
        per_class_f1 = f1_score(all_labels, all_predictions, average=None, zero_division=0)
        
        accuracy_info = {
            'timestamp': time.time() - self.start_time,
            'num_samples': len(data),
            'batch_size': batch_size,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'avg_inference_time_ms': np.mean(inference_times) * 1000,
            'total_inference_time_sec': np.sum(inference_times),
            'memory_used_mb': mem_after['ram_used_mb'] - mem_before['ram_used_mb'],
            'confusion_matrix': conf_matrix.tolist(),
            'per_class_metrics': {
                'precision': per_class_precision.tolist(),
                'recall': per_class_recall.tolist(),
                'f1_score': per_class_f1.tolist()
            }
        }
        
        self.metrics['accuracy'].append(accuracy_info)
        
        print(f"\nAccuracy Results:")
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall:    {recall*100:.2f}%")
        print(f"  F1 Score:  {f1*100:.2f}%")
        print(f"\nPer-class Metrics:")
        from config import FLoV2TConfig
        config = FLoV2TConfig()
        for i in range(len(per_class_precision)):
            if i < len(config.ATTACK_CATEGORIES):
                print(f"  Class {i} ({config.ATTACK_CATEGORIES[i]:20s}): "
                      f"P={per_class_precision[i]*100:5.2f}% "
                      f"R={per_class_recall[i]*100:5.2f}% "
                      f"F1={per_class_f1[i]*100:5.2f}%")
        
        return accuracy_info
    
    def record_federated_communication(self, model_state_dict: Dict, round_num: int):
        """Record communication overhead for federated round"""
        
        model_size_bytes = 0
        for param_tensor in model_state_dict.values():
            model_size_bytes += param_tensor.nelement() * param_tensor.element_size()
        
        comm_info = self._record_communication(
            bytes_sent=model_size_bytes,
            bytes_received=model_size_bytes
        )
        comm_info['round'] = round_num
        comm_info['model_size_mb'] = model_size_bytes / (1024**2)
        
        print(f"\nRound {round_num} Communication:")
        print(f"  Model size: {model_size_bytes / (1024**2):.2f} MB")
        print(f"  Sent/Received: {model_size_bytes / (1024**2):.2f} MB")
        
        return comm_info
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict:
        """Generate comprehensive monitoring report"""
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.log_dir, f'deployment_report_{timestamp}.json')
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'duration_sec': time.time() - self.start_time,
            'device': self.device,
            'metrics': self.metrics,
            'summary': self._compute_summary()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Deployment Report Generated")
        print(f"{'='*80}")
        print(f"File: {output_file}")
        print(f"Duration: {report['duration_sec']:.2f} seconds")
        print(f"{'='*80}\n")
        
        self._print_summary(report['summary'])
        
        return report
    
    def _compute_summary(self) -> Dict:
        """Compute summary statistics"""
        summary = {}
        
        if self.metrics['memory']:
            ram_used = [m['ram_used_mb'] for m in self.metrics['memory']]
            summary['memory'] = {
                'peak_ram_mb': max(ram_used),
                'avg_ram_mb': np.mean(ram_used),
                'peak_ram_gb': max(ram_used) / 1024
            }
            
            if 'gpu_allocated_mb' in self.metrics['memory'][0]:
                gpu_mem = [m['gpu_allocated_mb'] for m in self.metrics['memory']]
                summary['memory']['peak_gpu_mb'] = max(gpu_mem)
                summary['memory']['avg_gpu_mb'] = np.mean(gpu_mem)
        
        if self.metrics['computation']:
            cpu_usage = [m['cpu_percent'] for m in self.metrics['computation']]
            summary['computation'] = {
                'avg_cpu_percent': np.mean(cpu_usage),
                'peak_cpu_percent': max(cpu_usage)
            }
        
        if self.metrics['latency']:
            latest = self.metrics['latency'][-1]
            summary['latency'] = {
                'avg_ms': latest['avg_latency_ms'],
                'p95_ms': latest['p95_latency_ms'],
                'p99_ms': latest['p99_latency_ms'],
                'throughput_samples_per_sec': latest['throughput_samples_per_sec']
            }
        
        if self.metrics['accuracy']:
            latest = self.metrics['accuracy'][-1]
            summary['accuracy'] = {
                'accuracy': latest['accuracy'],
                'precision': latest['precision'],
                'recall': latest['recall'],
                'f1_score': latest['f1_score']
            }
        
        if self.metrics['communication']:
            total_sent = sum(m['bytes_sent'] for m in self.metrics['communication'])
            total_recv = sum(m['bytes_received'] for m in self.metrics['communication'])
            summary['communication'] = {
                'total_sent_mb': total_sent / (1024**2),
                'total_recv_mb': total_recv / (1024**2),
                'total_mb': (total_sent + total_recv) / (1024**2)
            }
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print summary to console"""
        print("DEPLOYMENT SUMMARY")
        print("="*80)
        
        if 'memory' in summary:
            print(f"\n Memory:")
            print(f"  Peak RAM: {summary['memory']['peak_ram_gb']:.2f} GB")
            print(f"  Avg RAM:  {summary['memory']['avg_ram_mb']:.2f} MB")
            if 'peak_gpu_mb' in summary['memory']:
                print(f"  Peak GPU: {summary['memory']['peak_gpu_mb']:.2f} MB")
        
        if 'computation' in summary:
            print(f"\n Computation:")
            print(f"  Avg CPU: {summary['computation']['avg_cpu_percent']:.1f}%")
            print(f"  Peak CPU: {summary['computation']['peak_cpu_percent']:.1f}%")
        
        if 'latency' in summary:
            print(f"\n Latency:")
            print(f"  Average: {summary['latency']['avg_ms']:.2f} ms")
            print(f"  P95: {summary['latency']['p95_ms']:.2f} ms")
            print(f"  P99: {summary['latency']['p99_ms']:.2f} ms")
            print(f"  Throughput: {summary['latency']['throughput_samples_per_sec']:.2f} samples/sec")
        
        if 'accuracy' in summary:
            print(f"\n Accuracy:")
            print(f"  Accuracy:  {summary['accuracy']['accuracy']*100:.2f}%")
            print(f"  Precision: {summary['accuracy']['precision']*100:.2f}%")
            print(f"  Recall:    {summary['accuracy']['recall']*100:.2f}%")
            print(f"  F1 Score:  {summary['accuracy']['f1_score']*100:.2f}%")
        
        if 'communication' in summary:
            print(f"\n Communication:")
            print(f"  Total Sent: {summary['communication']['total_sent_mb']:.2f} MB")
            print(f"  Total Received: {summary['communication']['total_recv_mb']:.2f} MB")
            print(f"  Total: {summary['communication']['total_mb']:.2f} MB")
        
        print("="*80 + "\n")
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot confusion matrix from latest accuracy evaluation"""
        if not self.metrics['accuracy']:
            print("No accuracy metrics available")
            return
        
        from config import FLoV2TConfig
        config = FLoV2TConfig()
        
        conf_matrix = np.array(self.metrics['accuracy'][-1]['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(config.ATTACK_CATEGORIES.values()),
                   yticklabels=list(config.ATTACK_CATEGORIES.values()))
        plt.title('Confusion Matrix - FLoV2T Deployment')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.log_dir, f'confusion_matrix_{timestamp}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        plt.close()
