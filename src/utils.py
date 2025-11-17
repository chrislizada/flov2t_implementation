import numpy as np
import torch
import random
from typing import List, Tuple, Dict

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_iid_partition(data: np.ndarray, labels: np.ndarray, 
                         num_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    num_samples = len(data)
    indices = np.random.permutation(num_samples)
    
    client_data = []
    samples_per_client = num_samples // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:
            end_idx = num_samples
        else:
            end_idx = (i + 1) * samples_per_client
        
        client_indices = indices[start_idx:end_idx]
        client_data.append((data[client_indices], labels[client_indices]))
    
    return client_data

def create_non_iid_partition(data: np.ndarray, labels: np.ndarray, 
                             num_clients: int, 
                             client_class_distribution: Dict[int, List[int]] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    if client_class_distribution is None:
        from config import FLoV2TConfig
        client_class_distribution = FLoV2TConfig.get_non_iid_distribution('CICIDS2017', num_clients)
    
    client_data = []
    
    for client_id in range(num_clients):
        assigned_classes = client_class_distribution.get(client_id, [])
        
        if not assigned_classes:
            print(f"Warning: Client {client_id} has no assigned classes")
            continue
        
        client_indices = []
        for class_label in assigned_classes:
            class_indices = np.where(labels == class_label)[0]
            client_indices.extend(class_indices)
        
        client_indices = np.array(client_indices)
        np.random.shuffle(client_indices)
        
        client_data.append((data[client_indices], labels[client_indices]))
    
    return client_data

def split_train_test(data: np.ndarray, labels: np.ndarray, 
                     test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_samples = len(data)
    indices = np.random.permutation(num_samples)
    
    test_size = int(num_samples * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    return data[train_indices], labels[train_indices], data[test_indices], labels[test_indices]

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    accuracy = np.mean(y_true == y_pred)
    
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    
    return {
        'accuracy': accuracy,
        'precision': np.mean(precision_per_class),
        'recall': np.mean(recall_per_class),
        'f1': np.mean(f1_per_class),
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def print_data_distribution(client_datasets: List[Tuple[np.ndarray, np.ndarray]], num_classes: int = 8):
    print("\n=== Client Data Distribution ===")
    for client_id, (data, labels) in enumerate(client_datasets):
        print(f"\nClient {client_id}: {len(data)} samples")
        for class_id in range(num_classes):
            count = np.sum(labels == class_id)
            if count > 0:
                from config import FLoV2TConfig
                class_name = FLoV2TConfig.ATTACK_CATEGORIES.get(class_id, f"Class {class_id}")
                print(f"  {class_name}: {count} samples")

def create_imbalanced_iid_partition(data: np.ndarray, labels: np.ndarray, 
                                   num_clients: int,
                                   class_sample_distribution: Dict[int, int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    if class_sample_distribution is None:
        from config import FLoV2TConfig
        class_sample_distribution = FLoV2TConfig.get_iid_sample_distribution(num_clients)
    
    client_data = [[] for _ in range(num_clients)]
    client_labels = [[] for _ in range(num_clients)]
    
    num_classes = len(np.unique(labels))
    
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)
        
        samples_per_client = class_sample_distribution.get(class_id, len(class_indices) // num_clients)
        
        for client_id in range(num_clients):
            start_idx = client_id * samples_per_client
            end_idx = min(start_idx + samples_per_client, len(class_indices))
            
            if start_idx < len(class_indices):
                selected_indices = class_indices[start_idx:end_idx]
                client_data[client_id].extend(data[selected_indices])
                client_labels[client_id].extend(labels[selected_indices])
    
    result = []
    for i in range(num_clients):
        combined_indices = list(range(len(client_data[i])))
        np.random.shuffle(combined_indices)
        
        shuffled_data = np.array([client_data[i][j] for j in combined_indices])
        shuffled_labels = np.array([client_labels[i][j] for j in combined_indices])
        
        result.append((shuffled_data, shuffled_labels))
    
    return result

def save_model(model, path: str):
    torch.save(model.state_dict(), path)

def load_model(model, path: str):
    model.load_state_dict(torch.load(path))
    return model
