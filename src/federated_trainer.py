import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime

from models import FLoV2TModel
from federated_aggregation import RegularizedGlobalAggregator, FedAvgAggregator

class Client:
    def __init__(self, client_id: int, data: np.ndarray, labels: np.ndarray, 
                 num_classes: int = 8, rank: int = 4, alpha: int = 8, device: str = 'cuda'):
        self.client_id = client_id
        self.device = device
        
        self.model = FLoV2TModel(num_classes=num_classes, rank=rank, alpha=alpha).to(device)
        
        self.dataset = TensorDataset(
            torch.from_numpy(data).float(),
            torch.from_numpy(labels).long()
        )
        self.data_size = len(self.dataset)
        
    def set_global_model(self, global_state: Dict):
        self.model.load_lora_state_dict(global_state)
    
    def train(self, epochs: int = 5, batch_size: int = 32, lr: float = 0.001) -> Dict:
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.get_trainable_parameters(), lr=lr, weight_decay=0.01)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        return self.model.get_lora_state_dict()
    
    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray, 
                batch_size: int = 32, return_confusion: bool = False) -> Tuple:
        test_dataset = TensorDataset(
            torch.from_numpy(test_data).float(),
            torch.from_numpy(test_labels).long()
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)
                outputs = self.model(batch_data)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(batch_labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = np.mean(all_preds == all_labels)
        
        num_classes = len(np.unique(all_labels))
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        
        for c in range(num_classes):
            tp = np.sum((all_preds == c) & (all_labels == c))
            fp = np.sum((all_preds == c) & (all_labels != c))
            fn = np.sum((all_preds != c) & (all_labels == c))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)
        
        avg_precision = np.mean(precision_per_class)
        avg_recall = np.mean(recall_per_class)
        avg_f1 = np.mean(f1_per_class)
        
        if return_confusion:
            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
            for true_label, pred_label in zip(all_labels, all_preds):
                confusion_matrix[true_label, pred_label] += 1
            return accuracy, avg_precision, avg_recall, avg_f1, confusion_matrix
        
        return accuracy, avg_precision, avg_recall, avg_f1

class FederatedServer:
    def __init__(self, num_classes: int = 8, rank: int = 4, alpha: int = 8, 
                 aggregation_method: str = 'rgpa', lambda_reg: float = 0.1, device: str = 'cuda',
                 checkpoint_dir: Optional[str] = None):
        self.device = device
        self.num_classes = num_classes
        self.rank = rank
        self.alpha = alpha
        self.checkpoint_dir = checkpoint_dir
        
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        self.global_model = FLoV2TModel(num_classes=num_classes, rank=rank, alpha=alpha).to(device)
        
        if aggregation_method == 'rgpa':
            self.aggregator = RegularizedGlobalAggregator(lambda_reg=lambda_reg)
        else:
            self.aggregator = FedAvgAggregator()
        
        self.best_accuracy = 0.0
        self.best_f1 = 0.0
    
    def get_global_state(self) -> Dict:
        return self.global_model.get_lora_state_dict()
    
    def aggregate_client_updates(self, client_states: List[Dict], client_data_sizes: List[int]):
        global_state = self.aggregator.aggregate(client_states, client_data_sizes)
        self.global_model.load_lora_state_dict(global_state)
    
    def save_checkpoint(self, round_num: int, history: Dict, is_best: bool = False):
        if self.checkpoint_dir is None:
            return
        
        checkpoint = {
            'round': round_num,
            'model_state_dict': self.global_model.get_lora_state_dict(),
            'history': history,
            'best_accuracy': self.best_accuracy,
            'best_f1': self.best_f1
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_round_{round_num}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (Acc: {self.best_accuracy:.4f}, F1: {self.best_f1:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.global_model.load_lora_state_dict(checkpoint['model_state_dict'])
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        self.best_f1 = checkpoint.get('best_f1', 0.0)
        return checkpoint.get('round', 0), checkpoint.get('history', {})
    
    def train_federated(self, clients: List[Client], num_rounds: int = 20, 
                       local_epochs: int = 5, batch_size: int = 32, lr: float = 0.001,
                       test_data: np.ndarray = None, test_labels: np.ndarray = None,
                       early_stopping_patience: int = 5, save_every: int = 5):
        
        history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        patience_counter = 0
        
        for round_num in range(num_rounds):
            print(f"\n=== Round {round_num + 1}/{num_rounds} ===")
            
            client_states = []
            client_data_sizes = []
            
            for client in tqdm(clients, desc="Training clients"):
                global_state = self.get_global_state()
                client.set_global_model(global_state)
                
                client_state = client.train(epochs=local_epochs, batch_size=batch_size, lr=lr)
                
                client_states.append(client_state)
                client_data_sizes.append(client.data_size)
            
            self.aggregate_client_updates(client_states, client_data_sizes)
            
            if test_data is not None and test_labels is not None:
                global_state = self.get_global_state()
                clients[0].set_global_model(global_state)
                
                acc, prec, rec, f1 = clients[0].evaluate(test_data, test_labels, batch_size=batch_size)
                
                history['accuracy'].append(acc)
                history['precision'].append(prec)
                history['recall'].append(rec)
                history['f1'].append(f1)
                
                print(f"Global Model - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
                
                is_best = False
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.best_accuracy = acc
                    is_best = True
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if (round_num + 1) % save_every == 0 or is_best:
                    self.save_checkpoint(round_num + 1, history, is_best)
                
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {round_num + 1} rounds (patience={early_stopping_patience})")
                    break
        
        return history
