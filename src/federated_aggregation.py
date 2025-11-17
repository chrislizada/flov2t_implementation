import torch
import copy
from typing import List, Dict

class RegularizedGlobalAggregator:
    def __init__(self, lambda_reg: float = 0.1):
        self.lambda_reg = lambda_reg
        self.global_state = None
    
    def compute_client_weights(self, client_data_sizes: List[int]) -> List[float]:
        total_samples = sum(client_data_sizes)
        weights = [n_k / total_samples for n_k in client_data_sizes]
        return weights
    
    def weighted_average(self, client_states: List[Dict], client_weights: List[float]) -> Dict:
        averaged_state = {}
        
        for key in client_states[0].keys():
            averaged_state[key] = torch.zeros_like(client_states[0][key])
            
            for client_state, weight in zip(client_states, client_weights):
                averaged_state[key] += weight * client_state[key]
        
        return averaged_state
    
    def apply_regularization(self, averaged_state: Dict, client_states: List[Dict], 
                            client_weights: List[float]) -> Dict:
        regularized_state = {}
        
        for key in averaged_state.keys():
            regularization_term = torch.zeros_like(averaged_state[key])
            
            for client_state, weight in zip(client_states, client_weights):
                regularization_term += weight * (averaged_state[key] - client_state[key])
            
            regularized_state[key] = averaged_state[key] - self.lambda_reg * regularization_term
        
        return regularized_state
    
    def aggregate(self, client_states: List[Dict], client_data_sizes: List[int]) -> Dict:
        client_weights = self.compute_client_weights(client_data_sizes)
        
        averaged_state = self.weighted_average(client_states, client_weights)
        
        regularized_state = self.apply_regularization(averaged_state, client_states, client_weights)
        
        self.global_state = regularized_state
        
        return regularized_state
    
    def get_global_state(self) -> Dict:
        return self.global_state

class FedAvgAggregator:
    def __init__(self):
        self.global_state = None
    
    def compute_client_weights(self, client_data_sizes: List[int]) -> List[float]:
        total_samples = sum(client_data_sizes)
        weights = [n_k / total_samples for n_k in client_data_sizes]
        return weights
    
    def aggregate(self, client_states: List[Dict], client_data_sizes: List[int]) -> Dict:
        client_weights = self.compute_client_weights(client_data_sizes)
        
        averaged_state = {}
        for key in client_states[0].keys():
            averaged_state[key] = torch.zeros_like(client_states[0][key])
            
            for client_state, weight in zip(client_states, client_weights):
                averaged_state[key] += weight * client_state[key]
        
        self.global_state = averaged_state
        return averaged_state
    
    def get_global_state(self) -> Dict:
        return self.global_state
