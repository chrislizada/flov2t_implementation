import unittest
import torch
import numpy as np
from federated_aggregation import RegularizedGlobalAggregator, FedAvgAggregator


class TestFedAvgAggregator(unittest.TestCase):
    
    def test_compute_client_weights(self):
        aggregator = FedAvgAggregator()
        
        client_data_sizes = [100, 200, 300]
        weights = aggregator.compute_client_weights(client_data_sizes)
        
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertAlmostEqual(weights[0], 100/600)
        self.assertAlmostEqual(weights[1], 200/600)
        self.assertAlmostEqual(weights[2], 300/600)
    
    def test_aggregate_basic(self):
        aggregator = FedAvgAggregator()
        
        client_states = [
            {'param1': torch.tensor([1.0, 2.0, 3.0])},
            {'param1': torch.tensor([4.0, 5.0, 6.0])},
            {'param1': torch.tensor([7.0, 8.0, 9.0])}
        ]
        client_data_sizes = [100, 100, 100]
        
        global_state = aggregator.aggregate(client_states, client_data_sizes)
        
        expected = torch.tensor([4.0, 5.0, 6.0])
        torch.testing.assert_close(global_state['param1'], expected)
    
    def test_aggregate_weighted(self):
        aggregator = FedAvgAggregator()
        
        client_states = [
            {'param1': torch.tensor([2.0])},
            {'param1': torch.tensor([4.0])}
        ]
        client_data_sizes = [100, 300]
        
        global_state = aggregator.aggregate(client_states, client_data_sizes)
        
        expected = torch.tensor([3.5])
        torch.testing.assert_close(global_state['param1'], expected)


class TestRegularizedGlobalAggregator(unittest.TestCase):
    
    def test_initialization(self):
        aggregator = RegularizedGlobalAggregator(lambda_reg=0.1)
        
        self.assertEqual(aggregator.lambda_reg, 0.1)
        self.assertIsNone(aggregator.global_state)
    
    def test_compute_client_weights(self):
        aggregator = RegularizedGlobalAggregator()
        
        client_data_sizes = [246, 684, 2494]
        weights = aggregator.compute_client_weights(client_data_sizes)
        
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(sum(weights), 1.0)
    
    def test_weighted_average(self):
        aggregator = RegularizedGlobalAggregator()
        
        client_states = [
            {'param1': torch.tensor([1.0, 2.0])},
            {'param1': torch.tensor([3.0, 4.0])}
        ]
        client_weights = [0.5, 0.5]
        
        averaged = aggregator.weighted_average(client_states, client_weights)
        
        expected = torch.tensor([2.0, 3.0])
        torch.testing.assert_close(averaged['param1'], expected)
    
    def test_apply_regularization(self):
        aggregator = RegularizedGlobalAggregator(lambda_reg=0.1)
        
        averaged_state = {'param1': torch.tensor([5.0])}
        client_states = [
            {'param1': torch.tensor([4.0])},
            {'param1': torch.tensor([6.0])}
        ]
        client_weights = [0.5, 0.5]
        
        regularized = aggregator.apply_regularization(
            averaged_state, client_states, client_weights
        )
        
        self.assertIn('param1', regularized)
        self.assertEqual(regularized['param1'].shape, averaged_state['param1'].shape)
    
    def test_aggregate_full(self):
        aggregator = RegularizedGlobalAggregator(lambda_reg=0.1)
        
        client_states = [
            {'param1': torch.tensor([1.0, 2.0, 3.0])},
            {'param1': torch.tensor([2.0, 3.0, 4.0])},
            {'param1': torch.tensor([3.0, 4.0, 5.0])}
        ]
        client_data_sizes = [100, 200, 300]
        
        global_state = aggregator.aggregate(client_states, client_data_sizes)
        
        self.assertIn('param1', global_state)
        self.assertEqual(global_state['param1'].shape, (3,))
        
        self.assertIsNotNone(aggregator.global_state)
    
    def test_regularization_effect(self):
        aggregator_no_reg = RegularizedGlobalAggregator(lambda_reg=0.0)
        aggregator_with_reg = RegularizedGlobalAggregator(lambda_reg=0.5)
        
        client_states = [
            {'param1': torch.tensor([1.0])},
            {'param1': torch.tensor([9.0])}
        ]
        client_data_sizes = [100, 100]
        
        state_no_reg = aggregator_no_reg.aggregate(client_states, client_data_sizes)
        state_with_reg = aggregator_with_reg.aggregate(client_states, client_data_sizes)
        
        self.assertFalse(torch.equal(state_no_reg['param1'], state_with_reg['param1']))


class TestAggregatorComparison(unittest.TestCase):
    
    def test_fedavg_vs_rgpa_same_data(self):
        fedavg = FedAvgAggregator()
        rgpa = RegularizedGlobalAggregator(lambda_reg=0.0)
        
        client_states = [
            {'param1': torch.tensor([1.0, 2.0])},
            {'param1': torch.tensor([3.0, 4.0])}
        ]
        client_data_sizes = [100, 100]
        
        fedavg_result = fedavg.aggregate(client_states, client_data_sizes)
        rgpa_result = rgpa.aggregate(client_states, client_data_sizes)
        
        torch.testing.assert_close(fedavg_result['param1'], rgpa_result['param1'])
    
    def test_multiple_parameters(self):
        aggregator = RegularizedGlobalAggregator(lambda_reg=0.1)
        
        client_states = [
            {
                'layer1.weight': torch.randn(10, 5),
                'layer1.bias': torch.randn(10),
                'layer2.weight': torch.randn(5, 3)
            },
            {
                'layer1.weight': torch.randn(10, 5),
                'layer1.bias': torch.randn(10),
                'layer2.weight': torch.randn(5, 3)
            }
        ]
        client_data_sizes = [100, 200]
        
        global_state = aggregator.aggregate(client_states, client_data_sizes)
        
        self.assertEqual(len(global_state), 3)
        self.assertIn('layer1.weight', global_state)
        self.assertIn('layer1.bias', global_state)
        self.assertIn('layer2.weight', global_state)


if __name__ == '__main__':
    unittest.main()
