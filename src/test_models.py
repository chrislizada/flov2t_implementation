import unittest
import torch
import numpy as np
from models import LoRALayer, LoRALinear, ViTWithLoRA, FLoV2TModel


class TestLoRALayer(unittest.TestCase):
    
    def test_lora_layer_initialization(self):
        lora = LoRALayer(in_features=64, out_features=64, rank=4, alpha=8)
        
        self.assertEqual(lora.lora_A.shape, (4, 64))
        self.assertEqual(lora.lora_B.shape, (64, 4))
        self.assertEqual(lora.scaling, 2.0)
    
    def test_lora_layer_forward(self):
        lora = LoRALayer(in_features=64, out_features=64, rank=4, alpha=8)
        
        x = torch.randn(8, 64)
        output = lora(x)
        
        self.assertEqual(output.shape, (8, 64))
    
    def test_lora_layer_rank(self):
        for rank in [2, 4, 8]:
            lora = LoRALayer(in_features=64, out_features=64, rank=rank, alpha=8)
            self.assertEqual(lora.lora_A.shape[0], rank)
            self.assertEqual(lora.lora_B.shape[1], rank)


class TestLoRALinear(unittest.TestCase):
    
    def test_lora_linear_frozen(self):
        linear = torch.nn.Linear(64, 64)
        lora_linear = LoRALinear(linear, rank=4, alpha=8)
        
        for param in lora_linear.linear.parameters():
            self.assertFalse(param.requires_grad)
        
        self.assertTrue(lora_linear.lora.lora_A.requires_grad)
        self.assertTrue(lora_linear.lora.lora_B.requires_grad)
    
    def test_lora_linear_forward(self):
        linear = torch.nn.Linear(64, 64)
        lora_linear = LoRALinear(linear, rank=4, alpha=8)
        
        x = torch.randn(8, 64)
        output = lora_linear(x)
        
        self.assertEqual(output.shape, (8, 64))


class TestViTWithLoRA(unittest.TestCase):
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_vit_initialization(self):
        model = ViTWithLoRA(num_classes=8, rank=4, alpha=8, pretrained=False)
        
        self.assertIsNotNone(model.vit)
        self.assertIsNotNone(model.classifier)
    
    def test_vit_frozen_weights(self):
        model = ViTWithLoRA(num_classes=8, rank=4, alpha=8, pretrained=False)
        
        frozen_count = 0
        trainable_count = 0
        
        for name, param in model.named_parameters():
            if 'lora' not in name and 'classifier' not in name:
                self.assertFalse(param.requires_grad, f"Parameter {name} should be frozen")
                frozen_count += 1
            else:
                trainable_count += 1
        
        self.assertGreater(frozen_count, 0)
        self.assertGreater(trainable_count, 0)
    
    def test_vit_forward(self):
        model = ViTWithLoRA(num_classes=8, rank=4, alpha=8, pretrained=False)
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 8))
    
    def test_lora_state_dict(self):
        model = ViTWithLoRA(num_classes=8, rank=4, alpha=8, pretrained=False)
        
        lora_state = model.get_lora_state_dict()
        
        self.assertGreater(len(lora_state), 0)
        
        for key in lora_state.keys():
            self.assertTrue('lora' in key or 'classifier' in key)


class TestFLoV2TModel(unittest.TestCase):
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_model_initialization(self):
        model = FLoV2TModel(num_classes=8, rank=4, alpha=8)
        
        self.assertIsNotNone(model.model)
    
    def test_model_forward_uint8(self):
        model = FLoV2TModel(num_classes=8, rank=4, alpha=8)
        
        x = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 8))
    
    def test_model_forward_float(self):
        model = FLoV2TModel(num_classes=8, rank=4, alpha=8)
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 8))
    
    def test_model_channel_expansion(self):
        model = FLoV2TModel(num_classes=8, rank=4, alpha=8)
        
        x = torch.randint(0, 256, (2, 1, 224, 224), dtype=torch.uint8)
        output = model(x)
        
        self.assertEqual(output.shape, (2, 8))
    
    def test_trainable_parameters(self):
        model = FLoV2TModel(num_classes=8, rank=4, alpha=8)
        
        trainable = model.get_trainable_parameters()
        
        self.assertGreater(len(trainable), 0)
        
        for param in trainable:
            self.assertTrue(param.requires_grad)
    
    def test_parameter_reduction(self):
        model = FLoV2TModel(num_classes=8, rank=4, alpha=8)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        reduction_ratio = total_params / trainable_params
        
        self.assertGreater(reduction_ratio, 10)
        print(f"\nParameter reduction: {reduction_ratio:.2f}x")
        print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")


if __name__ == '__main__':
    unittest.main()
