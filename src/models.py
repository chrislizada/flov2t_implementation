import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: int = 8):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

class LoRALinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, rank: int = 4, alpha: int = 8):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )
        
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)

class ViTWithLoRA(nn.Module):
    def __init__(self, num_classes: int = 8, rank: int = 4, alpha: int = 8, pretrained: bool = True):
        super().__init__()
        
        if pretrained:
            try:
                self.vit = ViTModel.from_pretrained('WinKawaks/vit-tiny-patch16-224')
            except:
                try:
                    self.vit = ViTModel.from_pretrained('facebook/vit-tiny-patch16-224')
                except:
                    print("Warning: Could not load pretrained ViT-tiny, using custom tiny config")
                    config = ViTConfig(
                        hidden_size=192,
                        num_hidden_layers=12,
                        num_attention_heads=3,
                        intermediate_size=768,
                        image_size=224,
                        patch_size=16,
                        num_channels=3
                    )
                    self.vit = ViTModel(config)
        else:
            config = ViTConfig(
                hidden_size=192,
                num_hidden_layers=12,
                num_attention_heads=3,
                intermediate_size=768,
                image_size=224,
                patch_size=16,
                num_channels=3
            )
            self.vit = ViTModel(config)
        
        for param in self.vit.parameters():
            param.requires_grad = False
        
        self._add_lora_to_model(rank, alpha)
        
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        
    def _add_lora_to_model(self, rank: int, alpha: int):
        lora_target_modules = ['query', 'key', 'value', 'dense', 'intermediate.dense', 'output.dense']
        
        for name, module in self.vit.named_modules():
            if isinstance(module, nn.Linear):
                should_apply_lora = any(target in name for target in lora_target_modules)
                
                if should_apply_lora:
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    parent = self.vit
                    for part in parent_name.split('.'):
                        if part:
                            parent = getattr(parent, part)
                    
                    lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                    setattr(parent, child_name, lora_layer)
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0]
        logits = self.classifier(cls_output)
        return logits
    
    def get_lora_parameters(self):
        lora_params = []
        for name, param in self.named_parameters():
            if 'lora' in name or 'classifier' in name:
                lora_params.append(param)
        return lora_params
    
    def get_lora_state_dict(self):
        lora_state = {}
        for name, param in self.named_parameters():
            if 'lora' in name or 'classifier' in name:
                lora_state[name] = param.data.clone()
        return lora_state
    
    def load_lora_state_dict(self, state_dict):
        for name, param in self.named_parameters():
            if name in state_dict:
                param.data.copy_(state_dict[name])

class FLoV2TModel(nn.Module):
    def __init__(self, num_classes: int = 8, rank: int = 4, alpha: int = 8):
        super().__init__()
        self.model = ViTWithLoRA(num_classes=num_classes, rank=rank, alpha=alpha)
    
    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        if x.shape[1] != 3:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            else:
                raise ValueError(f"Expected 1 or 3 channels, got {x.shape[1]}")
        
        return self.model(x)
    
    def get_trainable_parameters(self):
        return self.model.get_lora_parameters()
    
    def get_lora_state_dict(self):
        return self.model.get_lora_state_dict()
    
    def load_lora_state_dict(self, state_dict):
        self.model.load_lora_state_dict(state_dict)
