"""
Model utilities for Waterbirds B/C experiment V2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
from typing import Dict, List, Tuple
from config.config import *

class ResNetWithHead(nn.Module):
    """ResNet model with replaceable head for B/C experiment"""
    
    def __init__(self, num_classes: int = 2, freeze_backbone: bool = True):
        super(ResNetWithHead, self).__init__()
        
        # 加载 ResNet（兼容新旧 torchvision：新版已移除 pretrained 参数，直接传会 TypeError）
        if PRETRAINED:
            try:
                # IMAGENET1K_V1 与旧版 pretrained=True 是同一套权重，结果完全一致
                from torchvision.models import ResNet50_Weights
                self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            except ImportError:
                # 旧版 torchvision（<0.13）没有 Weights 枚举，退回 pretrained 参数
                self.backbone = models.resnet50(pretrained=True)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final layer with a new head
        self.backbone.fc = nn.Identity()
        
        # Create custom head
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize head weights
        self._initialize_head()
    
    def _initialize_head(self):
        """Initialize head weights"""
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply head
        logits = self.head(features)
        
        return logits
    
    def load_backbone_from_checkpoint(self, checkpoint_path: str):
        """Load only the backbone from a checkpoint

        save_model 保存的是嵌套 dict {'backbone': {...}, 'head': {...}, 'model_state_dict': {...}}；
        旧实现按 'backbone.' 前缀遍历顶层 key 永远匹配不上（顶层 key 是 'backbone'，不带点），
        导致静默加载空 dict、backbone 一直是 ImageNet 初始权重。现直接取 checkpoint['backbone']，
        并兼容旧的扁平格式；strict=True 保证结构对不上时立刻报错而不是静默跳过。
        """
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        if isinstance(checkpoint, dict) and "backbone" in checkpoint:
            # 新格式：save_model 的嵌套 dict
            backbone_state_dict = checkpoint["backbone"]
        else:
            # 兼容旧格式：扁平 state_dict（key 形如 'backbone.xxx'）
            backbone_state_dict = {
                k.replace("backbone.", "", 1): v
                for k, v in checkpoint.items()
                if k.startswith("backbone.")
            }

        # 结构必须完全对上；加载失败立即抛错，不允许静默通过
        self.backbone.load_state_dict(backbone_state_dict)
    
    def load_head_from_checkpoint(self, checkpoint_path: str):
        """Load only the head from a checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Load head state dict
        head_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('head.'):
                new_key = key.replace('head.', '')
                head_state_dict[new_key] = value
        
        # Load head weights
        self.head.load_state_dict(head_state_dict, strict=False)
    
    def save_head_initialization(self, save_path: str):
        """Save head initialization parameters"""
        head_state_dict = self.head.state_dict()
        torch.save(head_state_dict, save_path)
    
    def load_head_initialization(self, load_path: str):
        """Load head initialization parameters"""
        head_state_dict = torch.load(load_path, map_location=DEVICE)
        self.head.load_state_dict(head_state_dict)

def create_model(num_classes: int = 2, freeze_backbone: bool = True) -> ResNetWithHead:
    """Create a new model with frozen backbone"""
    model = ResNetWithHead(num_classes=num_classes, freeze_backbone=freeze_backbone)
    return model.to(DEVICE)

def save_model(model: nn.Module, save_path: str):
    """Save complete model state"""
    torch.save({
        'backbone': model.backbone.state_dict(),
        'head': model.head.state_dict(),
        'model_state_dict': model.state_dict()
    }, save_path)

def load_model(model: nn.Module, checkpoint_path: str, load_head: bool = True):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    if load_head:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Load only backbone
        model.backbone.load_state_dict(checkpoint['backbone'], strict=False)

def get_model_parameters(model: nn.Module, requires_grad: bool = True) -> int:
    """Get number of trainable parameters"""
    total_params = 0
    for param in model.parameters():
        if param.requires_grad == requires_grad:
            total_params += param.numel()
    return total_params

def count_parameters_by_layer(model: nn.Module) -> Dict[str, int]:
    """Count parameters by layer type"""
    counts = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_name = name.split('.')[0]
            if layer_name in counts:
                counts[layer_name] += param.numel()
            else:
                counts[layer_name] = param.numel()
    return counts