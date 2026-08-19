"""
模型定义：ImageNet 预训练 ResNet-18，backbone + 分类 head
"""
import torch
import torch.nn as nn
from torchvision import models


def get_device(requested=None):
    """返回可用设备。requested: None/auto/cuda/cpu"""
    if requested in (None, 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(requested)


def create_model(num_classes=2, pretrained=True, freeze_backbone=False, device='cpu'):
    """创建 ResNet-18 模型并移到指定设备。

    freeze_backbone=True 时冻结所有 backbone 参数，只允许 head 更新。
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    return model


def copy_model_state(model, device='cpu'):
    """深拷贝模型参数到指定设备（默认 CPU，用于保存权重文件）"""
    return {k: v.detach().cpu().clone() if device == 'cpu' else v.detach().clone()
            for k, v in model.state_dict().items()}


def load_state(model, state_dict):
    """加载状态字典，自动处理 CPU/CUDA 设备差异"""
    if state_dict is None:
        return model
    model_device = next(model.parameters()).device
    state = {k: v.to(model_device) for k, v in state_dict.items()}
    model.load_state_dict(state)
    return model
