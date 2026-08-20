"""ResNet-50 model matching the Waterbirds B/C V2 defaults."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class WaterbirdsResNet(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True) -> None:
        super().__init__()
        if pretrained:
            try:
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                backbone = models.resnet50(weights=weights)
            except AttributeError:  # torchvision < 0.13
                backbone = models.resnet50(pretrained=True)
        else:
            try:
                backbone = models.resnet50(weights=None)
            except TypeError:
                backbone = models.resnet50(pretrained=False)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )
        self._initialize_head()
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def _initialize_head(self) -> None:
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))

    @property
    def backbone_frozen(self) -> bool:
        return not any(parameter.requires_grad for parameter in self.backbone.parameters())


def checkpoint_payload(model: nn.Module) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "backbone_frozen": bool(getattr(model, "backbone_frozen", False)),
        "architecture": "resnet50+mlp_head_2048_512_2",
    }


def save_checkpoint(model: nn.Module, path: str) -> None:
    torch.save(checkpoint_payload(model), path)
