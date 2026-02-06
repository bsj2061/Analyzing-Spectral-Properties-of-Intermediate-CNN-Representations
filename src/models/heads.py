import torch
import torch.nn as nn
from typing import Optional

class MLPHead(nn.Module):
    def __init__(self, in_dim=2048, num_classes=100, hidden=512, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, num_classes)]
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 100):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=True)
        
    def forward(self, x):
        return self.fc(x)


class ConvHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def get_head(
    name: str,
    in_dim: Optional[int] = None,       # Linear, MLP
    in_channels: Optional[int] = None,   # ConvHead
    num_classes: int = 100,
) -> nn.Module:
    name = name.upper()
    
    if name in ("LINEAR", "LINEARHEAD"):
        if in_dim is None:
            raise ValueError("LinearHead needs in_dim")
        return LinearHead(in_dim=in_dim, num_classes=num_classes)
        
    elif name in ("MLP", "MLPHEAD"):
        if in_dim is None:
            raise ValueError("MLPHead needs in_dim")
        return MLPHead(in_dim=in_dim, num_classes=num_classes)
        
    elif name in ("CONV", "CONVHEAD"):
        if in_channels is None:
            raise ValueError("ConvHead needs in_channels (C)")
        return ConvHead(in_channels=in_channels, num_classes=num_classes)
        
    else:
        raise ValueError(
            f"Supported heads: Linear, MLP, Conv\n"
            f"Got: {name}\n"
        )