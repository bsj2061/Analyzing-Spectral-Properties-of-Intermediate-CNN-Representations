import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights

def get_resnet18(pretrained: bool = True, include_head:bool = True) -> nn.Module:
    """
    Load a ResNet-18 model, with options to include/exclude the classification head.
    
    Args:
        pretrained: If True, loads ImageNet-1K pretrained weights (DEFAULT weights)
        include_head: If True, returns the complete ResNet-18 (feature extractor + avgpool + fc)
                      If False, return the feature extractor only (up to layer4, excluding avgpool & fc)
                      
    Returns:
        nn.Module: Resnet18 or its backbone
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet18(weights=weights)
    if include_head:    # complete ResNet18 model
        return model
    else:   # only backbone
        return nn.Sequential(*list(model.children())[:-2])


def get_vit_b16(pretrained: bool = True, include_head:bool = True) -> nn.Module:
    """
    Load a ViT model, with options to include/exclude the classification head.
    
    Args:
        pretrained: If True, loads ImageNet-1K pretrained weights (DEFAULT weights)
        include_head: If True, returns the complete ViT 
                      If False, return the backbone only
                      
    Returns:
        nn.Module: ViT or its backbone
    """
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.vit_b_16(weights)
    if include_head:    # complete ResNet18 model
        return model
    else:   # only backbone
        return nn.Sequential(model.conv_proj, model.encoder)


def get_backbone(name: str, pretrained: bool = True, include_head: bool = True) -> nn.Module:
    """
    load the complete model or its backbone by model name.

    Args:
        name: Model identifier (e.g., "resnet18", "vit_b16")
        pretrained:  If True, loads ImageNet-1K pretrained weights (DEFAULT weights)
        include_head: If True, returns the complete model
                      If False, return the backbone only

    Returns:
        nn.Module: The requested model or backbone

    Raises:
        ValueError: If the model name is not supported
    """
    registry = {
        "resnet18": get_resnet18,
        "vit_b16": get_vit_b16,
        # expanding next: "swin_t", "convnext_t", ...
    }
    if name.lower() not in registry:
        raise ValueError(f"Unknown backbone: {name}")
    return registry[name.lower()](pretrained=pretrained)