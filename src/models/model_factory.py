# model_factory.py
import torch
import torch.nn as nn
from typing import Optional, Literal, Union

from .backbones import get_backbone
from .heads import get_head, MLPHead, LinearHead, ConvHead


class ClassificationModel(nn.Module):
    """
    Backbone + Head를 조합한 end-to-end classification 모델
    
    대부분의 downstream task에서 이 클래스를 직접 import해서 씁니다.
    """
    def __init__(
        self,
        backbone_name: str,
        head_name: str = "linear",
        num_classes: int = 100,
        pretrained: bool = True,
        include_head = True,
        # backbone 관련 추가 옵션 (필요할 때만)
        backbone_kwargs: Optional[dict] = None,
        # head 관련 추가 옵션
        head_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.head_name = head_name.lower()
        self.num_classes = num_classes
        
        backbone_kwargs = backbone_kwargs or {}
        head_kwargs = head_kwargs or {}
        if include_head:
            self.backbone = get_backbone(
                name=self.backbone_name,
                pretrained=pretrained,
                include_head=True,
                **backbone_kwargs
            )
        
        # 2. Feature dimension 자동 추출 (가장 중요한 부분!)
        self.feature_dim = self._get_feature_dim()
        self.head = get_head(self.head_name, num_classes=num_classes, **head_kwargs)
        # 3. Head 생성
        if self.head_name in ("linear", "linearhead"):
            self.head = LinearHead(
                in_dim=self.feature_dim,
                num_classes=num_classes,
                **head_kwargs
            )
            
        elif self.head_name in ("mlp", "mlphead"):
            self.head = MLPHead(
                in_dim=self.feature_dim,
                num_classes=num_classes,
                **head_kwargs
            )
            
        elif self.head_name in ("conv", "convhead", "conv1x1"):
            # ConvHead는 채널 수(in_channels)를 알아야 함
            if "in_channels" not in head_kwargs:
                head_kwargs["in_channels"] = self.feature_dim   # 보통 feature_dim == C
            self.head = ConvHead(
                num_classes=num_classes,
                **head_kwargs
            )
            
        else:
            raise ValueError(f"Unsupported head: {head_name}")
        
        # forward에서 사용할 hook 위치 (ViT vs CNN 구분용 플래그)
        self.is_vit_like = "vit" in self.backbone_name or "swin" in self.backbone_name

    def _get_feature_dim(self) -> int:
        """더미 입력으로 forward 해서 feature dimension 알아내기"""
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self.backbone(dummy)
            
            # ViT 계열은 [B, N+1, D] 또는 [B, D] 형태 → 마지막 차원
            if isinstance(feat, tuple):
                feat = feat[-1]
            if feat.ndim == 3:          # ViT, Swin 등 [B, L, D]
                feat = feat[:, -1]      # cls token or global
            elif feat.ndim == 4:        # CNN [B, C, H, W]
                feat = feat.mean(dim=[2,3])  # GAP처럼
            else:
                pass
                
            return feat.shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        
        # ViT-like → cls token or last token 사용
        if self.is_vit_like and features.ndim == 3:
            features = features[:, 0]   # cls token (보통 index 0)
        elif features.ndim == 4:
            # CNN 계열 → GAP (head에서 처리하는 경우도 많음)
            features = features.mean(dim=[2, 3])
            
        logits = self.head(features)
        return logits


def create_model(
    backbone: str,
    head: str = "linear",
    num_classes: int = 1000,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    편의 함수 (가장 많이 호출되는 형태)
    
    Usage:
        model = create_model("resnet18", "mlp", num_classes=10)
        model = create_model("vit_b16", "linear", num_classes=200, pretrained=False)
    """
    return ClassificationModel(
        backbone_name=backbone,
        head_name=head,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )