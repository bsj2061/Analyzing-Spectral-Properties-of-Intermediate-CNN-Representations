import torchvision.transforms as T


def get_transforms(dataset_name: str = "imagenet100"):
    """
    ImageNet pretrained 모델에 맞는 표준 transforms
    """
    if dataset_name == "imagenet100":
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])