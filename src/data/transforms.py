import torchvision.transforms as T


def get_transforms(dataset_name: str = "imagenet100"):
    """
    return the transformer corresponding to the dataset name.
    
    Args:
        dataset_name: Used for potential future dataset-specific logic

    Returns:
        torchvision.transforms.Compose pipeline
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Future extension point: override per dataset if needed
    if dataset_name.lower() in ["imagenet100", "imagenet1k"]:
        pass  # use ImageNet stats
    elif dataset_name.lower() == "cub200":
        # CUB-200 usually uses the same stats, but you can override here later
        pass
    else:
        print(f"Warning: Unknown dataset '{dataset_name}', using ImageNet stats.")

    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])