import torch
import warnings

def get_device(requested: str = "cuda") -> torch.device:
    """
    Resolve torch device with safe fallback.

    Args:
        requested (str): 'cuda' or 'cpu'

    Returns:
        torch.device
    """
    requested = requested.lower()

    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            warnings.warn(
                "CUDA requested but not available. Falling back to CPU."
            )
            return torch.device("cpu")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown device type: {requested}")