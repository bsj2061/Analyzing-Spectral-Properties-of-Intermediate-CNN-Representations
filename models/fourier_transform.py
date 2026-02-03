import torch
import torch.nn as nn


def radial_frequency_mask(
    h: int,
    w: int,
    r_min: float,
    r_max: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a radial (annular) frequency mask in the Fourier domain.

    This function generates a boolean mask where pixels within the normalized
    radial distance range [r_min, r_max) are kept (True), and all others are
    masked out (False). The center (DC component) is at (h//2, w//2).

    Args:
        h (int): Height of the feature map
        w (int): Width of the feature map
        r_min (float): Lower bound of the normalized radius (0.0 ~ 1.0)
        r_max (float): Upper bound of the normalized radius (r_min < r_max ≤ 1.0)
        device (torch.device): Device where the mask should be created

    Returns:
        torch.Tensor: Boolean mask of shape (h, w)
                      True for frequencies to keep, False for frequencies to remove
    """
    # Create coordinate grids (y increases downward, x increases rightward)
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )

    # Center of the frequency domain
    cy, cx = h // 2, w // 2

    # Euclidean distance from center
    dist = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)

    # Normalize distances so that the maximum radius becomes 1.0
    max_dist = dist.max()
    if max_dist > 0:
        dist = dist / max_dist

    # Create band-pass mask: keep only frequencies in [r_min, r_max)
    mask = (dist >= r_min) & (dist < r_max)

    return mask


class FourierBandTransform(nn.Module):
    """
    PyTorch module that applies band-pass filtering in the frequency domain.

    This module isolates a specific radial frequency band by:
      1. Computing the 2D FFT of the input
      2. Shifting zero-frequency to the center
      3. Applying a radial annular mask
      4. Inverse FFT and taking the real part

    Useful for frequency-domain analysis of feature maps (e.g., low-pass, high-pass,
    or band-pass filtering of intermediate representations in CNNs / ViTs).

    Args:
        r_min (float): Lower normalized radius of the frequency band to keep
        r_max (float): Upper normalized radius of the frequency band to keep

    Input shape:  (B, C, H, W) - typically feature maps
    Output shape: (B, C, H, W) - reconstructed spatial features (real part only)
    """

    def __init__(self, r_min: float, r_max: float):
        super().__init__()
        self.r_min = float(r_min)
        self.r_max = float(r_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: apply band-pass filter in frequency domain.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Filtered output of same shape (real part of inverse FFT)
        """
        B, C, H, W = x.shape

        # Step 1: Compute 2D FFT and center the zero-frequency component
        fft = torch.fft.fftshift(
            torch.fft.fft2(x, dim=(-2, -1)),
            dim=(-2, -1)
        )

        # Step 2: Generate the radial mask for the current spatial size
        mask = radial_frequency_mask(H, W, self.r_min, self.r_max, x.device)

        # Step 3: Apply mask (broadcast mask to (B, C, H, W))
        fft_masked = fft * mask[None, None, :, :]

        # Step 4: Inverse FFT → shift back → take real part (imag part should be near zero)
        x_rec = torch.fft.ifft2(
            torch.fft.ifftshift(fft_masked, dim=(-2, -1)),
            dim=(-2, -1)
        ).real

        return x_rec
