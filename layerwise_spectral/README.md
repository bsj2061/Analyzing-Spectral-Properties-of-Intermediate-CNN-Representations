# Layerwise Spectral Behavior

This experiment analyzes how frequency components evolve across intermediate layers in vision backbones (ResNet and ViT).  
It examines whether representations collapse toward low frequencies with increasing depth, and whether this behavior differs significantly between architectures, using direct frequency-band interventions.

## Setup
- **Backbones**: ResNet-18/50 (ReLU activation), ViT-B/16 (GELU activation)
- **Datasets**: ImageNet (using pretrained frozen backbones)
- **Layers analyzed**:
  - ResNet: layer1 ~ layer4
  - ViT: stage-wise (e.g., after each transformer block)
- **Frequency bands** (radial decomposition): To enable comparison of frequency bands across images of varying sizes (resolutions), we normalize the frequency domain such that the radial frequency ranges from 0 to 1 (where 1 corresponds to the Nyquist frequency).
  - < 0.05
  - 0.05 ~ 0.15
  - 0.15 ~ 0.3
  - 0.3 ~ 0.5
  - 0.5 ~ 0.7
  - 0.7 ~ 0.9
  - > 0.9 

## Method
1. Extract intermediate feature maps from the selected layer.
2. Apply 2D Fast Fourier Transform (FFT).
3. Apply radial band masks to:
   - remove (suppress) a specific band
   - isolate (keep only) a specific band
   - perturb selected frequencies
4. Perform inverse FFT to reconstruct the modified spatial-domain feature map.
5. Feed the reconstructed features into a linear classifier for evaluation.

**Evaluation metrics**:
- Top-1 classification accuracy drop
- Mean ground-truth (GT) rank degradation

## Key Findings
- **ResNet**: Representations collapse rapidly toward ultra-low frequencies as depth increases.  
  In deeper layers (especially layer4), removing high-frequency bands causes accuracy to drop close to 0.
- **ViT**: Much less pronounced collapse. Frequency information is preserved in a more band-invariant manner across layers.
- **Early layers**: Show relatively lower sensitivity to high frequencies, but lack strong frequency-specific specialization.

These results highlight strong architecture-level inductive biases in how spectral information is processed and preserved.

## Scripts
- `run_layerwise_intervention.py` : Run intervention on a single layer and band
- `batch_all_layers.py` : Batch process all layers for comparison
- `plot_layerwise_spectrum.py` : Visualize frequency energy distribution across layers

## Example Usage
```bash
python run_layerwise_intervention.py \
  --model resnet18 \
  --layer layer4 \
  --band ultra-low \
  --mode remove \
  --dataset imagenet-val
