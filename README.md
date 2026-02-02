# Analyzing-Spectral-Properties-of-Intermediate-CNN-Representations
Analyzing layer-wise spectral behavior of vision models via frequency-isolated intermediate representations

## Overview

This repository contains a series of controlled experiments analyzing how **intermediate representations** in vision models behave in the **frequency domain**.

Rather than asking whether neural networks exhibit spectral bias in general, this project focuses on a more concrete question:

> Which frequency components are actually preserved and used for discrimination across layers, and what factors determine this behavior?
> 

To answer this, we directly intervene on the frequency components of intermediate feature maps and evaluate their functional role in classification.

## Motivation

I started from a simple observation:

> Many vision models seem to rely heavily on low-frequency information for classification.
> 

A common explanation attributes this to *spectral bias*—the tendency of neural networks (especially ReLU networks) to learn low-frequency functions earlier or more easily.

However, most discussions of spectral bias focus on:

- training dynamics, or
- the input–output function as a whole.

In this project, I instead ask:

- What happens **inside** the network?
- Do intermediate layers preserve different frequency components?
- If certain frequencies are not used, is it because:
    - the classifier cannot exploit them, or
    - the backbone representation has already discarded them?
are actually necessary for discrimination,
and what factors are responsible for shaping this dependence.

## Experimental Questions

The experiments are organized around the following questions:

1. **Layer-wise behavior**
    
    How does the frequency dependence of representations change with depth?
    
2. **Architecture dependence**
    
    Do CNNs and Transformers exhibit different spectral behaviors?
    
3. **Classifier dependence**
    
    Are high-frequency components unused because linear classifiers cannot exploit them?
    
4. **Dataset dependence**
    
    Does a fine-grained dataset encourage the use of higher-frequency information?
    
5. **Training strategy**
    
    Can joint training with a more expressive head alter the spectral structure of representations?

## Method

### Frequency Intervention on Feature Maps

For a given intermediate feature map:

1. Apply a 2D Fourier transform.
2. Decompose frequencies into radial bands (e.g., ultra-low, low, mid, high).
3. Isolate a specific frequency band by masking out all other bands.
4. Apply inverse FFT to reconstruct the feature map using only the selected band.
5. Feed the reconstructed feature map into a classifier.

Performance is evaluated using:

- classification accuracy
- mean ground-truth (GT) rank
- (Spectral Content of Feature Maps)

This setup allows us to evaluate **which frequency components alone are sufficient for discrimination**.
