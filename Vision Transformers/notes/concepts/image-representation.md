---
title: Digital Image Representation
slug: image-representation
tags: ['vt/concept', 'vt/computer-vision', 'vt/data-representation']
introduced: "[[w1d1|W1D1]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Digital Image Representation

A digital color image is mathematically formulated as a third-order tensor $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$, where $H$ denotes spatial height, $W$ denotes spatial width, and $C$ represents the spectral channel dimension (typically $C = 3$ for Red, Green, and Blue channels). Each discrete coordinate $(i, j, c)$ holds a quantized pixel intensity, standardly represented as an 8-bit unsigned integer taking values in $\{0, 1, \dots, 255\}$, or normalized as a real-valued scalar $\in [0.0, 1.0]$ or standardized to zero mean and unit variance.

Digital images exhibit two fundamental physical properties:
1. **Local Spatial Correlation:** Neighboring pixels exhibit high statistical covariance and share semantic continuity (edges, textures, surfaces).
2. **Channel Coupling:** Multi-spectral channels capture coincident light reflections across complementary optical wavelengths at identical spatial coordinates.

When feeding raw images into classical machine learning or multi-layer perceptrons (MLPs), flattening the tensor into a 1D vector $\mathbf{x}_{flat} \in \mathbb{R}^{HWC}$ completely obliterates 2D neighborhood topology and requires an astronomical number of parameters ($\mathcal{O}(H \cdot W \cdot C \cdot D)$) for the first linear projection, motivating structured spatial operators.

**Why it matters here:** Understanding the tensor structure $[C, H, W]$ and the discrete spatial lattice is the prerequisite for designing both localized convolutional feature extractors and patch-based tokenization sequences in Vision Transformers.

**Appears in:** [[w1d1|W1D1]] · [[week-1-report|Week 1]]

## Related
- [[feature-map]]
- [[convolution]]
- [[patch-projection]]
