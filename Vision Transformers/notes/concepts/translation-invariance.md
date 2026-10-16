---
title: Translation Invariance & Equivariance
slug: translation-invariance
tags: ['vt/concept', 'vt/deep-learning', 'vt/cnn', 'vt/inductive-bias']
introduced: "[[w1d1|W1D1]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Translation Invariance & Equivariance

In computer vision and deep representation learning, **translation equivariance** and **translation invariance** represent two distinct mathematical symmetries:

1. **Translation Equivariance (Convolutions):** A transformation $f$ is equivariant to spatial translation $T_g$ if translating the input leads to an identically translated output feature map:
   $$f(T_g(\mathbf{x})) = T_g(f(\mathbf{x}))$$
   Discrete 2D convolution naturally satisfies translation equivariance due to spatial weight sharing. If an object (such as a car or a face) shifts by $(\Delta x, \Delta y)$ pixels in the input image, the corresponding activation peak in the output feature map shifts by exactly the same coordinates (scaled by stride $s$).
2. **Translation Invariance (Pooling & Global Embeddings):** A transformation $g$ is invariant to spatial translation if shifting the input produces an identical output representation:
   $$g(T_g(\mathbf{x})) = g(\mathbf{x})$$
   In classical CNNs, translation invariance is progressively achieved by combining spatial downsampling (Max/Average pooling) with final global pooling layers, ensuring that image-level classification predictions remain robust to shifting object positions.

### The Spatial Agnosticism Dilemma
While translation equivariance and invariance are powerful inductive biases for basic object classification, they impose severe constraints on complex spatial reasoning:
- **Loss of Absolute and Relative Coordinates:** Because convolutional kernels apply identical weights regardless of spatial position, a filter trained to detect cars will activate with equal magnitude whether the car is situated on a paved street or perched absurdly atop a skyscraper roof.
- **Inability to Model Distant Geometric Configurations:** Classical convolutions cannot natively distinguish between valid holistic spatial structures and scrambled, locally intact image patches without extensive receptive field stacking.

**Why it matters here:** Vision Transformers replace hardcoded translation equivariance with explicit, learnable positional embeddings ($\mathbf{E}_{pos}$), granting the architecture the flexibility to dynamically weigh both translation-invariant semantic concepts and coordinate-dependent spatial configurations.

**Appears in:** [[w1d1|W1D1]] · [[week-1-report|Week 1]]

## Related
- [[convolution]]
- [[pooling]]
- [[positional-embedding]]
- [[vision-transformer]]
