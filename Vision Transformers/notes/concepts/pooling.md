---
title: Pooling
slug: pooling
tags: ['vt/concept', 'vt/deep-learning', 'vt/cnn', 'vt/downsampling']
introduced: "[[w1d1|W1D1]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Pooling

Pooling is a non-parametric spatial downsampling operation that partitions a feature map into non-overlapping or strided spatial windows of size $K_p \times K_p$ (commonly $2 \times 2$ with stride $S = 2$) and computes a localized summary statistic across each window:

1. **Max Pooling:** Computes the maximum activation value in each quadrant:
   $$Y_{i, j} = \max_{m, n \in [0, K_p - 1]} X_{i \cdot S + m, j \cdot S + n}$$
   Max pooling acts as a dominant feature detector, preserving the most salient edges, gradients, and textural activations while providing minor translation invariance to small local shifts.
2. **Average Pooling:** Computes the arithmetic mean over the pooling window:
   $$Y_{i, j} = \frac{1}{K_p^2} \sum_{m=0}^{K_p - 1} \sum_{n=0}^{K_p - 1} X_{i \cdot S + m, j \cdot S + n}$$
   Average pooling provides spatial smoothing and aggregates regional background energy.
3. **Sum Pooling:** Accumulates the total unnormalized sum of activations across the window:
   $$Y_{i, j} = \sum_{m=0}^{K_p - 1} \sum_{n=0}^{K_p - 1} X_{i \cdot S + m, j \cdot S + n}$$

Pooling accomplishes three major architectural goals in CNNs:
- **Dimensionality Reduction:** Reduces spatial resolution by a factor of $S^2$ (e.g., $75\%$ reduction for $2 \times 2, S=2$), cutting computational cost in subsequent layers.
- **Receptive Field Expansion:** Accelerates the rate at which downstream convolutional kernels span larger portions of the input image.
- **Local Invariance:** Slight translations of an object within the pooling window produce identical pooled outputs.

**Why it matters here:** While pooling was indispensable for building hierarchical CNNs, it irreversibly discards fine-grained spatial coordinates and pixel alignments. Vision Transformers preserve exact spatial token associations across all encoder layers using positional embeddings without spatial destruction.

**Appears in:** [[w1d1|W1D1]] · [[week-1-report|Week 1]]

## Related
- [[convolution]]
- [[feature-map]]
- [[receptive-field]]
- [[translation-invariance]]
