---
title: Patch Projection
slug: patch-projection
tags: ['vt/concept', 'vt/deep-learning', 'vt/transformer', 'vt/tokenization']
introduced: "[[w1d1|W1D1]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Patch Projection

Patch projection is the tokenization mechanism by which a continuous 2D spatial image tensor is converted into a 1D discrete sequence of embedding vectors suitable for input into a standard Transformer encoder.

Given an input image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ and a chosen square patch resolution $P \times P$ (typically $P = 16$ or $P = 32$), the image is divided into a regular grid of non-overlapping rectangular patches:
$$N = \frac{H \cdot W}{P^2}$$
Each individual spatial patch $\mathbf{x}_p^{(i)} \in \mathbb{R}^{P \times P \times C}$ is unrolled into a flattened vector $\mathbf{x}_p^{(i)} \in \mathbb{R}^{P^2 C}$. For standard $224 \times 224 \times 3$ images with $P = 16$, the image yields $N = 14 \times 14 = 196$ patches, each of dimensionality $16 \times 16 \times 3 = 768$.

To project each flattened patch vector into the Transformer's hidden latent dimension $D$ (e.g., $D = 768$ in ViT-Base), the flattened vectors are multiplied by a learnable linear projection matrix $\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$:
$$\mathbf{z}_0^{(i)} = \mathbf{x}_p^{(i)} \mathbf{E}$$

### Efficient PyTorch Formulation
Computationally, patch extraction and linear projection are executed in a single hardware-accelerated step using a 2D convolution:
```python
# Equivalent single-layer implementation:
patch_embed = nn.Conv2d(in_channels=C, out_channels=D, kernel_size=P, stride=P)
# Output tensor: [Batch, D, H/P, W/P] -> flattened to [Batch, N, D]
```
Because `kernel_size == stride == P`, the convolution slides across adjacent non-overlapping spatial blocks and computes the linear projection without extracting explicit slice views in memory.

**Why it matters here:** Patch projection serves as the entry port into the Vision Transformer, bridging the continuous 2D topology of computer vision with sequence-based attention architectures.

**Appears in:** [[w1d1|W1D1]] · [[week-1-report|Week 1]]

## Related
- [[vision-transformer]]
- [[positional-embedding]]
- [[class-token]]
- [[image-representation]]
