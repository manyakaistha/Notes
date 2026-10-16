---
title: Convolution
slug: convolution
tags: ['vt/concept', 'vt/deep-learning', 'vt/cnn', 'vt/filtering']
introduced: "[[w1d1|W1D1]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Convolution

In deep learning and computer vision, a discrete 2D spatial convolution is a linear filtering operation that slides a learnable tensor kernel $\mathbf{K} \in \mathbb{R}^{C_{in} \times K_h \times K_w}$ across an input feature tensor $\mathbf{X} \in \mathbb{R}^{C_{in} \times H \times W}$ to yield an output activation map $\mathbf{Y} \in \mathbb{R}^{C_{out} \times H' \times W'}$:
$$Y_{c_{out}, i, j} = b_{c_{out}} + \sum_{c_{in}=1}^{C_{in}} \sum_{m=-k}^{k} \sum_{n=-k}^{k} X_{c_{in}, i \cdot s + m, j \cdot s + n} \, K_{c_{out}, c_{in}, m, n}$$
where $s$ denotes the stride, $b$ is a learnable bias scalar, and $(K_h, K_w) = (2k+1, 2k+1)$ denotes the spatial kernel support.

The convolutional operator embeds two profound inductive biases:
1. **Local Connectivity (Sparse Interactions):** Each output neuron calculates a dot product over a localized spatial window of size $K_h \times K_w$, exploiting the physical prior that natural image pixels in close proximity share mutual informational relevance.
2. **Weight Sharing (Parameter Tying):** The identical kernel weights $\mathbf{K}$ are evaluated across every spatial coordinate $(i, j)$, enforcing **translation equivariance**—a visual pattern (such as an edge, texture, or motif) produces an identical localized activation regardless of its coordinate location in the visual field.

**Why it matters here:** While convolution radically reduces parameter complexity from $\mathcal{O}(H \cdot W \cdot D)$ to $\mathcal{O}(K^2 \cdot C_{in} \cdot C_{out})$ and enabled the modern deep vision revolution (AlexNet, VGG, ResNet), its strictly local spatial support creates an architectural bottleneck when capturing global, long-range dependencies across distant regions of an image.

**Appears in:** [[w1d1|W1D1]] · [[week-1-report|Week 1]]

## Related
- [[feature-map]]
- [[padding]]
- [[pooling]]
- [[receptive-field]]
- [[translation-invariance]]
- [[long-range-dependency]]
