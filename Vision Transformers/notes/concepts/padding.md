---
title: Padding
slug: padding
tags: ['vt/concept', 'vt/deep-learning', 'vt/cnn', 'vt/spatial-geometry']
introduced: "[[w1d1|W1D1]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Padding

Padding is the process of appending symmetrically structured artificial boundary pixels—typically initialized to zero (zero-padding)—around the outer perimeter of an input spatial tensor before executing a convolutional kernel operation. For an input of spatial dimension $H \times W$, applying symmetric zero-padding $P$ on all four margins expands the effective tensor dimensions to $(H + 2P) \times (W + 2P)$.

Given a kernel of size $K$ and stride $S$, the output spatial dimension $O$ is governed by the standard formula:
$$O = \left\lfloor \frac{H - K + 2P}{S} \right\rfloor + 1$$

Padding resolves two critical issues in convolutional architectures:
1. **Boundary Shrinkage Prevention:** In unpadded ("valid") convolutions ($P = 0$), each layer reduces the spatial grid by $K - 1$ pixels. In deep networks with hundreds of layers, spatial dimensions would prematurely shrink to $1 \times 1$ without padding. Setting $P = \frac{K - 1}{2}$ for odd kernels with stride $S = 1$ ensures "same" padding, where output spatial dimensions exactly equal input dimensions ($O = H$).
2. **Border Information Loss:** Pixels at the corners and edges of an unpadded image are only sampled once or twice by a sliding filter, whereas central pixels are sampled $K^2$ times. Padding ensures border pixels participate equally in feature representation.

**Why it matters here:** While CNNs require meticulous spatial margin bookkeeping via padding parameters ($P=1, P=2$) to preserve spatial resolution, Vision Transformers avoid boundary zero-padding artifacts entirely by tokenizing the image into non-overlapping discrete patches.

**Appears in:** [[w1d1|W1D1]] · [[week-1-report|Week 1]]

## Related
- [[convolution]]
- [[pooling]]
- [[feature-map]]
- [[patch-projection]]
