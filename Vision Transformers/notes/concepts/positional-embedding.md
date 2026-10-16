---
title: Positional Embedding
slug: positional-embedding
tags: ['vt/concept', 'vt/deep-learning', 'vt/transformer', 'vt/position-encoding']
introduced: "[[w1d1|W1D1]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Positional Embedding

Positional embedding is a mathematical mechanism used in Transformer architectures to inject spatial or sequential order into token representations, compensating for the permutation equivariance of the self-attention operation.

### The Permutation Equivariance Problem
Standard Multi-Head Self-Attention treats its inputs as an unindexed multiset (or "bag of tokens"). If the input tokens are shuffled across indices according to an arbitrary permutation matrix $\mathbf{\Pi}$, the output tokens are permuted identically:
$$\operatorname{MSA}(\mathbf{\Pi} \mathbf{X}) = \mathbf{\Pi} \operatorname{MSA}(\mathbf{X})$$
In computer vision, spatial geometry is paramount: swapping the sky patch at the top with a pavement patch at the bottom fundamentally alters the semantic scene. Without positional cues, a pure Transformer cannot distinguish between a coherent image and a scrambled collage of identical patches.

### ViT 1D Learnable Position Embeddings
In the original Vision Transformer (Dosovitskiy et al., 2021), spatial order is supplied by adding a learnable 1D tensor $\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$ directly to the patch projection embeddings prior to the first encoder layer:
$$\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{x}_p^1 \mathbf{E}; \dots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}$$
- **1D vs 2D:** While 2D positional embeddings (explicitly encoding row and column indices separately) were evaluated in the paper, empirical benchmarks revealed that learnable 1D embeddings perform equally well. During pretraining, the 1D position vectors automatically organize themselves into 2D spatial topologies: cosine similarity plots reveal that adjacent patches in the 2D grid learn highly similar position vectors, reflecting both row and column proximity.
- **Interpolation for Variable Resolutions:** If an image of higher resolution is fed at inference time ($N' > N$), the positional embedding grid can be 2D-bicubically interpolated to match the new token sequence length.

**Why it matters here:** Positional embeddings re-introduce spatial coordinate awareness into the model without constraining the receptive field to localized spatial kernels.

**Appears in:** [[w1d1|W1D1]] · [[week-1-report|Week 1]]

## Related
- [[vision-transformer]]
- [[patch-projection]]
- [[class-token]]
- [[translation-invariance]]
