---
title: Vision Transformer (ViT)
slug: vision-transformer
tags: ['vt/concept', 'vt/deep-learning', 'vt/transformer', 'vt/architecture']
introduced: "[[w1d1|W1D1]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Vision Transformer (ViT)

The Vision Transformer (ViT), introduced by Dosovitskiy et al. (ICLR 2021) in the seminal paper *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*, is a pure Transformer architecture applied directly to sequences of flattened image patches for computer vision tasks without convolutional inductive biases.

### The Canonical Pipeline
1. **Patch Extraction & Flattening:** A 2D image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ is partitioned into a sequence of $N = \frac{HW}{P^2}$ non-overlapping spatial patches of size $P \times P \times C$. Each patch is flattened into $\mathbf{x}_p^i \in \mathbb{R}^{P^2 C}$.
2. **Linear Projection (Patch Embedding):** Patches are linearly projected to a latent model dimension $D$ via a learnable weight matrix $\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$.
3. **Class Token Prepending:** A learnable classification token $\mathbf{x}_{class} \in \mathbb{R}^D$ is prepended to the sequence at index 0, yielding a sequence of length $N + 1$.
4. **1D Positional Embeddings:** Learnable spatial position embeddings $\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$ are element-wise added to retain 2D spatial arrangement.
5. **Transformer Encoder:** The resulting sequence $\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{x}_p^1 \mathbf{E}; \dots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}$ passes through $L$ identical Transformer encoder blocks composed of Pre-Layer Normalization, Multi-Head Self-Attention (MSA), Residual Connections, and Multi-Layer Perceptrons (MLP with GELU non-linearities):
   $$\mathbf{z}'_l = \operatorname{MSA}(\operatorname{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}$$
   $$\mathbf{z}_l = \operatorname{MLP}(\operatorname{LN}(\mathbf{z}'_l)) + \mathbf{z}'_l$$
6. **MLP Classification Head:** A final layer normalizes the output token at index 0 ($\mathbf{z}_L^0$), feeding it into an MLP head to output classification class logits $\hat{\mathbf{y}}$.

### Core Empirical Findings
ViT lacks the hardcoded translation equivariance and local receptive field priors of CNNs. While this causes ViT to underperform slightly on small datasets without strong regularization, when pretrained on massive datasets (ImageNet-21k, JFT-300M), ViT achieves state-of-the-art accuracy while requiring substantially fewer computational resources to train (FLOPs).

**Why it matters here:** ViT represents the central paradigm of this entire track, proving that general-purpose attention-based sequence models can surpass specialized convolutional architectures in computer vision.

**Appears in:** [[w1d1|W1D1]] · [[week-1-report|Week 1]]

## Related
- [[patch-projection]]
- [[class-token]]
- [[positional-embedding]]
- [[long-range-dependency]]
- [[convolution]]
