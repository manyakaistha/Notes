---
title: Vision Transformers Knowledge Map
tags: [vt/moc]
---

# Vision Transformers — Knowledge Map

## The 4-Week Spine

| Week | Report | Core Question | Status |
| :--- | :--- | :--- | :--- |
| **1** | [[week-1-report\|Week 1]] | Why do we need Vision Transformers when CNNs revolutionized computer vision? | in-progress |
| **2** | [[week-2-report\|Week 2]] | How does Multi-Head Self-Attention capture spatial dependencies without convolution? | planned |
| **3** | [[week-3-report\|Week 3]] | How do Vision Transformers overcome the lack of inductive bias at scale? | planned |
| **4** | [[week-4-report\|Week 4]] | How do hierarchical and hybrid vision architectures resolve quadratic complexity for dense prediction? | planned |

---

### Week 1 Daily Notes
- [[w1d1|W1D1]]: Motivation & Foundations — Computer Vision in Practice, AlexNet Watershed, CNN Limitations, and the ViT Paradigm
- [[w1d2|W1D2]]: The Mathematics of 2D Patch Projection and Linear Embeddings
- [[w1d3|W1D3]]: Self-Attention in 2D Space: Direct Pairwise Routing vs. Convolutional Kernels
- [[w1d4|W1D4]]: Positional Encoding Paradigms: 1D, 2D, and Relative Biases
- [[w1d5|W1D5]]: Full ViT Architecture Assembly & Empirical PyTorch Verification

---

## Concept Spine

- **Data & Spatial Representation** — [[image-representation]] · [[feature-map]] · [[patch-projection]] · [[positional-embedding]] · [[class-token]]
- **Convolutional Mechanics & Inductive Biases** — [[convolution]] · [[padding]] · [[pooling]] · [[receptive-field]] · [[translation-invariance]] · [[vanishing-gradients]]
- **Self-Attention & Transformer Architecture** — [[self-attention]] · [[multi-head-attention]] · [[layer-normalization]] · [[residual-connection]] · [[long-range-dependency]] · [[vision-transformer]]

---

## Central Paradigm

$$\text{Raw Image } \mathbf{x} \in \mathbb{R}^{H \times W \times C} \xrightarrow{\text{Patch Partition}} \mathbf{x}_p \in \mathbb{R}^{N \times (P^2 C)} \xrightarrow{\text{Linear Projection} + \mathbf{E}_{pos}} \mathbf{z}_0 \in \mathbb{R}^{(N+1) \times D} \xrightarrow{\text{Transformer Encoder}} \mathbf{z}_L^0 \xrightarrow{\text{MLP Head}} \hat{\mathbf{y}}$$

Defined on Day 1 ([[w1d1|W1D1]]) through the deconstruction of classical computer vision (from handcrafted morphology to AlexNet 2012) and the introduction of the Vision Transformer (ViT) paradigm by Dosovitskiy et al. (ICLR 2021). The foundational transition replaces hardcoded localized receptive fields and translation equivariance with flexible, dynamic pairwise self-attention routing across non-overlapping patches.

---

## The Handoff Across Foundations

Day 1 grounds the structural limitations of classical convolutional filters (local receptive field constraints, translation invariance, gradient decay in deep stacking, and inability to natively model long-range context across distant visual tokens), formalizes scaled dot-product attention and Transformer block modularity, and presents the canonical ViT architecture. Day 2 expands into the exact mathematical tensor mechanics of 2D patch projection and embedding spaces.
