---
title: Multi-Head Attention (MHA)
slug: multi-head-attention
tags: ['vt/concept', 'vt/deep-learning', 'vt/attention', 'vt/transformer']
introduced: "[[w1d1|W1D1]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Multi-Head Attention (MHA)

Multi-Head Attention (MHA) extends the self-attention mechanism by linearly projecting the queries, keys, and values $h$ times with different, learned linear projections to dimensions $d_k, d_k,$ and $d_v$ respectively. This allows the model to jointly attend to information from different representation subspaces at different positions.

### 1. Mathematical Formulation
$$\operatorname{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{Concat}(\operatorname{head}_1, \dots, \operatorname{head}_h) \mathbf{W}^O$$
$$\text{where } \operatorname{head}_i = \operatorname{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$
where projections are parameter matrices:
- $\mathbf{W}_i^Q \in \mathbb{R}^{D \times d_k}$
- $\mathbf{W}_i^K \in \mathbb{R}^{D \times d_k}$
- $\mathbf{W}_i^V \in \mathbb{R}^{D \times d_v}$
- $\mathbf{W}^O \in \mathbb{R}^{h d_v \times D}$

In standard Vision Transformers (e.g., ViT-Base with $D = 768$ and $h = 12$), the head dimension is set to $d_k = d_v = D / h = 768 / 12 = 64$. Due to the reduced dimension of each head, the total computational cost is identical to single-head attention with full dimensionality.

### 2. Efficient Implementation via Weight Splitting
In production frameworks (such as PyTorch), iterating through each attention head in a Python loop is computationally inefficient. Instead, modern implementations employ **weight splitting**:
1. Input sequence $\mathbf{X} \in \mathbb{R}^{B \times N \times D}$ is projected by a single combined weight matrix $\mathbf{W}_{qkv} \in \mathbb{R}^{D \times 3D}$, yielding a tensor of shape $[B, N, 3, h, d_k]$.
2. The tensor is reshaped and permuted to shape $[3, B, h, N, d_k]$, isolating $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ each of shape $[B, h, N, d_k]$.
3. Attention maps are evaluated across all heads simultaneously using batch matrix multiplication:
   $$\mathbf{A} = \operatorname{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times h \times N \times N}$$
4. The output $[B, h, N, d_k]$ is transposed to $[B, N, h, d_k]$, flattened back to $[B, N, D]$, and projected through $\mathbf{W}^O \in \mathbb{R}^{D \times D}$.

**Why it matters here:** In computer vision, distinct attention heads specialize in distinct visual relationships: some heads attend locally to adjacent patch contours (mimicking classical convolutional edge detectors), while other heads attend globally across distant objects (synthesizing scene semantics).

**Appears in:** [[w1d1|W1D1]] · [[week-1-report|Week 1]]

## Related
- [[self-attention]]
- [[vision-transformer]]
- [[patch-projection]]
