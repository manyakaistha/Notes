---
title: Self-Attention Mechanism
slug: self-attention
tags: ['vt/concept', 'vt/deep-learning', 'vt/attention', 'vt/transformer']
introduced: "[[w1d1|W1D1]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Self-Attention Mechanism

Self-attention is an attention mechanism relating different positions of a single sequence to compute a representation of the sequence. Unlike recurrence (which processes tokens sequentially) or convolutions (which compute local sliding window dot products), self-attention computes direct, content-based pairwise interactions across all tokens in $\mathcal{O}(1)$ computational path length.

### 1. Simplified Dot-Product Attention (Untrained)
Given a sequence of token embedding vectors $\mathbf{x}_1, \dots, \mathbf{x}_N \in \mathbb{R}^D$:
1. **Raw Attention Scores:** The compatibility between query token $\mathbf{x}_i$ and key token $\mathbf{x}_j$ is computed via vector dot product:
   $$s_{i, j} = \mathbf{x}_i \cdot \mathbf{x}_j = \|\mathbf{x}_i\| \|\mathbf{x}_j\| \cos(\theta)$$
2. **Probability Normalization:** Scores are converted into attention weights using the softmax function:
   $$\alpha_{i, j} = \frac{\exp(s_{i, j})}{\sum_{k=1}^N \exp(s_{i, k})}$$
3. **Context Vector:** The updated representation $\mathbf{z}_i$ is the convex combination of all tokens weighted by attention:
   $$\mathbf{z}_i = \sum_{j=1}^N \alpha_{i, j} \mathbf{x}_j$$

### 2. Scaled Dot-Product Attention with Trainable QKV (Vaswani et al., 2017)
Real-world architectures project input tokens $\mathbf{X} \in \mathbb{R}^{N \times D}$ into three distinct learned functional representations:
- **Queries ($\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$):** "What the token is searching for."
- **Keys ($\mathbf{K} = \mathbf{X}\mathbf{W}_K$):** "What information the token offers."
- **Values ($\mathbf{V} = \mathbf{X}\mathbf{W}_V$):** "The actual informational payload carried by the token."

The attention equation is formulated in matrix form:
$$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

### 3. The Scaling Factor $\frac{1}{\sqrt{d_k}}$
For high-dimensional key vectors ($d_k \gg 1$), if components of $\mathbf{q}$ and $\mathbf{k}$ are independent random variables with zero mean and unit variance, their dot product $\mathbf{q} \cdot \mathbf{k} = \sum_{m=1}^{d_k} q_m k_m$ has mean $0$ and variance $d_k$. As $d_k$ grows large, dot products attain extreme magnitudes, pushing the softmax function into regions with near-zero gradients ($\sigma'(z) \approx 0$). Dividing by $\sqrt{d_k}$ normalizes the variance back to $1$, ensuring numerical stability during backpropagation.

**Why it matters here:** Self-attention provides the foundational computational engine of the Vision Transformer, allowing visual patch tokens across distant image coordinates to interact directly without being constrained by convolutional receptive fields.

**Appears in:** [[w1d1|W1D1]] · [[week-1-report|Week 1]]

## Related
- [[multi-head-attention]]
- [[long-range-dependency]]
- [[vision-transformer]]
- [[patch-projection]]
