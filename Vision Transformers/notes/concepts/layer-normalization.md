---
title: Layer Normalization
slug: layer-normalization
tags: ['vt/concept', 'vt/deep-learning', 'vt/normalization', 'vt/transformer']
introduced: "[[w1d1|W1D1]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Layer Normalization

Layer Normalization (Ba, Kiros, & Hinton, 2016) is a normalization technique that normalizes activations across the feature channel dimension for each individual token independently, ensuring stable hidden state dynamics during training.

### 1. Mathematical Formulation
Given an input vector $\mathbf{x} \in \mathbb{R}^D$ representing a single token embedding at any position in the sequence:
1. **Mean and Variance:**
   $$\mu = \frac{1}{D} \sum_{i=1}^D x_i, \quad \sigma^2 = \frac{1}{D} \sum_{i=1}^D (x_i - \mu)^2$$
2. **Standardization & Affine Transformation:**
   $$\operatorname{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$
   where $\epsilon > 0$ is a small constant preventing division by zero, and $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^D$ are learnable scale (gain) and shift (bias) parameters initialized to $1$ and $0$, respectively.

### 2. LayerNorm vs. Batch Normalization in Transformers
- **Batch Normalization (BN):** Normalizes activations across the batch dimension for each spatial coordinate independently. BN depends strictly on batch statistics, introducing inter-sample dependencies and exhibiting instability when batch sizes are small or sequence lengths vary.
- **Layer Normalization (LN):** Normalizes across features for each individual token completely independently of other samples in the batch. This makes LN batch-size independent and ideal for variable-length sequences and distributed data-parallel training.

### 3. Pre-LN vs. Post-LN Architectures
- **Post-LN (Original Transformer, Vaswani et al., 2017):**
  $$\mathbf{x}_{l} = \operatorname{LN}(\mathbf{x}_{l-1} + \operatorname{Sublayer}(\mathbf{x}_{l-1}))$$
  Normalizing after the residual addition leads to gradient accumulation pathologies in very deep networks, requiring delicate learning rate warmups.
- **Pre-LN (Vision Transformer, Dosovitskiy et al., 2021):**
  $$\mathbf{x}_{l} = \mathbf{x}_{l-1} + \operatorname{Sublayer}(\operatorname{LN}(\mathbf{x}_{l-1}))$$
  Applying LayerNorm directly to the input of each sublayer preserves an uninterrupted residual identity highway ($\mathbf{x}_L = \mathbf{x}_0 + \sum_{l=1}^L \dots$), stabilizing gradient propagation and enabling effortless optimization of deep architectures.

**Why it matters here:** Layer Normalization guarantees zero mean and unit variance inputs to Multi-Head Attention and MLP blocks in the Vision Transformer, preventing activation drift and ensuring training stability.

**Appears in:** [[w1d1|W1D1]] · [[week-1-report|Week 1]]

## Related
- [[residual-connection]]
- [[vision-transformer]]
- [[self-attention]]
