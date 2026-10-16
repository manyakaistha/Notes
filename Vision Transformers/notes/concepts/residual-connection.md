---
title: Residual Connection (Skip Connection)
slug: residual-connection
tags: ['vt/concept', 'vt/deep-learning', 'vt/optimization', 'vt/architecture']
introduced: "[[w1d1|W1D1]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Residual Connection (Skip Connection)

A residual connection (or skip connection), pioneered by He et al. (2016) in Deep Residual Networks (ResNet), is an architectural mechanism that adds the identity input of a layer directly to its transformed output:
$$\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})$$
where $\mathbf{x}$ is the input tensor and $\mathcal{F}(\mathbf{x})$ represents the non-linear sublayer transformation (such as a convolutional block, multi-head self-attention module, or feed-forward network).

### 1. Mathematical Mechanics of Gradient Propagation
During backpropagation, the gradient of a scalar loss $\mathcal{L}$ with respect to the input $\mathbf{x}$ decomposes via the chain rule into:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \left( \mathbf{I} + \frac{\partial \mathcal{F}(\mathbf{x})}{\partial \mathbf{x}} \right) = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} + \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \frac{\partial \mathcal{F}(\mathbf{x})}{\partial \mathbf{x}}$$

The identity term $\mathbf{I}$ establishes an unobstructed gradient highway directly connecting the output of the network back to earlier layers. Even if the Jacobian of the sublayer $\frac{\partial \mathcal{F}(\mathbf{x})}{\partial \mathbf{x}}$ vanishes or approaches zero (e.g. due to saturating activations or small weight magnitudes), the total gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ preserves the incoming gradient signal $\frac{\partial \mathcal{L}}{\partial \mathbf{y}}$ without decaying exponentially.

### 2. Residual Connections in Transformer Blocks
In the canonical Transformer block, residual connections wrap around both major sublayers:
1. **Around Multi-Head Self-Attention:**
   $$\mathbf{z}'_l = \mathbf{z}_{l-1} + \operatorname{MSA}(\operatorname{LN}(\mathbf{z}_{l-1}))$$
2. **Around the Position-wise Feed-Forward Network:**
   $$\mathbf{z}_l = \mathbf{z}'_l + \operatorname{MLP}(\operatorname{LN}(\mathbf{z}'_l))$$

In a deep $L$-layer Transformer encoder, the output at layer $L$ expands into an unrolled sum of residual perturbations:
$$\mathbf{z}_L = \mathbf{z}_0 + \sum_{l=1}^L \operatorname{MSA}(\operatorname{LN}(\dots)) + \sum_{l=1}^L \operatorname{MLP}(\operatorname{LN}(\dots))$$
This ensures that the initial patch embeddings and positional representations $\mathbf{z}_0$ remain directly accessible throughout all encoder stages.

**Why it matters here:** Residual connections prevent the vanishing gradient pathology when stacking deep Transformer encoders (e.g., 12 layers in ViT-Base, 24 in ViT-Large, 32 in ViT-Huge), enabling stable optimization and high representational capacity.

**Appears in:** [[w1d1|W1D1]] · [[week-1-report|Week 1]]

## Related
- [[vanishing-gradients]]
- [[layer-normalization]]
- [[vision-transformer]]
- [[multi-head-attention]]
