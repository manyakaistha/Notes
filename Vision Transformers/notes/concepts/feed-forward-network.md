---
title: Position-Wise Feed-Forward Network (FFN)
slug: feed-forward-network
tags: ['vt/concept', 'vt/deep-learning', 'vt/transformer', 'vt/architecture']
introduced: "[[w1d2|W1D2]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Position-Wise Feed-Forward Network (FFN)

The position-wise Feed-Forward Network (FFN)—also termed the Multi-Layer Perceptron (MLP) block—is a sublayer present within every Transformer block that processes each sequence token independently and identically through two linear coordinate transformations separated by a non-linear activation function.

### 1. Mathematical Formulation
For a token vector $\mathbf{x} \in \mathbb{R}^D$:
$$\operatorname{FFN}(\mathbf{x}) = \sigma(\mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$
where:
- $\mathbf{W}_1 \in \mathbb{R}^{D \times d_{ff}}$ projects the token from dimension $D$ up to an expanded hidden dimension $d_{ff}$ (conventionally $d_{ff} = 4D$, e.g., $768 \to 3072$).
- $\mathbf{b}_1 \in \mathbb{R}^{d_{ff}}$ is the first intermediate bias vector.
- $\sigma(\cdot)$ is a non-linear activation function, standardly Gaussian Error Linear Units (GELU) or Rectified Linear Units (ReLU).
- $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times D}$ projects the expanded representation back to model dimension $D$.
- $\mathbf{b}_2 \in \mathbb{R}^D$ is the final output bias vector.

### 2. Why Expand to $4D$ Instead of Compressing?
- **Information Preservation & Non-Linear Capacity:** Compressing $\mathbf{x}$ to a lower dimension (e.g., $768 \to 300 \to 768$) acts as a bottleneck that irreversibly destroys high-frequency latent information. In contrast, projecting to a $4\times$ higher-dimensional space ($768 \to 3072$) creates an overcomplete basis that enriches feature capacity, allowing non-linear activation functions to separate complex, non-linear relationships before projecting back.
- **Key-Value Associative Memory:** Research (Geva et al., 2021) demonstrates that the FFN operates as an associative key-value memory: the first layer $\mathbf{W}_1$ detects specific textual/visual concepts (keys), while the second layer $\mathbf{W}_2$ produces corresponding contextual modifications (values).

### 3. Parameter Dominance
The FFN accounts for approximately **two-thirds ($\approx 66\%$)** of all trainable parameters in standard Transformer blocks. In ViT-Base ($D = 768, d_{ff} = 3072$):
- $\mathbf{W}_1$: $768 \times 3072 \approx 2.36\text{M}$ weights.
- $\mathbf{W}_2$: $3072 \times 768 \approx 2.36\text{M}$ weights.
- Total per block $\approx 4.72\text{M}$ parameters (compared to $\approx 2.36\text{M}$ in Multi-Head Attention).

**Why it matters here:** While Multi-Head Attention routes information across different spatial tokens, the FFN transforms and refines the feature representations within each individual token independently.

**Appears in:** [[w1d2|W1D2]] · [[week-1-report|Week 1]]

## Related
- [[multi-head-attention]]
- [[layer-normalization]]
- [[residual-connection]]
- [[vision-transformer]]
