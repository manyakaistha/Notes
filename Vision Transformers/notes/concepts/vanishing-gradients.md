---
title: Vanishing Gradients
slug: vanishing-gradients
tags: ['vt/concept', 'vt/deep-learning', 'vt/optimization']
introduced: "[[w1d1|W1D1]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Vanishing Gradients

The vanishing gradient pathology refers to the exponential decay of backpropagated gradient magnitudes as error signals traverse backwards through deep neural network layers from the output to the initial parameters.

In a deep feedforward network parameterized by weight matrices $\mathbf{W}_1, \dots, \mathbf{W}_L$ with activation functions $\sigma(\cdot)$, the gradient of the scalar loss $\mathcal{L}$ with respect to the input weights $\mathbf{W}_1$ is formulated via the chain rule of calculus:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}_L} \cdot \prod_{l=2}^{L} \left( \mathbf{W}_l^T \cdot \operatorname{diag}(\sigma'(\mathbf{z}_{l-1})) \right) \cdot \frac{\partial \mathbf{a}_1}{\partial \mathbf{W}_1}$$

If the spectral norms of the weight matrices $\|\mathbf{W}_l\|_2$ or the derivatives of the activation functions $|\sigma'(z)|$ are less than $1$ (as is strictly true for saturating activations such as sigmoid and tanh where $|\sigma'(z)| \le 0.25$, and for unnormalized linear transformations), the continuous product $\prod_{l=2}^L$ causes the magnitude of $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1}$ to decay exponentially toward zero as depth $L \to \infty$.

### Impact on Deep Vision Models
When convolutional networks attempt to expand their receptive field solely by stacking dozens or hundreds of sequential layers to capture global context, earlier feature extractors receive negligible gradient updates. This leaves early edge and texture filters unoptimized or randomly initialized, halting training progress. While architectural innovations such as Residual Connections (He et al., 2016) and Batch Normalization mitigate gradient decay, deep CNNs still suffer from optimization degradation and excessive computational latency compared to parallel self-attention routing.

**Why it matters here:** The risk of gradient degradation when expanding receptive fields in deep CNNs directly motivates architectures with flat interaction topologies, such as Vision Transformers, which connect all input tokens in $\mathcal{O}(1)$ operations.

**Appears in:** [[w1d1|W1D1]] · [[week-1-report|Week 1]]

## Related
- [[receptive-field]]
- [[convolution]]
- [[vision-transformer]]
- [[long-range-dependency]]
