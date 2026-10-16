---
title: Dropout
slug: dropout
tags: ['vt/concept', 'vt/deep-learning', 'vt/regularization', 'vt/optimization']
introduced: "[[w1d2|W1D2]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Dropout

Dropout (Srivastava et al., 2014) is a stochastic regularization technique designed to prevent complex co-adaptations between hidden neurons during deep neural network training.

### 1. Intuition: Forcing Lazy Neurons to Learn
In unregularized feedforward networks, a small subset of dominant neurons frequently performs the majority of the optimization heavy lifting, while neighboring neurons remain passive ("lazy neurons"). If those dominant units are stochastically masked during training, the network is compelled to distribute feature representations across all available units, learning redundant and robust pathways.

### 2. Mathematical Mechanics: Inverted Dropout
During each training step, each neuron's activation is independently set to zero with probability $p \in (0, 1)$ (the dropout rate, typically $0.1$ to $0.5$):
$$r_j \sim \operatorname{Bernoulli}(1 - p)$$
$$\tilde{x}_j = \frac{1}{1 - p} \cdot r_j \cdot x_j$$

- **Inverted Scaling ($\frac{1}{1 - p}$):** Because dropping a fraction $p$ of units reduces the expected sum of activations by a factor of $(1 - p)$, dividing by $(1 - p)$ during training ensures that the expected activation magnitude remains invariant:
  $$\mathbb{E}[\tilde{x}_j] = \frac{1}{1 - p} \mathbb{E}[r_j] x_j = \frac{1 - p}{1 - p} x_j = x_j$$
  This inverted scaling eliminates the need to rescale weights or activations during inference, allowing the model to execute standard deterministic forward passes at test time.

### 3. Dropout in Transformers
Within the canonical Transformer block, Dropout is strategically applied at two distinct interfaces:
1. **Post-Attention Dropout:** Applied immediately following the Multi-Head Self-Attention linear projection $\mathbf{W}^O$ before the residual skip addition, forcing attention weights to distribute broadly rather than focusing exclusively on single strong associations.
2. **Post-FFN Dropout:** Applied following the second linear transformation $\mathbf{W}_2$ in the Feed-Forward Network before the second residual skip addition.

**Why it matters here:** In both large language models and Vision Transformers, dropout regularizes high-capacity models against overfitting, stabilizing training on web-scale corpora.

**Appears in:** [[w1d2|W1D2]] · [[week-1-report|Week 1]]

## Related
- [[feed-forward-network]]
- [[residual-connection]]
- [[layer-normalization]]
- [[vision-transformer]]
