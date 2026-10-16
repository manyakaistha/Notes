---
title: Artificial Neuron
slug: artificial-neuron
tags: ['ns/concept', 'ns/deep-learning', 'ns/neural-networks']
introduced: "[[week-4-report|Week 4]]"
---

# Artificial Neuron

An artificial neuron is an elementary computational unit that computes a scalar activation from an input vector $x \in \mathbb{R}^P$. It performs an affine transformation parameterized by a weight vector $w \in \mathbb{R}^P$ and a bias scalar $b \in \mathbb{R}$, yielding an intermediate logit $z = w^T x + b = \sum_{j=1}^P w_j x_j + b$, followed by an element-wise non-linear activation function $f(z)$:
$$\hat{y} = f(z) = f\left(\sum_{j=1}^P w_j x_j + bight)$$
In binary classification, $f$ is typically the logistic sigmoid $\sigma(z) = (1 + e^{-z})^{-1}$, bounding the output to $(0, 1)$. The weights quantify the sensitivity and polarity of the response to each input feature, while the bias offsets the threshold. Unlike biological neurons, artificial neurons abstract away biophysical membrane dynamics, action potential generation, temporal ion conductances, and metabolic dendritic integration into a static, synchronized algebraic point-process.

**Why it matters here:** The artificial neuron is the foundational building block for multi-layer perceptrons, convolutional feature detectors, and recurrent architectures in neural decoding.

**Appears in:** [[week-4-report|Week 4]] · [[w4d1|W4D1]]

## Related
- [[activation-function]]
- [[forward-propagation]]
- [[backpropagation]]
- [[logistic-regression]]
