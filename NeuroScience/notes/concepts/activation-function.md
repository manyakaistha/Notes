---
title: Activation Function
slug: activation-function
tags: ['ns/concept', 'ns/deep-learning', 'ns/non-linearity']
introduced: "[[week-4-report|Week 4]]"
---

# Activation Function

An activation function is a non-linear operator $f: \mathbb{R} 	o \mathbb{R}$ applied element-wise to the pre-activation logit $z$ of an artificial neuron. Without non-linear activation functions, any composition of affine layers $W_2 (W_1 x + b_1) + b_2$ collapses mathematically to a single affine transformation $(W_2 W_1) x + (W_2 b_1 + b_2)$, rendering deep architectures incapable of learning non-linear decision boundaries.

Common activations include:
1. **Rectified Linear Unit (ReLU):** $f(z) = \max(0, z)$, standard for hidden layers due to its constant unit gradient for $z > 0$, mitigating vanishing gradients.
2. **Logistic Sigmoid:** $\sigma(z) = (1 + e^{-z})^{-1} \in (0, 1)$, used at output heads for binary classification.
3. **Hyperbolic Tangent ($	anh$):** $	anh(z) = rac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1, 1)$, zero-centered for recurrent hidden state updates.
4. **Softmax:** Multi-class generalization producing normalized posterior class probability distributions summing to 1.

**Why it matters here:** Non-linear activations allow networks to model complex cross-frequency interactions and non-linear phase-amplitude couplings present in biological EEG signals.

**Appears in:** [[week-4-report|Week 4]] · [[w4d1|W4D1]] · [[w4d2|W4D2]]

## Related
- [[artificial-neuron]]
- [[forward-propagation]]
- [[backpropagation]]
