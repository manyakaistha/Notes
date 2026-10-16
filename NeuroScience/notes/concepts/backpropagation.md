---
title: Backpropagation
slug: backpropagation
tags: ['ns/concept', 'ns/deep-learning', 'ns/optimization']
introduced: "[[week-4-report|Week 4]]"
---

# Backpropagation

Backpropagation (backward propagation of errors) is the computationally efficient algorithm that evaluates the exact analytical gradient of a scalar objective loss $L$ with respect to every trainable parameter $	heta = \{W, b\}$ in a neural network using the multi-variable calculus chain rule.

Starting from the output loss $L(y, \hat{y})$, errors are propagated backward layer-by-layer:
$$\delta^{(l)} = rac{\partial L}{\partial z^{(l)}} = \left( (W^{(l+1)})^T \delta^{(l+1)} ight) \odot f'^{(l)}(z^{(l)})$$
$$rac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T, \quad rac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$
Parameters are then iteratively updated via gradient descent: $	heta \leftarrow 	heta - \eta 
abla_	heta L$. Backpropagation does not represent biological learning; the biological brain lacks symmetric backward connectivity matrices, synchronized global error transport, and frozen forward-backward phases (the "weight transport problem").

**Why it matters here:** Backpropagation enables end-to-end representation learning, allowing convolutional kernels to automatically specialize into neurophysiologically relevant band-pass filters without manual heuristic tuning.

**Appears in:** [[week-4-report|Week 4]] · [[w4d1|W4D1]]

## Related
- [[forward-propagation]]
- [[artificial-neuron]]
- [[activation-function]]
