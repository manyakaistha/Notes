---
title: Forward Propagation
slug: forward-propagation
tags: ['ns/concept', 'ns/deep-learning', 'ns/inference']
introduced: "[[week-4-report|Week 4]]"
---

# Forward Propagation

Forward propagation is the deterministic sequence of mathematical transformations that maps an input tensor through the successive layers of a parameterized neural network to produce an output prediction or loss value. For an $L$-layer network, forward propagation proceeds sequentially:
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad a^{(l)} = f^{(l)}(z^{(l)}) \quad 	ext{for } l = 1, \dots, L$$
where $a^{(0)} = x$ is the raw or preprocessed neural input, $W^{(l)}$ and $b^{(l)}$ are layer weight matrices and bias vectors, and $f^{(l)}$ is the layer activation function.

In neural decoding, forward propagation tracks explicit tensor geometries: an input epoch of shape $[B, C, T]$ passes through convolutional filters, spatial projections, and pooling operators to yield a latent feature tensor, finally arriving at a logit vector $[B, K]$ for $K$ target behavioral classes. Intermediate layer activations (activations $a^{(l)}$) are cached during forward propagation to evaluate gradients during the backward pass.

**Why it matters here:** Forward propagation defines the inference pipeline that transforms multichannel EEG waveforms into real-time device control commands or diagnostic classifications.

**Appears in:** [[week-4-report|Week 4]] · [[w4d1|W4D1]] · [[w4d2|W4D2]]

## Related
- [[artificial-neuron]]
- [[backpropagation]]
- [[representation-learning]]
