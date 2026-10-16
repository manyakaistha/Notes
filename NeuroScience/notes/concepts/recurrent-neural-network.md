---
title: Recurrent Neural Network
slug: recurrent-neural-network
tags: ['ns/concept', 'ns/deep-learning', 'ns/sequential']
introduced: "[[week-4-report|Week 4]]"
---

# Recurrent Neural Network

A Recurrent Neural Network (RNN) is a neural architecture specialized for processing sequential data $x = (x_1, x_2, \dots, x_T)$ by maintaining an internal hidden state vector $h_t \in \mathbb{R}^H$ that acts as a working memory of historical context:
$$h_t = 	anh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$$
The recurrent weight matrix $W_{hh} \in \mathbb{R}^{H 	imes H}$ is shared across all time steps $t$, allowing the model to process sequences of arbitrary length with a constant parameter budget.

When applied to long continuous sequences such as raw EEG (where a 3-second trial spans 480 time steps at 160 Hz), standard RNNs suffer from vanishing and exploding gradients during backpropagation through time (BPTT). Repeated matrix multiplications by $W_{hh}^T$ cause error gradients to decay exponentially to zero or blow up to infinity, preventing standard RNNs from learning long-range temporal dependencies.

**Why it matters here:** RNNs formalize sequential processing in neural data, explaining why raw multi-hundred-sample EEG trials require specialized gated or convolutional designs rather than vanilla recurrent loops.

**Appears in:** [[week-4-report|Week 4]] · [[w4d3|W4D3]]

## Related
- [[long-short-term-memory]]
- [[self-attention]]
- [[representation-learning]]
