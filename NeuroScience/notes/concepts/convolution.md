---
title: Convolution
slug: convolution
tags: ['ns/concept', 'ns/deep-learning', 'ns/cnn', 'ns/filtering']
introduced: "[[week-4-report|Week 4]]"
---

# Convolution

In deep learning, a 1D discrete convolution is a linear filtering operation that slides a learnable weight kernel $w \in \mathbb{R}^K$ across an input sequence $x \in \mathbb{R}^T$ to produce an output feature map $y \in \mathbb{R}^{T'}$:
$$y[t] = (x * w)[t] = \sum_{k=0}^{K-1} x[t \cdot s + k] \, w[k] + b$$
where $s$ denotes the stride and $K$ represents the kernel size (receptive field). 

Convolution incorporates two fundamental architectural priors:
1. **Local Connectivity:** Each output unit depends solely on a localized temporal patch of duration $K / f_s$ seconds, matching the transient nature of electrophysiological oscillations.
2. **Weight Sharing:** The identical kernel $w$ is evaluated across all time positions, ensuring shift-invariance (a neural transient like an epileptiform spike or sensorimotor mu desynchronization is recognized regardless of onset latency).

In multichannel EEG architectures, temporal convolutions operate along the time axis ($1 	imes K$), while spatial convolutions operate across sensor channels ($C 	imes 1$) to compute instantaneous spatial filters analogous to bipolar rereferencing or Common Spatial Patterns.

**Why it matters here:** Convolution replaces manual Fourier or wavelet feature engineering by learning task-specific finite impulse response (FIR) filters directly from raw microvolt traces.

**Appears in:** [[week-4-report|Week 4]] · [[w4d2|W4D2]] · [[w4d5|W4D5]]

## Related
- [[pooling]]
- [[representation-learning]]
- [[band-pass-filtering]]
