---
title: Pooling
slug: pooling
tags: ['ns/concept', 'ns/deep-learning', 'ns/cnn']
introduced: "[[week-4-report|Week 4]]"
---

# Pooling

Pooling is a non-linear or averaging downsampling operation applied along the temporal or spatial dimensions of intermediate convolutional feature maps. Operating with a window length $P$ and stride $s$, pooling transforms an input map of length $T$ into a condensed map of length $\lfloor (T - P)/s floor + 1$.

Common pooling operations include:
1. **Max Pooling:** Computes $\max_{p \in [0, P-1]} x[t \cdot s + p]$, preserving the peak activation (oscillation envelope) while introducing local translation invariance to minor phase shifts.
2. **Average Pooling:** Computes $rac{1}{P} \sum_{p=0}^{P-1} x[t \cdot s + p]$, attenuating high-frequency noise and smoothing activations.
3. **Adaptive Average Pooling:** Dynamically scales window parameters to collapse arbitrary sequence lengths into a fixed dimensional vector (e.g., length 1) prior to linear classification.

In EEG analysis, temporal pooling downsamples effective sampling rates (e.g., from 160 Hz to 40 Hz), reducing memory consumption, mitigating overfitting, and expanding the effective receptive field of subsequent deeper layers.

**Why it matters here:** Pooling extracts low-frequency power envelopes from high-frequency oscillatory waveforms, mimicking biological demodulation in the sensorimotor cortex.

**Appears in:** [[week-4-report|Week 4]] · [[w4d2|W4D2]]

## Related
- [[convolution]]
- [[representation-learning]]
- [[nyquist-criterion]]
