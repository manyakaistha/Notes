---
title: Representation Learning
slug: representation-learning
tags: ['ns/concept', 'ns/deep-learning', 'ns/representations']
introduced: "[[week-4-report|Week 4]]"
---

# Representation Learning

Representation learning is the machine learning paradigm wherein a parameterized system automatically discovers and extracts internal feature representations directly from raw or minimally preprocessed data, replacing manual, handcrafted heuristic feature engineering.

In classical electrophysiology (Weeks 1–3), practitioners manually construct domain features (e.g., Welch power spectral densities in pre-defined alpha and beta bands, band variance, time-domain peak voltages), which irreversibly discard continuous waveform phase and non-linear harmonic interactions. In deep representation learning (Week 4), end-to-end backpropagation shapes early layers into specialized spatio-temporal filters, intermediate layers into envelope and rhythm detectors, and late layers into compact class manifolds. 

While representation learning eliminates human feature design bias, it demands substantially larger training datasets to avoid memorizing biophysical sensor artifacts and subject-specific idiosyncratic patterns.

**Why it matters here:** Representation learning fundamentally distinguishes deep neural networks from classical shallow decoders, shifting the burden of inductive bias from manual engineering to learned filter banks.

**Appears in:** [[week-4-report|Week 4]] · [[w4d1|W4D1]] · [[w4d2|W4D2]] · [[w4d5|W4D5]]

## Related
- [[forward-propagation]]
- [[convolution]]
- [[supervised-learning]]
- [[feature-matrix]]
