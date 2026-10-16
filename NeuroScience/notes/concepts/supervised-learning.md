---
title: Supervised Learning
slug: supervised-learning
tags: [ns/concept, ns/machine-learning]
introduced: "[[week-3-report|Week 3]]"
reinforced: []
---

# Supervised Learning

Supervised learning is the machine learning paradigm in which an algorithm learns a mapping function $f: \mathcal{X} \to \mathcal{Y}$ from an explicit training dataset of input-target pairs $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$. Each observation $x_i \in \mathbb{R}^P$ is a vector of measured or extracted features, and $y_i$ is a known ground-truth label. The algorithm optimizes model parameters to minimize a loss function $\mathcal{L}(f(x), y)$ on training data, with the goal of generalizing to unseen test observations. In electrophysiology, supervised learning encompasses discrete classification (e.g., motor imagery intent), event detection (e.g., epileptic seizures), and continuous decoding (e.g., cursor kinematics).

**Why it matters here:** Supervised learning provides an operational test for neural coding: if an algorithm decodes a cognitive or behavioral state from neural features above chance on held-out data, those features carry statistically reliable information about that state.

**Appears in:** [[week-3-report|Week 3]] · [[w3d1|W3D1]] · [[w3d2|W3D2]] · [[w3d5|W3D5]]

## Related
- [[feature-matrix]]
- [[experimental-labels]]
- [[classification-metrics]]
- [[overfitting]]
