---
title: Feature Matrix
slug: feature-matrix
tags: [ns/concept, ns/data-representation]
introduced: "[[week-3-report|Week 3]]"
reinforced: []
---

# Feature Matrix

A feature matrix $X \in \mathbb{R}^{N \times P}$ is a structured two-dimensional numerical array in which each row represents an independent biological observation or trial ($i \in \{1, \dots, N\}$) and each column represents a specific quantified property or feature ($j \in \{1, \dots, P\}$). This contrasts with continuous electrophysiological recording matrices $V \in \mathbb{R}^{C \times T}$, where dimensions represent physical channels and sequential time samples. Constructing $X$ from $V$ requires epoching around experimental events and compressing temporal intervals into scalar descriptors (e.g., mean voltage, signal variance, spectral band power). Because features often have vastly different physical units ($\mu\text{V}$ vs. $\mu\text{V}^2$), column-wise standardization is mandatory prior to training distance-based or regularized models.

**Why it matters here:** The feature matrix is the universal tabular format required by standard machine learning estimators. Correct row alignment and training-only feature standardization are essential to prevent data leakage.

**Appears in:** [[week-3-report|Week 3]] · [[w3d1|W3D1]] · [[w3d2|W3D2]]

## Related
- [[supervised-learning]]
- [[experimental-labels]]
- [[data-leakage]]
