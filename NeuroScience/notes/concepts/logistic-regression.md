---
title: Logistic Regression
slug: logistic-regression
tags: [ns/concept, ns/machine-learning]
introduced: "[[week-3-report|Week 3]]"
reinforced: []
---

# Logistic Regression

Logistic regression is a linear probabilistic classification model that estimates the posterior probability of a binary target as a logistic sigmoid function of a linear combination of features:
$$\hat{p}(x) = P(Y = 1 \mid X = x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$
The weight vector $w \in \mathbb{R}^P$ specifies the orientation of the flat decision hyperplane $w^T x + b = 0$, while the bias scalar $b$ dictates its spatial offset. Parameters are estimated by minimizing the cross-entropy loss augmented by an $L_2$ (Ridge) regularization penalty $\frac{1}{2C}\|w\|_2^2$ to prevent overfitting on collinear neural features. Model probabilities reflect the mathematical distance to the decision boundary; they do not represent biological certainty or calibrated physiological belief.

**Why it matters here:** Logistic regression provides the standard, highly interpretable baseline for neural decoding, revealing the direction and relative importance of individual electrophysiological features via weight coefficients.

**Appears in:** [[week-3-report|Week 3]] · [[w3d2|W3D2]] · [[w3d3|W3D3]]

## Related
- [[supervised-learning]]
- [[decision-tree]]
- [[support-vector-machine]]
- [[overfitting]]
