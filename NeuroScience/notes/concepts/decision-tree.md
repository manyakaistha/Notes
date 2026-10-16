---
title: Decision Tree
slug: decision-tree
tags: [ns/concept, ns/machine-learning]
introduced: "[[week-3-report|Week 3]]"
reinforced: []
---

# Decision Tree

A decision tree is a non-parametric supervised learning algorithm that partitions the feature space into hyper-rectangular regions via recursive binary splitting. At each internal node, the algorithm selects the feature $j$ and numerical threshold $t$ that maximizes the reduction in node impurity, typically quantified by Gini impurity $I_G = 1 - \sum p_k^2$ or entropy. Decision trees are invariant to monotonic feature scaling, eliminating the requirement for feature standardization. However, unconstrained trees split until every training leaf is pure, resulting in severe overfitting and brittle decision boundaries. Depth constraints (`max_depth`) and minimum leaf sizes (`min_samples_leaf`) are mandatory regularizers when classifying noisy electrophysiological data.

**Why it matters here:** Decision trees provide transparent, rule-based decision logic that can capture non-linear feature interactions, serving as a non-linear contrast to linear models.

**Appears in:** [[week-3-report|Week 3]] · [[w3d3|W3D3]]

## Related
- [[logistic-regression]]
- [[support-vector-machine]]
- [[overfitting]]
