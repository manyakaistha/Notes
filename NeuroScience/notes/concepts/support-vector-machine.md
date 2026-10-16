---
title: Support Vector Machine
slug: support-vector-machine
tags: [ns/concept, ns/machine-learning]
introduced: "[[week-3-report|Week 3]]"
reinforced: []
---

# Support Vector Machine

A Support Vector Machine (SVM) is a supervised classification algorithm that constructs an optimal separating hyperplane $w^T x + b = 0$ that maximizes the geometric margin $\frac{2}{\|w\|_2}$ between classes. For non-separable neural data, the soft-margin formulation introduces slack variables $\xi_i \ge 0$ to penalize margin violations, governed by regularization parameter $C > 0$. The optimal hyperplane depends exclusively on the **support vectors**—the subset of training observations that lie on the margin boundaries or violate them ($\alpha_i > 0$). In noisy EEG feature spaces, a large fraction of training observations become support vectors due to extensive distribution overlap. SVMs require feature standardization because the margin is defined in Euclidean metric space.

**Why it matters here:** Linear SVM is widely considered the gold-standard linear classifier in brain-computer interfaces due to its theoretical generalization bounds and robust resistance to non-support outliers.

**Appears in:** [[week-3-report|Week 3]] · [[w3d3|W3D3]] · [[w3d5|W3D5]]

## Related
- [[logistic-regression]]
- [[decision-tree]]
- [[subject-generalization]]
