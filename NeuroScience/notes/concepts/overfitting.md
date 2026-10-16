---
title: Overfitting
slug: overfitting
tags: [ns/concept, ns/machine-learning]
introduced: "[[week-3-report|Week 3]]"
reinforced: []
---

# Overfitting

Overfitting occurs when a machine learning model learns idiosyncrasies, random fluctuations, and noise specific to the training dataset rather than true underlying population structure. As model capacity increases (e.g., deeper decision trees, unregularized logistic regression, or unconstrained deep neural networks), training error decreases monotonically toward zero while generalization error on unseen test data increases. In electrophysiology, overfitting is exacerbated by low sample sizes ($N \sim 10^1\text{--}10^2$ trials), high feature dimensionality, and pervasive non-neural artifacts. Regularization techniques—such as $L_2$ weight shrinkage, tree depth limits, and cross-validation hyperparameter selection—are mandatory defenses against memorizing noise.

**Why it matters here:** EEG signals contain massive non-task noise; an unregularized model easily achieves 100% training accuracy by memorizing artifactual spikes while failing completely on real-time held-out data.

**Appears in:** [[week-3-report|Week 3]] · [[w3d3|W3D3]] · [[w3d4|W3D4]]

## Related
- [[supervised-learning]]
- [[decision-tree]]
- [[logistic-regression]]
- [[data-leakage]]
