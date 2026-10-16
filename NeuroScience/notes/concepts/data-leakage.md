---
title: Data Leakage
slug: data-leakage
tags: [ns/concept, ns/evaluation]
introduced: "[[week-3-report|Week 3]]"
reinforced: []
---

# Data Leakage

Data leakage is the spurious sharing of information between the training pipeline and the held-out evaluation dataset, yielding artificially inflated performance estimates that fail to generalize to real-world deployment. In electrophysiology, leakage manifests in three primary modes:
1. **Preprocessing Leakage:** Fitting scalers, band-pass filters, ICA decompositions, or PCA projections across the entire dataset prior to partitioning folds.
2. **Temporal Correlation Leakage:** Randomly shuffling continuous time series or overlapping sliding windows, scattering temporally autocorrelated epochs across training and test splits.
3. **Subject Identity Leakage:** Randomly shuffling trials pooled from multiple subjects, allowing the classifier to identify individual anatomical head geometry rather than generalizable cognitive dynamics.

Preventing leakage requires isolating preprocessing transformations inside scikit-learn `Pipeline` objects and grouping cross-validation splits by discrete recording runs (`LeaveOneGroupOut`) or subject identities (`LeaveOneSubjectOut`).

**Why it matters here:** Data leakage is the most prevalent methodological flaw in computational neuroscience publications, producing false claims of state-of-the-art neural decoding.

**Appears in:** [[week-3-report|Week 3]] · [[w3d2|W3D2]] · [[w3d4|W3D4]] · [[w3d5|W3D5]]

## Related
- [[classification-metrics]]
- [[overfitting]]
- [[subject-generalization]]
