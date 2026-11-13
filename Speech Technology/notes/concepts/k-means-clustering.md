---
title: K-Means Clustering in Speech
tags: [speech/concept, machine-learning/clustering, vector-quantization]
---

# K-Means Clustering in Speech

**Formal definition.** K-means partitions observations $x_i\in\mathbb R^D$ into at most $K$ nonempty clusters by minimizing within-cluster squared Euclidean distortion $J=\sum_{k=1}^K\sum_{i:z_i=k}\|x_i-\mu_k\|_2^2$. Lloyd iterations alternate nearest-centroid assignments and centroid updates to the assigned arithmetic means. Each step does not increase $J$, but the objective is nonconvex: the result depends on initialization, ties, empty-cluster handling, and feature scaling, and convergence is to a local optimum rather than a guaranteed global minimum. In speech, the procedure supports vector quantization and codebook initialization; Linde-Buzo-Gray (LBG) grows codebooks by splitting centroids with perturbations and refining them. K-means uses hard assignments and squared Euclidean geometry, so it is not equivalent to probabilistic mixture estimation and may poorly represent clusters with unequal covariance or density.

**Why it matters here.** It supplies a baseline discrete acoustic partition and an initialization intuition for the soft responsibilities of GMMs.

**Appears in:** [[Knowledge-Map]] (supplementary registry; course week unassigned)

## Related
- [[gaussian-mixture-models]]
- [[cepstral-analysis-mfcc]]
