---
title: K-Means Clustering in Speech
tags: [speech/concept, machine-learning/clustering, vector-quantization]
---

# K-Means Clustering in Speech

**K-Means Clustering** is a deterministic, hard-assignment partitioning algorithm that segments an unlabelled acoustic dataset $\mathcal{X} = \{x_1, \dots, x_N\}$ into $K$ disjoint clusters by minimizing the within-cluster sum of squares (WCSS) distortion objective:

$$J = \sum_{k=1}^K \sum_{x_i \in S_k} \| x_i - \mu_k \|^2$$

### Algorithmic Mechanics
1. **Assignment Step:** Assign each continuous acoustic observation vector $x_i$ strictly to the closest centroid $\mu_k$ according to Euclidean distance.
2. **Update Step:** Recompute each centroid as the arithmetic mean of all vectors assigned to it:
   $$\mu_k = \frac{1}{|S_k|} \sum_{x_i \in S_k} x_i$$
3. **Convergence:** Iterate until centroid shifts fall below a threshold or assignments stabilize.

### Vector Quantization & LBG Splitting
To initialize cluster centers without getting trapped in bad local minima, the Linde-Buzo-Gray (LBG) algorithm starts with the global dataset mean and splits each centroid using a small perturbation factor $\epsilon$:

$$c_1 = \mu (1 + \epsilon), \quad c_2 = \mu (1 - \epsilon)$$

### Key Properties & Constraints
- **Fixed Hyperparameter:** The number of clusters $K$ is set prior to running and remains strictly fixed throughout convergence.
- **Hard Assignment:** Points belong 100% to one cluster; outliers directly drag the centroid of their assigned cluster.
- **Local Minima:** Optimization relies on greedy coordinate descent and does not guarantee reaching the global minimum.

**Appears in:** [[week 2/week-2-notes|Week 2 Notes]]

## Related
- [[gaussian-mixture-models]]
- [[cepstral-analysis-mfcc]]
