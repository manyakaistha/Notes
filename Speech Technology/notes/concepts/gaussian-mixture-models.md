---
title: Gaussian Mixture Models
tags: [speech/concept, machine-learning/gmm, acoustic-modeling]
---

# Gaussian Mixture Models (GMMs)

A **Gaussian Mixture Model (GMM)** is a continuous parametric probability density model formed by a convex combination of $K$ multivariate Gaussian components:

$$p(x) = \sum_{k=1}^K w_k \mathcal{N}_k(x; \mu_k, \Sigma_k), \quad \sum_{k=1}^K w_k = 1, \quad w_k \ge 0$$

### Why GMMs are Essential in Speech
A single Gaussian is unimodal and cannot capture the acoustic diversity of a phoneme across different speakers (males, females, children) and phonetic contexts. GMMs act as universal continuous density approximators, modeling complex multimodal feature distributions.

### Expectation-Maximization (EM) Training
Parameters $\theta = \{w_k, \mu_k, \Sigma_k\}$ are trained iteratively via Expectation-Maximization:
1. **E-step (Responsibility Calculation):**
   $$\gamma_{ik} = P(C_k \mid x_i) = \frac{w_k \mathcal{N}(x_i; \mu_k, \Sigma_k)}{\sum_{j=1}^K w_j \mathcal{N}(x_i; \mu_j, \Sigma_j)}$$
2. **M-step (Parameter Update):**
   $$\hat{w}_k = \frac{1}{N} \sum_{i=1}^N \gamma_{ik}, \quad \hat{\mu}_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}, \quad \hat{\Sigma}_k = \frac{\sum_i \gamma_{ik} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T}{\sum_i \gamma_{ik}}$$

### Diagonal Covariance Assumption
Because MFCCs are decorrelated via the DCT, acoustic models enforce diagonal covariance matrices $\Sigma_k = \operatorname{diag}(\sigma_{k1}^2, \dots, \sigma_{kD}^2)$, reducing parameter complexity from $O(D^2)$ to $O(D)$.

**Appears in:** [[week 2/week-2-notes|Week 2 Notes]]

## Related
- [[k-means-clustering]]
- [[hidden-markov-models]]
- [[cepstral-analysis-mfcc]]
