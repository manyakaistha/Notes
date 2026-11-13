---
title: Gaussian Mixture Models
tags: [speech/concept, machine-learning/gmm, acoustic-modeling]
---

# Gaussian Mixture Models (GMMs)

**Formal definition.** A $K$-component Gaussian mixture model (GMM) represents a density over $x\in\mathbb R^D$ as $p(x)=\sum_{k=1}^K w_k\mathcal N(x;\mu_k,\Sigma_k)$, where $w_k\ge0$, $\sum_kw_k=1$, and each covariance is positive definite or regularized. A latent component indicator $z$ selects the Gaussian; its posterior responsibility for datum $x_i$ is $\gamma_{ik}=P(z_i=k\mid x_i)=w_k\mathcal N(x_i;\mu_k,\Sigma_k)/p(x_i)$. Unlike a single Gaussian, the weighted sum can represent multiple modes, but likelihood optimization is nonconvex and covariance collapse can make the unconstrained likelihood unbounded. Expectation-maximization alternates posterior responsibility calculation with weighted maximum-likelihood updates; each exact iteration does not decrease observed-data likelihood, but it need not find a global maximum.

**Why it matters here.** Week 3 generalizes the Week 2 single-Gaussian model to multimodal density estimation and develops parameter estimation using EM. Week 4 applies GMM densities as continuous acoustic emissions for HMM states, where the state sequence captures temporal structure and the mixture captures frame-level spectral variability. This acoustic use is distinct from interpreting mixture components as phonetic labels.

**Appears in:** [[week 3/week-3-notes|Week 3 Notes]]; [[week 4/week-4-notes|Week 4 Notes]]

## Related
- [[gaussian-distribution]]
- [[expectation-maximization]]
- [[k-means-clustering]]
- [[cepstral-analysis-mfcc]]
