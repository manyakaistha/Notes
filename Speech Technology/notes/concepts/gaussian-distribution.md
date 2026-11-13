---
title: Single Gaussian Distribution, Covariance and Maximum Likelihood
tags: [speech/concept, probability, statistics]
---

# Single Gaussian Distribution, Covariance and Maximum Likelihood

**Formal definition.** A nondegenerate $D$-dimensional Gaussian distribution with mean vector $\boldsymbol{\mu}\in\mathbb{R}^D$ and symmetric positive-definite covariance matrix $\boldsymbol{\Sigma}\in\mathbb{R}^{D\times D}$ has density

$$p(\mathbf{x})=(2\pi)^{-D/2}|\boldsymbol{\Sigma}|^{-1/2}\exp\left[-\tfrac12(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right].$$

The mean sets the distribution's center; the covariance specifies component variances on its diagonal and pairwise covariances off the diagonal. Given independent observations $\mathbf{x}_1,\ldots,\mathbf{x}_N$ from one Gaussian, the maximum-likelihood estimates are $\hat{\boldsymbol{\mu}}=N^{-1}\sum_n\mathbf{x}_n$ and $\hat{\boldsymbol{\Sigma}}=N^{-1}\sum_n(\mathbf{x}_n-\hat{\boldsymbol{\mu}})(\mathbf{x}_n-\hat{\boldsymbol{\mu}})^T$. This covariance estimate uses divisor $N$; the unbiased sample covariance instead uses $N-1$. A single Gaussian is one distribution, not a mixture or a sequential model.

In the bivariate case, the covariance off-diagonal controls the orientation of equal-density ellipses; zero covariance makes the coordinates independent only under this joint Gaussian model. For class-conditional classification, Bayes' rule combines the density with class priors. A mixture replaces one Gaussian density with a weighted sum of several such densities, allowing multimodal structure while retaining Gaussian components.

**Why it matters here.** Week 2 introduces single-Gaussian densities, covariance, and maximum-likelihood estimates. Week 3 uses bivariate Gaussian geometry and classification as a bridge to multimodal Gaussian mixtures. Mean, covariance, and priors determine both the shape of class evidence and the resulting decision boundary.

**Appears in:** [[week-2-notes|Week 2 Notes]]; [[week 3/week-3-notes|Week 3 Notes]]

## Related
- [[cepstral-analysis-mfcc]]
- [[gaussian-mixture-models]]
- [[gaussian-binary-classification]]
