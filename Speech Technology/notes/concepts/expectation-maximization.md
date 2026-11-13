---
title: Expectation-Maximization for Gaussian Mixtures
tags: [speech/concept, statistics, gmm, optimization]
---

# Expectation-Maximization for Gaussian Mixtures

**Formal definition.** Expectation-maximization (EM) is an iterative likelihood method for models with latent variables. For mixture observations $x_1,\ldots,x_N$, let $z_i\in\{1,\ldots,K\}$ identify the Gaussian component. At iteration $r$, the E-step computes responsibilities $\gamma_{ik}=P(z_i=k\mid x_i,\theta^{(r)})$. The M-step maximizes the expected complete-data log likelihood, yielding $N_k=\sum_i\gamma_{ik}$, $w_k=N_k/N$, $\mu_k=N_k^{-1}\sum_i\gamma_{ik}x_i$, and $\Sigma_k=N_k^{-1}\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^T$. These updates define a coordinate ascent on a lower bound to observed-data log likelihood: exact iterations do not decrease that likelihood, but convergence can be to a local optimum or degenerate solution. Initialization, covariance regularization, and stopping criteria therefore affect the result. Responsibilities are soft assignments; they are not observed labels.

**Why it matters here.** Week 3 develops parameter estimation for GMMs, where the component identity is unknown. EM supplies a tractable alternative to directly optimizing the coupled mixture likelihood and clarifies why soft membership updates precede parameter re-estimation.

**Appears in:** [[week 3/week-3-notes|Week 3 Notes]]; [[week 4/week-4-notes|Week 4 Notes]]

## Related
- [[gaussian-mixture-models]]
- [[gaussian-distribution]]
- [[forward-backward-algorithm]]
- [[hidden-markov-models]]
