---
title: Gaussian Binary Classification
tags: [speech/concept, probability, classification]
---

# Gaussian Binary Classification

**Formal definition.** Let $Y\in\{0,1\}$ be a class label and $X\in\mathbb R^D$ an observation. Gaussian binary classification assigns each class a prior $\pi_c=P(Y=c)$ and a class-conditional density $p(x\mid Y=c)=\mathcal N(x;\mu_c,\Sigma_c)$. Bayes' rule gives the posterior $P(Y=c\mid x)=\frac{\pi_c\mathcal N(x;\mu_c,\Sigma_c)}{\sum_{r\in\{0,1\}}\pi_r\mathcal N(x;\mu_r,\Sigma_r)}$; under zero-one loss, predict the class with the larger posterior, equivalently the larger log prior plus log density. If both classes share covariance, the log-likelihood ratio is affine in $x$ and the decision boundary is linear. Distinct covariance matrices generally yield a quadratic boundary. The rule depends on model adequacy, priors, and the chosen error costs; a larger class-conditional density alone does not always imply a larger posterior.

**Why it matters here.** Week 3 uses this model to connect the single-Gaussian density and covariance studied in Week 2 to supervised decisions. It makes the roles of class priors, overlap, and covariance explicit before extending a single density to a multimodal mixture.

**Appears in:** [[week 3/week-3-notes|Week 3 Notes]]

## Related
- [[gaussian-distribution]]
- [[gaussian-mixture-models]]
