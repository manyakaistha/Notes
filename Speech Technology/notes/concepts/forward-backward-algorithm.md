---
title: Forward-Backward Algorithm
tags: [speech/concept, sequence-modeling, hmm, inference]
---

# Forward-Backward Algorithm

**Formal definition.** For HMM parameters $\lambda=(\pi,A,B)$, the forward variable $\alpha_t(i)=P(o_{1:t},q_t=i\mid\lambda)$ obeys $\alpha_1(i)=\pi_i b_i(o_1)$ and $\alpha_t(j)=b_j(o_t)\sum_i\alpha_{t-1}(i)a_{ij}$. Summing terminal forward values gives the sequence evidence $P(O\mid\lambda)=\sum_i\alpha_T(i)$. The backward variable $\beta_t(i)=P(o_{t+1:T}\mid q_t=i,\lambda)$ obeys $\beta_T(i)=1$ and $\beta_t(i)=\sum_j a_{ij}b_j(o_{t+1})\beta_{t+1}(j)$. Only their combination yields smoothed state posteriors, $\gamma_t(i)=\alpha_t(i)\beta_t(i)/P(O\mid\lambda)$, and adjacent-state transition posteriors, $\xi_t(i,j)=\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)/P(O\mid\lambda)$. Forward evidence alone is not a smoothed posterior and does not provide these Baum–Welch sufficient statistics. Unlike Viterbi’s maximum over paths, the forward and backward recurrences sum probability mass over all compatible paths. Scaling or log-sum-exp is needed for numerical stability on long sequences.

**Why it matters here.** Week 4's Forward Algorithm lecture establishes total observation evidence; the backward pass and paired posterior calculations explain how Baum–Welch obtains soft occupancies and transition counts for parameter estimation. Viterbi can instead provide a hard path alignment, but does not equal the posterior distribution.

**Appears in:** [[Knowledge-Map]] (Week 4); [[week 4/week-4-notes|Week 4 Notes]]

## Related
- [[hidden-markov-models]]
- [[viterbi-algorithm]]
- [[gaussian-mixture-models]]
