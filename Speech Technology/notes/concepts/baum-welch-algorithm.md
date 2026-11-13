---
title: Baum–Welch Algorithm
tags: [speech/concept, sequence-modeling, hmm, em, training]
---

# Baum–Welch Algorithm

**Formal definition.** Baum–Welch is expectation–maximization for hidden Markov models with incomplete state sequences. Given observations $O$ and current parameters $\lambda=(\pi,A,B)$, the E-step uses forward–backward to compute state occupancies $\gamma_t(i)=P(q_t=i\mid O,\lambda)$ and transition occupancies $\xi_t(i,j)=P(q_t=i,q_{t+1}=j\mid O,\lambda)$. For a single sequence, the maximum-likelihood updates include $\pi_i'=\gamma_1(i)$ and $a_{ij}'=\sum_{t=1}^{T-1}\xi_t(i,j)/\sum_{t=1}^{T-1}\gamma_t(i)$. Emission parameters are updated using posterior-weighted sufficient statistics; for GMM emissions, responsibilities additionally distribute each state's occupancy across mixture components. Exact EM iterations do not decrease the observed-data likelihood, but can converge to local optima and inherit mixture degeneracies. Scaling or log-domain arithmetic is required for long sequences.

**Why it matters here.** Week 4's HMM treatment distinguishes soft sequence training from selecting a single Viterbi path. Baum–Welch uses uncertainty over all state paths to estimate transitions and emissions; Viterbi hard realignment is a different training procedure.

**Appears in:** [[Knowledge-Map]] (Week 4); [[week 4/week-4-notes|Week 4 Notes]]

## Related
- [[expectation-maximization]]
- [[forward-backward-algorithm]]
- [[hidden-markov-models]]
- [[hmm-gmm-acoustic-modeling]]
- [[gaussian-mixture-models]]
