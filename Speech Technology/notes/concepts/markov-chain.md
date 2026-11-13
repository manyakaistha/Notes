---
title: Markov Chain
tags: [speech/concept, probability, sequence-modeling]
---

# Markov Chain

**Formal definition.** A time-homogeneous discrete-time Markov chain is a stochastic process $(S_t)_{t\ge0}$ over a finite or countable state space $\mathcal S$ satisfying the first-order Markov property $P(S_{t+1}=j\mid S_t=i,S_{t-1},\ldots,S_0)=P(S_{t+1}=j\mid S_t=i)=a_{ij}$. Its transition matrix $A=[a_{ij}]$ is row-stochastic: $a_{ij}\ge0$ and $\sum_j a_{ij}=1$. Given current state distribution $\boldsymbol p_t$ as a row vector, the one-step distribution is $\boldsymbol p_{t+1}=\boldsymbol p_tA$, and the $h$-step distribution is $\boldsymbol p_tA^h$. For a fully observed chain, the state is known at each observed time; prediction can rank future paths by their transition-probability products. This differs from a hidden Markov model, which adds an unobserved state process and an observation or emission model. A Markov chain describes conditional dependence, not causal influence.

**Why it matters here.** Week 3 uses a weather-state transition model for sequence prediction. The row-stochastic matrix gives a compact way to represent the next-day conditional distribution, while Viterbi identifies the most probable state path when a sequence-level answer is needed.

**Appears in:** [[week 3/week-3-notes|Week 3 Notes]]

## Related
- [[viterbi-algorithm]]
- [[hidden-markov-models]]
