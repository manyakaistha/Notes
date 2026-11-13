---
title: "Discrete and Continuous Observation HMMs"
tags: [speech/concept, hmm, acoustic-modeling]
---

# Discrete and Continuous Observation HMMs

**Formal definition.** A hidden Markov model (HMM) has latent states $q_t$ and observations $o_t$. In a discrete-observation HMM, $o_t$ takes values from a finite alphabet and each state has categorical emission probabilities $b_j(k)=P(o_t=k\mid q_t=j)$. Speech systems may quantize frame vectors to codebook indices before applying such a model. In a continuous-observation HMM, each $o_t\in\mathbb R^d$ is a feature vector and state $j$ specifies a density $b_j(o)=p(o\mid q_t=j)$, for example a finite Gaussian mixture. Probability of an exact point in a continuous space is generally zero; the model assigns density, whose integral over a region gives probability. Both models remain discrete-time latent-state chains: “continuous” describes the observation space, not continuous time. A finite $K$-component GMM has $K$ components; it is not an infinite-component model.

**Why it matters here.** Speech frames are continuous-valued acoustic vectors, so recognizing the observation model clarifies why GMM-HMM emissions are densities rather than a table of symbol probabilities.

**Appears in:** [[Knowledge-Map]], [[week 4/week-4-notes|Week 4]], [[week 5/week-5-notes|Week 5]]

## Related
- [[hidden-markov-models]]
- [[gaussian-mixture-models]]
- [[hmm-topologies]]
