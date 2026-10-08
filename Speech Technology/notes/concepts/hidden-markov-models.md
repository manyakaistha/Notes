---
title: Hidden Markov Models in Speech
tags: [speech/concept, sequence-modeling, hmm, asr]
---

# Hidden Markov Models in Speech

A **Hidden Markov Model (HMM)** is a doubly stochastic statistical model where the underlying state sequence $Q = (q_1, \dots, q_T)$ is an unobservable (hidden) Markov process, while the system generates an observable continuous acoustic vector sequence $O = (o_1, \dots, o_T)$.

### 5-Tuple Formulation $\lambda = (S, V, A, B, \pi)$
1. **States ($S$):** Sub-phonetic acoustic units (typically 3 states per phone: onset, steady, offset).
2. **Observations ($V$):** Continuous $D$-dimensional feature vectors ($V = \mathbb{R}^{39}$ for MFCCs).
3. **Transition Probabilities ($A$):** $a_{ij} = P(q_{t+1} = s_j \mid q_t = s_i)$. In speech, models use a **Left-to-Right (Bakis)** topology where $a_{ij} = 0$ for $j < i$.
4. **Observation Densities ($B$):** $b_j(o_t) = P(o_t \mid q_t = s_j)$, parameterized by continuous Gaussian Mixture Models (GMMs) or Deep Neural Networks (DNNs).
5. **Initial State Distribution ($\pi$):** Prior probabilities of starting states ($\pi_1 = 1$, $\pi_i = 0$ for $i > 1$).

### Tri-State vs. Tri-Phone
- **Tri-State Model:** Decomposes a single phoneme into 3 temporal stages (onset, steady, coda) to model intra-phone spectral variation.
- **Tri-Phone Model:** Conditions a phoneme on its preceding and succeeding phonemes (`L - P + R`) to capture inter-phone coarticulation.

**Appears in:** [[week 2/week-2-notes|Week 2 Notes]]

## Related
- [[gaussian-mixture-models]]
- [[viterbi-algorithm]]
- [[lexicon-pronunciation-dictionary]]
