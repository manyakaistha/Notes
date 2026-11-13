---
title: Hidden Markov Models in Speech
tags: [speech/concept, sequence-modeling, hmm, asr]
---

# Hidden Markov Models in Speech

**Formal definition.** A hidden Markov model (HMM) is specified by $\lambda=(\pi,A,B)$: initial-state probabilities $\pi_i=P(q_1=i)$, transition probabilities $a_{ij}=P(q_t=j\mid q_{t-1}=i)$, and state-conditional observation distributions $b_j(o)=P(o_t=o\mid q_t=j)$. For latent states $Q=(q_1,\ldots,q_T)$ and observations $O=(o_1,\ldots,o_T)$, $P(Q,O\mid\lambda)=\pi_{q_1}b_{q_1}(o_1)\prod_{t=2}^T a_{q_{t-1}q_t}b_{q_t}(o_t)$. The Markov assumption concerns the latent state given its predecessor; conditional independence of observations given states is a separate assumption. In speech systems, continuous feature frames are often scored by GMM state emissions; left-to-right phone topologies represent temporal progression. Context-dependent triphones distinguish a central phone by neighboring phones, and state tying shares parameters where data are sparse. Three emitting states per phone is a common convention, not a universal law.

**Why it matters here.** Week 4 introduces the hidden-state version of the Week 3 Markov chain and adds emissions so that acoustic frames provide evidence about a latent phone-state path. The forward algorithm sums the probability mass of all paths to compute $P(O\mid\lambda)$; forward–backward combines prefix and suffix evidence to obtain smoothed state and transition posteriors used by Baum–Welch. Viterbi instead returns one joint maximum-probability path and can support hard alignment or realignment. The three calculations answer different questions: evidence, posterior occupancy, and best path. GMM emissions model continuous feature vectors, while topology and state tying encode speech dynamics and coarticulation.

**Appears in:** [[Knowledge-Map]] (Week 4); [[week 4/week-4-notes|Week 4 Notes]]

## Related
- [[gaussian-mixture-models]]
- [[viterbi-algorithm]]
- [[forward-backward-algorithm]]
- [[triphone-state-tying]]
- [[lexicon-pronunciation-dictionary]]
