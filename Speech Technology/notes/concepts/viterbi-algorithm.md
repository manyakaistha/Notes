---
title: Viterbi Decoding Algorithm
tags: [speech/concept, dynamic-programming, viterbi, decoding]
---

# Viterbi Decoding Algorithm

**Formal definition.** Viterbi is max-product dynamic programming for the most probable path through a finite-state chain. For the Week 3 forecast, the current state is known and future states are unknown predictions, not hidden causes of acoustic observations. Set $\delta_0(i)=1$ for the known starting state and zero elsewhere; with transition probabilities $a_{ij}$, compute $\delta_t(j)=\max_i[\delta_{t-1}(i)a_{ij}]$. Store the maximizing predecessor $\psi_t(j)$, select the best terminal score, and follow those backpointers to recover the entire path. At each state and time, all continuations share the same future conditional distribution, so a lower-scoring prefix ending there cannot improve the final optimum. Log probabilities turn products into sums and avoid underflow. For a dense $N$-state chain the cost is $O(TN^2)$, compared with enumerating $N^T$ future paths. Viterbi returns a joint maximum-probability sequence, not the separate marginal forecast at each time. HMM decoding generalizes the recurrence by multiplying an emission likelihood into each state score.

**Why it matters here.** Week 3 applies the recurrence to weather prediction in a Markov chain; Week 4 adds emission likelihoods for HMM decoding. In speech modeling, a Viterbi path may also provide a hard alignment or realignment for training. This is distinct from Baum–Welch's soft posterior occupancies, and Viterbi is not limited to recognition-time decoding.

**Appears in:** [[week 3/week-3-notes|Week 3 Notes]]; [[week 4/week-4-notes|Week 4 Notes]]

## Related
- [[hidden-markov-models]]
- [[markov-chain]]
- [[forward-backward-algorithm]]
- [[lexicon-pronunciation-dictionary]]
