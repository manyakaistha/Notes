---
title: HMM–GMM Acoustic Modeling
tags: [speech/concept, acoustic-modeling, hmm, gmm, asr]
---

# HMM–GMM Acoustic Modeling

**Formal definition.** An HMM–GMM acoustic model assigns a latent Markov state to each feature frame and scores the observed vector using a state-conditional Gaussian mixture. For state $j$, $b_j(o_t)=\sum_{k=1}^{K_j}w_{jk}\mathcal N(o_t;\mu_{jk},\Sigma_{jk})$; state progression is governed by initial probabilities $\pi$ and transitions $A$. A left-to-right topology constrains progression through phone substates and permits variable duration through self-loops. A conventional three-emitting-state phone is a modeling choice rather than a physical law. Context-dependent triphones distinguish a central phone by neighboring phones, while state tying shares emission parameters across acoustically related states to control data sparsity. Viterbi decoding finds a joint maximum-probability state path and can be reused for hard alignment or realignment. Forward–backward instead computes soft state and transition posteriors for Baum–Welch estimation. The model therefore separates acoustic fit, encoded by emissions, from temporal plausibility, encoded by the state graph.

**Why it matters here.** Week 4 connects Week 3's GMM and Markov prerequisites into a classical speech recognition acoustic model. It explains why frame classification alone does not express phone duration or sequence constraints, and why context expansion requires parameter sharing.

**Appears in:** [[Knowledge-Map]] (Week 4); [[week 4/week-4-notes|Week 4 Notes]]

## Related
- [[hidden-markov-models]]
- [[gaussian-mixture-models]]
- [[viterbi-algorithm]]
- [[forward-backward-algorithm]]
- [[triphone-state-tying]]
- [[expectation-maximization]]
- [[baum-welch-algorithm]]
