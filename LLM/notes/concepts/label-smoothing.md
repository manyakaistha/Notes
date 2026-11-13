---
title: Label Smoothing
tags: [llm/concept, transformer, training]
---

# Label Smoothing

**Formal definition.** Label smoothing replaces a one-hot class target with a distribution that assigns some probability to alternatives. For vocabulary size $V$, gold class $y$, and smoothing coefficient $\epsilon$, a common uniform-mixture convention is $q_i=(1-\epsilon)\mathbf{1}[i=y]+\epsilon/V$. Training minimizes cross-entropy $-\sum_i q_i\log p_\theta(i)$ rather than $-\log p_\theta(y)$. Another convention distributes $\epsilon$ only among non-gold classes; the two produce different target probabilities and must not be conflated. Special tokens may also be excluded by an implementation. Szegedy et al. (2016) describe label smoothing as a regularization technique.

**Why it matters here.** Vaswani et al. (2017) report $\epsilon_{ls}=0.1$ for Transformer training; this value comes from the paper. Smoothing discourages concentrating all predicted probability on the observed label and changes the training objective, so its loss is not directly the unsmoothed negative log-likelihood used to calculate ordinary perplexity. The paper reports that smoothing hurts perplexity while improving accuracy and BLEU; an improvement is an empirical outcome, not a guarantee for every dataset.

**Appears in:** [[week-2-notes]]; [[week-2-exam-focused-notes]]

## Related
- [[teacher-forcing]]
- [[language-modelling-objective]]
- [[perplexity]]
