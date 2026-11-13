---
title: Temperature Scaling
tags: [llm/concept, decoding, probability]
---

# Temperature Scaling

**Formal definition.** For finite vocabulary logits $z_i$ and temperature $T>0$, temperature scaling defines $P_T(i)=\exp(z_i/T)/\sum_j\exp(z_j/T)$. Dividing logits by a positive temperature preserves their ordering but changes their probability ratios: $P_T(i)/P_T(j)=\exp((z_i-z_j)/T)$. Temperatures below one sharpen the distribution relative to the original softmax, while temperatures above one flatten it. As $T\to0^+$, probability concentrates on the maximum-logit tokens; if several maxima tie, the limiting distribution is uniform over those ties. As $T\to\infty$, it approaches the uniform distribution over finite logits. Zero is not a valid divisor, though implementations sometimes use it as shorthand for deterministic argmax. Temperature is a distribution transformation, not itself a sampling algorithm; a token can subsequently be sampled or selected deterministically. With top-$p$ truncation, scaling first can change which tokens enter the nucleus, whereas top-$k$ membership is unchanged apart from ties.

**Why it matters here.** Week 4 uses temperature to control the probability distribution presented to decoding, while recognizing that neither low nor high temperature guarantees factuality or quality.

**Appears in:** [[week-4-notes]]; [[week-4-exam-focused-notes]]

## Related
- [[top-k-top-p-sampling]]
- [[greedy-and-beam-search]]
- [[language-modelling-objective]]
