---
title: Greedy and Beam Search
tags: [llm/concept, decoding, search]
---

# Greedy and Beam Search

**Formal definition.** An autoregressive model assigns a sequence $y_{1:T}$ the probability $\prod_{t=1}^{T}P(y_t\mid y_{<t},x)$, including an end-of-sequence token when termination is modeled. Exhaustive search compares all allowed sequences, but its search space grows exponentially with the length bound. Greedy decoding instead selects the highest-probability next token at each step. This local choice need not maximize the probability of the complete sequence, because later conditional probabilities depend on the prefix. Beam search retains up to $b$ partial hypotheses at each step, expands them, and prunes by a score, commonly the sum of token log probabilities. A pruned prefix cannot later be recovered, so finite beam search does not guarantee a global optimum. Under the same expansion, scoring, and stopping rules, beam width one reduces to greedy search. Ended hypotheses and stopping rules require explicit handling. Length normalization changes the ranking objective and can mitigate preference for short outputs; it does not guarantee an appropriate length or improve every task.

**Why it matters here.** Week 4 distinguishes search error from model quality: finding a higher-probability sequence does not necessarily make open-ended text more useful or less repetitive.

**Appears in:** [[week-4-notes]]; [[week-4-exam-focused-notes]]

## Related
- [[causal-language-modelling]]
- [[top-k-top-p-sampling]]
- [[temperature-scaling]]
