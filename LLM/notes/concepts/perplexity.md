---
title: Perplexity
tags: [llm/concept, language-modeling, evaluation]
---

# Perplexity

**Formal definition.** Perplexity is the exponential of mean negative log-likelihood over $N$ scored tokens, using natural logarithms:

$$\mathrm{PPL}=\exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(w_i\mid w_{<i})\right).$$

It is the reciprocal geometric mean of the probabilities assigned to the observed tokens. Lower perplexity means higher average log probability on the same evaluation data; a uniform distribution over $V$ tokens has perplexity $V$. The quantity is dimensionless and measures predictive fit under a specified context and scoring protocol, not the actual number of plausible next words. Comparisons require matching tokenization, evaluated text, treatment of start and end tokens, and context-window handling: changing the unit of prediction changes the denominator and generally changes perplexity. A zero probability for a scored token yields infinite perplexity. Perplexity alone does not establish that generated text is useful, factual, or safe, and training-set perplexity does not measure held-out generalization.

**Why it matters here.** It connects the Week 3 likelihood objective to an interpretable held-out evaluation measure. This definition is added from the literature; perplexity is not defined in the ebook transcript.

**Appears in:** [[week-3-notes]]; [[week-3-exam-focused-notes]]

## Related
- [[language-modelling-objective]]
- [[causal-language-modelling]]
- [[bert-masked-language-modelling]]
