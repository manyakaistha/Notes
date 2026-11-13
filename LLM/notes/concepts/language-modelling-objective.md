---
title: Language-Modelling Objective
tags: [llm/concept, language-modeling, likelihood]
---

# Language-Modelling Objective

**Formal definition.** A language model assigns probabilities to token sequences. The probability chain rule factorizes a sequence $w_{1:T}$ as $p_\theta(w_{1:T})=\prod_{i=1}^{T}p_\theta(w_i\mid w_{<i})$, with a specified start context when needed. This identity does not assume token independence; a finite-context model instead approximates each conditional using an available prefix or context window. Maximum-likelihood training maximizes the sum of observed conditional log probabilities, equivalently minimizing negative log-likelihood $-\sum_i\log p_\theta(w_i\mid w_{<i})$. For one-hot targets this is token-level cross-entropy; dividing by the number of scored tokens gives the mean loss. Parameters are shared across positions, and each conditional is normalized over the vocabulary. Sequence probability also depends on how termination and tokenization are defined. A high likelihood indicates agreement with observed sequences under this scoring rule, not a proof of factual accuracy or linguistic understanding.

**Why it matters here.** GPT-1 applies this objective to unlabeled text before supervised adaptation. The chain rule explains how local next-token predictions form a joint model without the ebook's rejected independence assumption.

**Appears in:** [[week-3-notes]]; [[week-3-exam-focused-notes]]

## Related
- [[causal-language-modelling]]
- [[perplexity]]
- [[teacher-forcing]]
- [[pretrain-finetune-paradigm]]
