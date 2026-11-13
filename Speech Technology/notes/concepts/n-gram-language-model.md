---
title: "N-gram Language Model"
tags: [speech/concept, language-model, n-gram]
---

# N-gram Language Model

**Formal definition.** The chain rule expresses a word sequence exactly as $P(w_{1:T})=\prod_{t=1}^{T}P(w_t\mid w_{1:t-1})$, with explicit beginning-of-sequence history and an optional end-of-sequence token. An $(n-1)$-order context approximation replaces the full history by its most recent $n-1$ words: $P(w_t\mid w_{1:t-1})\approx P(w_t\mid w_{t-n+1:t-1})$. Maximum-likelihood estimates use counts from the same tokenization and boundary convention: $\hat P(w\mid h)=C(h,w)/C(h)$, where $C(h)=\sum_v C(h,v)$ counts outgoing continuations of history $h$. Thus bigrams condition on one prior word and trigrams on two. The chain-rule factorization is exact; the finite-context approximation and unsmoothed count estimate are modeling choices. For an observed history, an unseen continuation has MLE probability zero; a wholly unseen history has no uniquely identified MLE row. This sparsity motivates smoothing or backoff as extensions.

**Why it matters here.** The language model scores word-sequence plausibility alongside acoustic evidence and pronunciation constraints during ASR search.

**Source boundary.** The Week 5 curriculum screenshot lists n-gram models, but the retained lecture PDFs do not provide a dedicated count-estimation lesson. This definition is an authored derivation, not a lecture transcript.

**Appears in:** [[Knowledge-Map]], [[week 5/week-5-notes|Week 5]]

## Related
- [[asr-bayesian-decoding]]
- [[lexicon-pronunciation-dictionary]]
