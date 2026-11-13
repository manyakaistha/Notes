---
title: "Bayesian Decoding in Automatic Speech Recognition"
tags: [speech/concept, asr, decoding]
---

# Bayesian Decoding in Automatic Speech Recognition

**Formal definition.** Automatic speech recognition seeks a word sequence $W^*$ for acoustic observations $X$. Bayes’ rule gives $W^*=\arg\max_W P(W\mid X)=\arg\max_W P(X\mid W)P(W)$ because the evidence $P(X)$ is constant across candidates. A pronunciation lexicon relates word sequences to phone sequences and possible acoustic-state paths. In a generative HMM account, the acoustic likelihood may marginalize over those latent pronunciations and state paths; practical decoders commonly search a composed graph and use a max-path approximation such as Viterbi. The language-model prior $P(W)$ may be supplied by an n-gram model. Implementations often introduce tunable language-model weights, insertion penalties, or score scaling, so an operational decoder score need not equal an unweighted Bayesian posterior. The acoustic model, lexicon, and word prior constrain a joint search; they are not necessarily executed as three hard serial stages.

**Why it matters here.** It explains how Week 4 HMM acoustic scores connect to Week 5 pronunciation and word-sequence models without mistaking phone decoding for the final word decision.

**Appears in:** [[Knowledge-Map]], [[week 5/week-5-notes|Week 5]]

## Related
- [[hidden-markov-models]]
- [[lexicon-pronunciation-dictionary]]
- [[n-gram-language-model]]
- [[viterbi-algorithm]]
