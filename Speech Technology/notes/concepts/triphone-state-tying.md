---
title: Triphones and State Tying
tags: [speech/concept, acoustic-modeling, hmm, phonetics]
---

# Triphones and State Tying

**Formal definition.** A context-dependent triphone labels a central phoneme by its immediate left and right phonetic context, conventionally written $(l-c+r)$. This captures coarticulation: the acoustic realization of /t/ can differ depending on neighboring phones. Expanding every phone into all possible contexts creates many HMM states and sparse training counts. State tying clusters acoustically or phonologically related context-dependent substates and shares emission parameters across them; decision-tree questions about neighboring phones are a standard clustering method. The resulting tied states are often called senones. Tying trades model capacity for statistical robustness: overly broad clusters erase useful distinctions, while overly fine clusters overfit or remain poorly estimated. A triphone typically has internal emitting substates, so “triphone” denotes phonetic context, whereas “three-state” denotes temporal topology; the terms are not interchangeable.

**Why it matters here.** Context expansion and parameter sharing explain how classical recognizers represent coarticulation without requiring reliable data for every possible phone context.

**Appears in:** [[Knowledge-Map]] (Week 4); [[week 4/week-4-notes|Week 4 Notes]]

## Related
- [[hidden-markov-models]]
- [[gaussian-mixture-models]]
- [[lexicon-pronunciation-dictionary]]
