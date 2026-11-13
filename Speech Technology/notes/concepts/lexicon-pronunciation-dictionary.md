---
title: Lexicon (Pronunciation Dictionary)
tags: [speech/concept, lexicon, phonetics, asr]
---

# Lexicon (Pronunciation Dictionary)

**Formal definition.** An ASR pronunciation lexicon maps each orthographic vocabulary item to one or more permissible phoneme sequences, optionally with variant-specific probabilities or costs. For example, a word may map to /k ae t/ in a chosen phoneme inventory. The lexicon is composed with acoustic and language-model constraints during decoding: acoustic scores evaluate observations against phone or state hypotheses, the lexicon constrains which pronunciations realize words, and the language model scores word sequences. Thus the lexicon is not merely a post-processing lookup from an already decoded phone string; in large-vocabulary systems it participates in the search graph, including weighted finite-state representations. ARPAbet is one phoneme-label inventory and is not interchangeable with IPA without a mapping. A grapheme-to-phoneme (G2P) model can propose pronunciations for new spellings, but it does not automatically guarantee that an out-of-vocabulary word is represented in the decoder vocabulary or graph. Pronunciation variants, stress, silence, and context-dependent phone expansion require explicit conventions. Errors or omissions in the dictionary can make otherwise acoustically plausible word sequences unreachable.

**Why it matters here.** It connects subword acoustic modeling to word hypotheses and makes the boundary between pronunciation, acoustics, and linguistic context explicit.

**Appears in:** [[Knowledge-Map]], [[week 5/week-5-notes|Week 5]]

## Related
- [[hidden-markov-models]]
- [[triphone-state-tying]]
