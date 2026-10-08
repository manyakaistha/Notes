---
title: Lexicon (Pronunciation Dictionary)
tags: [speech/concept, lexicon, phonetics, asr]
---

# Lexicon (Pronunciation Dictionary)

In automatic speech recognition, the **Lexicon** (pronunciation dictionary) is the architectural mapping lookup table that bridges sub-word phonetic units output by the acoustic-HMM model into orthographic words recognized by the language model:

$$\text{Phoneme String } (/k/ , /ae/ , /t/) \xrightarrow{\text{Lexicon}} \text{Word Unit } ("\text{CAT}")$$

### Architectural Boundary
- **Acoustic Model (HMM-GMM / HMM-DNN):** Responsible strictly for evaluating acoustic feature vectors into phonemic state likelihoods and sequences. It has no intrinsic knowledge of word spellings.
- **Lexicon:** Specifies the phonetic transcriptions of each vocabulary word. Allows multiple pronunciation entries for dialectal variants.
- **Language Model:** Assigns syntax and context priors $P(W)$ over words.

**Appears in:** [[week 2/week-2-notes|Week 2 Notes]]

## Related
- [[hidden-markov-models]]
- [[speech-chain]]
