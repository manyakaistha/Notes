---
title: Byte Pair Encoding (BPE)
slug: byte-pair-encoding
tags: ['vt/concept', 'vt/nlp', 'vt/tokenization', 'vt/transformer']
introduced: "[[w1d2|W1D2]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Byte Pair Encoding (BPE)

Byte Pair Encoding (BPE) is a sub-word tokenization algorithm originally developed by Gage (1994) as a data compression heuristic and adapted by Sennrich et al. (2016) for modern sequence modeling and large language models (LLMs).

BPE resolves the fundamental trade-off between **word-based** and **character-based** tokenization:
- **Word-based Tokenization:** Yields compact sequences but suffers from out-of-vocabulary (OOV) failure on unseen words, fails to share morphological roots (`cat` vs. `cats`, `eat` vs. `eating`), and requires massive vocabularies ($170,000–200,000+$ words).
- **Character-based Tokenization:** Has a small vocabulary ($26$ letters plus symbols) and eliminates OOV, but destroys word semantics and produces excessively long token sequences, exacerbating $\mathcal{O}(N^2)$ attention costs.

### The BPE Algorithm
1. **Base Initialization:** Initialize vocabulary $\mathcal{V}$ with all unique atomic characters in the training corpus.
2. **Frequency Counting:** Segment all training words into individual characters. Count the frequencies of all adjacent symbol pairs.
3. **Iterative Merging:** Identify the most frequently occurring symbol pair $(c_1, c_2)$ and merge it into a single new sub-word token $c_{new} = c_1 c_2$. Add $c_{new}$ to vocabulary $\mathcal{V}$.
4. **Stopping Criterion:** Repeat step 3 iteratively until the vocabulary reaches a predetermined size limit (e.g., $50,257$ in GPT-2/GPT-3) or no frequent pairs remain.

Frequent words eventually become single tokens, while rare, specialized, or inflected words (e.g., medical terms like `"ozempic"`) are decomposed into recognizable constituent sub-words without OOV failure.

**Why it matters here:** BPE governs the discrete tokenization stage of Transformer language models. In Vision Transformers, the direct analog to BPE sub-words is the non-overlapping image patch ($P \times P$).

**Appears in:** [[w1d2|W1D2]] · [[week-1-report|Week 1]]

## Related
- [[token-embedding]]
- [[patch-projection]]
- [[vision-transformer]]
