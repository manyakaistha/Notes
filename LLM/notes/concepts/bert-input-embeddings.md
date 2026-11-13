---
title: BERT Input Embeddings
tags: [llm/concept, bert, embeddings]
---

# BERT Input Embeddings

**Formal definition.** Original BERT represents each input position by the sum of learned token, position, and segment embeddings. For token id $w_i$, position $i$, and segment id $s_i$, the embedding is $e_i=E_{\mathrm{token}}[w_i]+E_{\mathrm{position}}[i]+E_{\mathrm{segment}}[s_i]$, with each term a vector of dimension `[H]`. WordPiece tokenization supplies the token ids, while learned absolute position vectors distinguish sequence order. Segment embeddings distinguish the A and B portions of a pair; they identify segments rather than individual natural sentences. A standard pair takes the form `[CLS]` A `[SEP]` B `[SEP]`, while a single sequence uses `[CLS]` A `[SEP]`. The special tokens have learned token embeddings like other vocabulary entries. After the embedding sum, BERT applies embedding layer normalization and dropout before the encoder stack. The final contextual `[CLS]` representation supports sequence-level classification, whereas `[SEP]` marks boundaries; neither is interchangeable with the corruption token `[MASK]`. Padding positions are handled by an attention mask, not by segment embeddings alone.

**Why it matters here.** The additive representation lets the same encoder combine lexical identity, position, and pair membership without concatenating three separate feature vectors.

**Appears in:** [[week-4-notes]]; [[week-4-exam-focused-notes]]

## Related
- [[bert-masked-language-modelling]]
- [[next-sentence-prediction]]
- [[sinusoidal-positional-encoding]]
- [[layer-normalization]]
