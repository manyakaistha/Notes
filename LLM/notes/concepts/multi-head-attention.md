---
title: Multi-Head Attention
tags: [llm/concept, attention, transformer]
---

# Multi-Head Attention

**Formal definition.** Multi-head attention applies independently learned query, key, and value projections for each head, concatenates the resulting attention outputs along their feature axis, and applies an output projection. Using row vectors, $\operatorname{head}_i=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V),\quad \operatorname{MultiHead}=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O.$ For self-attention input `[B, T, d_model]`, each head has output `[B, T, d_v]`; concatenation has `[B, T, h*d_v]`, and $W^O$ maps that width to $d_{\rm model}$. In the original base configuration $d_k=d_v=d_{\rm model}/h$. Heads can learn different relations, but specialization is not guaranteed. $W^O$ learns a linear combination across head features; subsequent layers can also mix those features.

**Why it matters here.** The ebook writes $W^O\operatorname{Concat}(\operatorname{head}_i)$ while its preceding heads are row-oriented. With those shapes the projection multiplies on the right, as in the original paper; left multiplication requires transposing the concatenated output and using the corresponding column-oriented projection. Source: Vaswani et al. (2017), “Attention Is All You Need,” Advances in Neural Information Processing Systems.

**Appears in:** [[week-1-notes]]; [[week-1-exam-focused-notes]]

## Related
- [[self-attention]]
- [[scaled-dot-product-attention]]
- [[encoder-layer-parameter-count]]
