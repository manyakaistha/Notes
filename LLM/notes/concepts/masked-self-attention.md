---
title: Masked Self-Attention
tags: [llm/concept, transformer, attention]
---

# Masked Self-Attention

**Formal definition.** Causal self-attention restricts each query position to keys at its own position or earlier. For $Q,K$ of shape `[T, d_k]` and $V$ of shape `[T, d_v]`, one head computes $\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,\qquad M_{ij}=\begin{cases}0&j\le i,\\-\infty&j>i.\end{cases}$ Softmax acts across keys, giving forbidden positions zero weight in exact arithmetic. The output has shape `[T, d_v]`. This assumes every query has at least one unmasked key; a fully masked row has undefined softmax and needs explicit handling. A padding mask separately excludes padded keys and can be combined with the causal mask.

**Why it matters here.** With shifted target inputs, masking prevents a decoder position from seeing future target information during parallel teacher-forced training. During incremental inference, a query attends to the available prefix; cached keys and values can avoid recomputing earlier projections. A causal mask restricts target-side information, whereas source padding in encoder–decoder attention is a separate concern.

**Appears in:** [[week-2-notes]]; [[week-2-exam-focused-notes]]

## Related
- [[teacher-forcing]]
- [[scaled-dot-product-attention]]
- [[cross-attention]]
- [[causal-language-modelling]]
