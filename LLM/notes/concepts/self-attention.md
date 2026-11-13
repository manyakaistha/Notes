---
title: Self-Attention
tags: [llm/concept, attention, transformer]
---

# Self-Attention

**Formal definition.** Self-attention constructs each output from values drawn from the same sequence that supplies its queries and keys. For input $X$ of shape `[B, T, d_model]`, learned projections give $Q=XW^Q$, $K=XW^K$, and $V=XW^V$, with shapes `[B, T, d_k]`, `[B, T, d_k]`, and `[B, T, d_v]`. Each query is compared with eligible keys; row-wise normalized scores weight their corresponding values. The result has shape `[B, T, d_v]` and is contextual: a position can incorporate information from other positions and itself. Padding or causal masks restrict eligible keys. Without position information or an order-dependent mask, the operation is permutation equivariant, so position must be represented separately when order matters.

**Why it matters here.** Unlike recurrent processing, an encoder's self-attention can compute all query positions in parallel, while its dense score matrix has `[B, T, T]` entries per head. Parallel computation does not make its quadratic sequence-length cost disappear. In cross-attention, queries and keys/values instead come from different sequences. Source: Vaswani et al. (2017), “Attention Is All You Need,” Advances in Neural Information Processing Systems.

**Appears in:** [[week-1-notes]]; [[week-1-exam-focused-notes]]

## Related
- [[scaled-dot-product-attention]]
- [[multi-head-attention]]
- [[cross-attention]]
- [[sinusoidal-positional-encoding]]
