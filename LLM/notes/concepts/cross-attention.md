---
title: Cross-Attention
tags: [llm/concept, transformer, attention]
---

# Cross-Attention

**Formal definition.** Cross-attention computes queries from one sequence and keys and values from another. In the original Transformer decoder, decoder representations supply $Q$ and the final encoder output supplies $K,V$. For one head, $Q$ has shape `[T_tgt, d_k]`, $K$ `[T_src, d_k]`, and $V$ `[T_src, d_v]`. The head output is $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$, with shape `[T_tgt, d_v]`; its attention matrix has shape `[T_tgt, T_src]`. Multi-head attention uses separate learned projections and combines head outputs with an output projection. Each decoder layer has its own projections even though it attends to the same final encoder output. Thus each target position obtains a weighted combination of source value vectors, with weights normalized over source positions.

**Why it matters here.** Cross-attention provides a target-dependent route to source information at every decoding position, avoiding reliance on a single fixed-length source vector. In ordinary translation the full source is available, so target causality does not require a triangular mask over source positions; source padding still needs masking. Streaming or other restricted-source settings can impose additional visibility constraints. Attention weights are learned routing coefficients, not necessarily faithful explanations of a prediction.

**Appears in:** [[week-2-notes]]; [[week-2-exam-focused-notes]]

## Related
- [[bahdanau-attention]]
- [[multi-head-attention]]
- [[masked-self-attention]]
- [[seq2seq-bottleneck]]
