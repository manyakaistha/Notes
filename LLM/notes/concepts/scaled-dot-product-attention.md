---
title: Scaled Dot-Product Attention
tags: [llm/concept, attention, transformer]
---

# Scaled Dot-Product Attention

**Formal definition.** For row-oriented queries $Q$, keys $K$, and values $V$, scaled dot-product attention is

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}_{\rm keys}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.$$

For shapes `[B, T_q, d_k]`, `[B, T_k, d_k]`, and `[B, T_k, d_v]`, respectively, the score and output shapes are `[B, T_q, T_k]` and `[B, T_q, d_v]`. Mask $M$ excludes forbidden keys by assigning their logits negative infinity; each query needs at least one eligible key. If query and key coordinates are independent, zero-mean, unit-variance variables, their dot product has variance $d_k$; division by $\sqrt{d_k}$ gives unit variance under those assumptions. This mitigates, but cannot guarantee avoidance of, softmax saturation for learned vectors.

**Why it matters here.** The ebook claims its column-input formula and the paper's row-input formula give outputs that are transposes of one another. With $Q_{\rm col}=Q_{\rm row}^{\top}$ and similarly for $K,V$, both displayed formulas produce the same row-oriented output; only a column-oriented output is its transpose. Source: Vaswani et al. (2017), “Attention Is All You Need,” Advances in Neural Information Processing Systems.

**Appears in:** [[week-1-notes]]; [[week-1-exam-focused-notes]]

## Related
- [[self-attention]]
- [[multi-head-attention]]
- [[masked-self-attention]]
