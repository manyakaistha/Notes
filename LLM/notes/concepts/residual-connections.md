---
title: Residual Connections
tags: [llm/concept, transformer, optimization]
---

# Residual Connections

**Formal definition.** A residual connection adds a transformation $F(x)$ to its input: $y=x+F(x)$, with both operands of matching shape `[d]` at each token. He et al. (2016) formulate residual learning as learning $F$ relative to an identity shortcut. For this bare addition, the Jacobian is $I+J_F$, giving gradients a direct additive path alongside the transformed path. This does not ensure that the learned residual is small or that gradients cannot vanish or explode. Projections can align dimensions when the shortcut and transformation otherwise differ.

**Why it matters here.** Residual additions wrap each attention and feed-forward sublayer in the original Transformer. Its post-normalized form is $\operatorname{LN}(x+\operatorname{Sublayer}(x))$, so the Jacobian also includes the LayerNorm Jacobian; it is not simply $I+J_F$. In pre-normalization, $x+\operatorname{Sublayer}(\operatorname{LN}(x))$ instead retains an unnormalized identity path across the complete block. The ebook's characterization of the residual as necessarily a small correction is an intuition, not a mathematical constraint.

**Appears in:** [[week-2-notes]]; [[week-2-exam-focused-notes]]

## Related
- [[layer-normalization]]
- [[multi-head-attention]]
- [[position-wise-feed-forward]]
- [[learning-rate-warmup]]
