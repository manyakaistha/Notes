---
title: Layer Normalization
tags: [llm/concept, transformer, normalization]
---

# Layer Normalization

**Formal definition.** For a token representation $x$ of shape `[d]`, layer normalization calculates $\mu=d^{-1}\sum_i x_i$ and $\sigma^2=d^{-1}\sum_i(x_i-\mu)^2$, then returns $y_i=\gamma_i(x_i-\mu)/\sqrt{\sigma^2+\epsilon}+\beta_i$. The learned scale and offset have shape `[d]` and are shared across positions; statistics are computed independently for each token and example. The positive $\epsilon$ stabilizes division. Ba, Kiros and Hinton (2016) introduced layer normalization with statistics computed within an individual case, rather than across a minibatch.

**Why it matters here.** In the original Transformer, each sublayer uses post-normalization: $\operatorname{LN}(x+\operatorname{Sublayer}(x))$. Per-token statistics do not depend on batch composition and apply consistently to variable-length sequences. The ebook incorrectly calls single-value BatchNorm a no-op: with one value in the normalization set its normalized activation is zero and its affine output is $\beta$. BatchNorm may also aggregate across spatial or sequence positions, so batch size one does not invariably imply a single normalization value. Normalization does not itself guarantee stable optimization.

**Appears in:** [[week-2-notes]]; [[week-2-exam-focused-notes]]

## Related
- [[residual-connections]]
- [[learning-rate-warmup]]
- [[position-wise-feed-forward]]
