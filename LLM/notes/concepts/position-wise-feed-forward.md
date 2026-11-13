---
title: Position-Wise Feed-Forward Network
tags: [llm/concept, transformer, feed-forward]
---

# Position-Wise Feed-Forward Network

**Formal definition.** The original Transformer applies the same two-layer feed-forward network independently to each sequence position. For a row vector $x$ of width $d_{\rm model}$, $\operatorname{FFN}(x)=\operatorname{ReLU}(xW_1+b_1)W_2+b_2,$ where $W_1$ has shape `[d_model, d_ff]`, $b_1$ `[d_ff]`, $W_2$ `[d_ff, d_model]`, and $b_2$ `[d_model]`. A batch `[B, T, d_model]` retains that shape after this transformation. The matrices and biases are shared across positions within a layer, but different layers have their own parameters. The nonlinearity makes the mapping more expressive than one affine transformation. Because no position reads another position inside this sublayer, token-to-token mixing is performed by attention, not by the position-wise FFN itself.

**Why it matters here.** In the original encoder, self-attention and the FFN are each wrapped with a residual connection and layer normalization. The FFN transforms features at each position after attention has combined information across positions. Source: Vaswani et al. (2017), “Attention Is All You Need,” Advances in Neural Information Processing Systems.

**Appears in:** [[week-1-notes]]; [[week-1-exam-focused-notes]]

## Related
- [[self-attention]]
- [[residual-connections]]
- [[layer-normalization]]
- [[encoder-layer-parameter-count]]
