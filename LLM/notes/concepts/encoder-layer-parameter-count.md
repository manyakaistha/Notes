---
title: Encoder Layer Parameter Count
tags: [llm/concept, transformer, parameter-counting]
---

# Encoder Layer Parameter Count

**Formal definition.** Let $d=d_{\rm model}$ and $f=d_{\rm ff}$. In standard multi-head attention with $h d_k=h d_v=d$, the query, key, value, and output projection weights total $4d^2$. The position-wise FFN adds $2df$ weights and $f+d$ biases. Thus, excluding attention biases and normalization, one layer has $4d^2+2df+f+d$ trainable parameters. If attention projections each include a bias, add $4d$; two layer normalizations with learned scale and offset add another $4d$. These additions depend on the implementation and must not be silently assumed. A stack of $N$ independently parameterized identical layers multiplies the per-layer count by $N$; embeddings and output heads are separate. Sequence length changes activation sizes, not these shared weight counts.

**Why it matters here.** At fixed $d$ and standard head widths, changing $h$ does not alter the projection weight total. The ebook's multiple-choice explanation says “The only extra cost is $W^O$” for multi-head versus single-head attention; this is false when both use the paper's output projection. Both then have $4d^2$ projection weights. Source: Vaswani et al. (2017), “Attention Is All You Need,” Advances in Neural Information Processing Systems.

**Appears in:** [[week-1-notes]]; [[week-1-exam-focused-notes]]

## Related
- [[multi-head-attention]]
- [[position-wise-feed-forward]]
- [[layer-normalization]]
- [[bert-parameter-count]]
