---
title: Sinusoidal Positional Encoding
tags: [llm/concept, transformer, position]
---

# Sinusoidal Positional Encoding

**Formal definition.** The original Transformer adds a fixed position-dependent vector to each token embedding. For even model width $d$, position $p$, and $0\le i<d/2$, Vaswani et al. (2017) define $PE(p,2i)=\sin(p/10000^{2i/d}),\qquad PE(p,2i+1)=\cos(p/10000^{2i/d}).$ Each pair has a common angular frequency. For a fixed offset $k$, angle-addition identities express the pair at $p+k$ as a linear transformation of the pair at $p$; the full transformation is block diagonal. Both the embedding and positional vector have shape `[d]`. This construction makes relative offsets representable through linear transformations, but does not force attention to use them.

**Why it matters here.** Unmasked self-attention without positional information is permutation equivariant, so order must enter the model to distinguish reordered inputs. The ebook's claim that distance grows roughly monotonically with offset is not generally valid: the squared distance is $2\sum_i[1-\cos(k/10000^{2i/d})]$, an oscillatory function. Encodings are computable beyond training lengths, but that alone does not guarantee successful length extrapolation.

**Appears in:** [[week-2-notes]]; [[week-2-exam-focused-notes]]

## Related
- [[self-attention]]
- [[masked-self-attention]]
- [[bert-input-embeddings]]
