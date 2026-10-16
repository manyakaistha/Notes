---
title: Class Token ([CLS])
slug: class-token
tags: ['vt/concept', 'vt/deep-learning', 'vt/transformer', 'vt/classification']
introduced: "[[w1d1|W1D1]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Class Token ([CLS])

The class token, denoted $\mathbf{x}_{class} \in \mathbb{R}^D$ (often initialized as a learnable parameter `nn.Parameter(torch.randn(1, 1, D))`), is a dedicated token prepended to the sequence of patch embeddings prior to the initial Transformer encoder block.

### Historical Lineage & Purpose
Inherited directly from BERT (Devlin et al., 2018), where a `[CLS]` token is positioned at sequence index 0 to aggregate sequence-level sentence representations, the class token in Vision Transformers (Dosovitskiy et al., 2021) serves as an unbiased global aggregator of whole-image context.

The input sequence to the first Transformer layer is formulated as:
$$\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{x}_p^1 \mathbf{E}; \mathbf{x}_p^2 \mathbf{E}; \dots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}$$
where $\mathbf{z}_0 \in \mathbb{R}^{(N+1) \times D}$.

### Why Use a Dedicated Class Token Instead of Global Average Pooling?
1. **Unbiased Attention Routing:** Because the class token has no intrinsic spatial coordinates or tied patch pixels, it interacts symmetrically with all $N$ visual tokens across all self-attention layers based purely on semantic query-key matching.
2. **Representation Purity:** The Transformer encoder layers can freely adapt the patch tokens for local-to-global spatial relationships, while the class token specializes purely in synthesizing the global discriminative representation $\mathbf{z}_L^0$ required by the final classification MLP head:
   $$\hat{\mathbf{y}} = \operatorname{MLP}(\operatorname{LN}(\mathbf{z}_L^0))$$

**Why it matters here:** The class token eliminates the need for ad-hoc spatial feature flattening or unweighted average pooling across patch tokens, establishing a clean, standard interface between sequence attention and category prediction.

**Appears in:** [[w1d1|W1D1]] · [[week-1-report|Week 1]]

## Related
- [[vision-transformer]]
- [[patch-projection]]
- [[positional-embedding]]
