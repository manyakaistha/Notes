---
title: BERT Parameter Count
tags: [llm/concept, bert, parameters]
---

# BERT Parameter Count

**Formal definition.** A BERT parameter count sums distinct trainable tensors, counting tied weights once and specifying whether pre-training or downstream heads are included. For vocabulary size $V$, maximum positions $P$, segment types $S$, hidden width $H$, feed-forward width $I$, and $L$ encoder layers, the embedding tables contain $(V+P+S)H$ weights. Each standard encoder layer has $4H^2+2HI$ attention and feed-forward matrix weights: four `[H, H]` attention projections and two feed-forward matrices of shapes `[H, I]` and `[I, H]`. Its attention and feed-forward biases add $5H+I$, and two layer-normalization scale and shift pairs add $4H$, for $4H^2+2HI+9H+I$ parameters per layer. Embedding normalization, pooler, and task or pre-training heads must be accounted for separately. Devlin et al. (2019) report approximately 110 million parameters for BERT-base and 340 million for BERT-large. The ebook's $7$ million per base layer is premature rounding: its own approximately 2.36 million attention plus 4.72 million feed-forward figures give about 7.08 million, so multiplying 7 million by 12 understates the stack. Exact totals depend on vocabulary and included components.

**Why it matters here.** Explicit accounting explains the reported scale and prevents approximate intermediate rounding or shared embeddings from distorting comparisons.

**Appears in:** [[week-4-notes]]; [[week-4-exam-focused-notes]]

## Related
- [[bert-input-embeddings]]
- [[encoder-layer-parameter-count]]
- [[multi-head-attention]]
- [[position-wise-feed-forward]]
