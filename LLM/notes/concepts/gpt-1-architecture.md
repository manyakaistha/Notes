---
title: GPT-1 Architecture
tags: [llm/concept, gpt, transformer]
---

# GPT-1 Architecture

**Formal definition.** GPT-1 is a decoder-only Transformer language model introduced by Radford, Narasimhan, Salimans, and Sutskever in 2018. Its stack has 12 Transformer layers, hidden width 768, and 12 attention heads; its position-wise feed-forward intermediate width is 3072. Token and learned position embeddings are added before the stack, which combines causally masked multi-head self-attention, feed-forward transformations, residual connections, and layer normalization. There is no encoder or encoder-decoder cross-attention. The paper's language-model output uses the transpose of the token embedding matrix: with hidden states `[T, 768]` and embedding matrix `[V, 768]`, multiplication by its transpose yields vocabulary logits `[T, V]`, followed by softmax. This weight tying means that the output matrix is not an additional independent embedding-sized parameter matrix. GPT-1 was pretrained on contiguous BookCorpus text with a finite context and transferred to supervised tasks.

**Why it matters here.** The ebook's parameter breakdown contains inconsistent subtotals and omits feed-forward biases in one calculation; it should not be used as an exact count. Its separate output-matrix description must also be read with the paper's explicit weight tying.

**Appears in:** [[week-3-notes]]; [[week-3-exam-focused-notes]]

## Related
- [[causal-language-modelling]]
- [[multi-head-attention]]
- [[position-wise-feed-forward]]
- [[pretrain-finetune-paradigm]]
