---
title: Next Sentence Prediction
tags: [llm/concept, bert, pretraining]
---

# Next Sentence Prediction

**Formal definition.** Next sentence prediction (NSP) is the binary sentence-pair pre-training task used in original BERT. Given an input formatted as `[CLS]` A `[SEP]` B `[SEP]`, a classifier uses the final representation at `[CLS]` to predict whether B is the actual continuation of A or a randomly sampled sentence. Devlin et al. (2019) construct training pairs with 50% actual continuations and 50% random alternatives. The random class describes the sampling procedure, not a logical guarantee that every random pair is semantically unrelated or cannot occur consecutively. NSP is trained jointly with masked language modelling: the corresponding prediction losses are combined, while the encoder parameters are shared. The first input segment and second input segment may contain more than one natural sentence, despite the task's name. The ebook's claim that NSP was shown categorically harmful or redundant is too strong: later ablations changed other training choices as well, so better results without NSP do not isolate a universal causal effect of removing it.

**Why it matters here.** NSP illustrates how BERT adds a sequence-level objective to token-level pre-training and prepares a pooled representation for pair classification.

**Appears in:** [[week-4-notes]]; [[week-4-exam-focused-notes]]

## Related
- [[bert-masked-language-modelling]]
- [[bert-input-embeddings]]
- [[pretrain-finetune-paradigm]]
- [[task-specific-input-transformations]]
