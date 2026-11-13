---
title: BERT Masked Language Modelling
tags: [llm/concept, bert, pretraining]
---

# BERT Masked Language Modelling

**Formal definition.** BERT's masked language modelling (MLM) pre-training objective selects 15% of WordPiece token positions for prediction and minimizes cross-entropy for their original identities using the corrupted sequence as input. Among selected positions, 80% are replaced by `[MASK]`, 10% by a random token, and 10% are left unchanged; the loss is evaluated on selected positions in all three cases. The bidirectional encoder can use both left and right context, unlike an ordinary left-to-right next-token predictor. The unchanged and random cases reduce the mismatch between pre-training corruption and downstream inputs and prevent the model from relying exclusively on `[MASK]` as a prediction cue. These percentages describe the original BERT recipe, not an optimal masking rate for every model or task. The ebook's claim that BERT's only mask is the `[MASK]` token is incorrect: padding attention masks can also exclude padding positions, although BERT does not use GPT's causal attention mask. Devlin et al. (2019), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, define this objective alongside next sentence prediction.

**Why it matters here.** MLM explains how self-supervised token prediction trains contextual representations that can later be adapted to classification and span tasks.

**Appears in:** [[week-4-notes]]; [[week-4-exam-focused-notes]]

## Related
- [[next-sentence-prediction]]
- [[bert-input-embeddings]]
- [[causal-language-modelling]]
- [[pretrain-finetune-paradigm]]
