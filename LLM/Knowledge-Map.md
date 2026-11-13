---
title: LLM Knowledge Map
tags: [llm/moc, llm/transformers]
---

# LLM — Knowledge Map

This map organizes an IIT Madras DA5004-style course for Weeks 1–4, scoped by the supplied ebook. Later-week material is archived; the ebook guides scope and the primary papers adjudicate technical claims.

## Curriculum Spine

| Week | Notes | Exam-focused notes | Question bank | Core question | Status |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | [[week-1-notes]] | [[week-1-exam-focused-notes]] | [[week-1-exam-questions]] | How does attention address the seq2seq bottleneck and build an encoder? | Notes available; bank planned |
| 2 | [[week-2-notes]] | [[week-2-exam-focused-notes]] | [[week-2-exam-questions]] | How do decoder attention, position and training machinery complete the Transformer? | Notes available; bank planned |
| 3 | [[week-3-notes]] | [[week-3-exam-focused-notes]] | [[week-3-exam-questions]] | How do autoregressive language models support GPT pre-training and task adaptation? | Notes available; bank planned |
| 4 | [[week-4-notes]] | [[week-4-exam-focused-notes]] | [[week-4-exam-questions]] | How do decoding strategies and bidirectional BERT representations serve different language tasks? | Notes available; bank planned |

## Weekly Topics

Topic names follow `sources/topics.json`; links identify related concept notes. Some topics also require the weekly notes for full coverage.

### Week 1

- **Seq2seq and the RNN bottleneck** — [[seq2seq-bottleneck]]
- **Bahdanau (additive) attention** — [[bahdanau-attention]]
- **Self-attention and Q/K/V projections** — [[self-attention]]
- **Scaled dot-product attention and the sqrt(d_k) factor** — [[scaled-dot-product-attention]]
- **Multi-head attention and W^O** — [[multi-head-attention]]
- **Position-wise FFN and the encoder layer/stack** — [[position-wise-feed-forward]]
- **Parameter counting for the encoder** — [[encoder-layer-parameter-count]]

### Week 2

- **Teacher forcing** — [[teacher-forcing]]
- **Masked (causal) self-attention** — [[masked-self-attention]]
- **Cross-attention (encoder-decoder attention)** — [[cross-attention]]
- **Decoder output layer and decoder parameter counts** — [[cross-attention]] · [[encoder-layer-parameter-count]]
- **Sinusoidal positional encoding** — [[sinusoidal-positional-encoding]]
- **Residual connections, LayerNorm vs BatchNorm** — [[residual-connections]] · [[layer-normalization]]
- **Training tricks: label smoothing, warm-up schedule, dropout** — [[label-smoothing]] · [[learning-rate-warmup]]
- **The full Transformer data flow** — [[masked-self-attention]] · [[cross-attention]] · [[position-wise-feed-forward]]

### Week 3

- **Language-modelling objective, chain rule and perplexity** — [[language-modelling-objective]] · [[perplexity]]
- **Ways to parameterise a language model** — [[language-modelling-objective]]
- **Causal language modelling with a decoder-only stack** — [[causal-language-modelling]]
- **GPT-1 architecture and pre-train/fine-tune paradigm** — [[gpt-1-architecture]] · [[pretrain-finetune-paradigm]]
- **GPT-1 parameter count (117M)** — [[gpt-1-architecture]]
- **Pre-training objective and teacher forcing** — [[language-modelling-objective]] · [[teacher-forcing]]
- **Fine-tuning GPT and the fine-tuning loss** — [[pretrain-finetune-paradigm]] · [[gpt-1-architecture]]
- **Task-specific input transformations** — [[task-specific-input-transformations]]

### Week 4

- **Exhaustive and greedy search** — [[greedy-and-beam-search]]
- **Beam search and length normalisation** — [[greedy-and-beam-search]]
- **Top-k sampling and temperature** — [[top-k-top-p-sampling]] · [[temperature-scaling]]
- **Top-p (nucleus) sampling and repetition issues** — [[top-k-top-p-sampling]]
- **BERT masked language modelling (80/10/10)** — [[bert-masked-language-modelling]]
- **NSP, [CLS]/[SEP] and the three input embeddings** — [[next-sentence-prediction]] · [[bert-input-embeddings]]
- **BERT sizes and parameter counts** — [[bert-parameter-count]]
- **BERT fine-tuning heads and GPT vs BERT** — [[pretrain-finetune-paradigm]] · [[bert-masked-language-modelling]] · [[gpt-1-architecture]]

## Concept Taxonomy

### Attention family
[[seq2seq-bottleneck]] · [[bahdanau-attention]] · [[self-attention]] · [[scaled-dot-product-attention]] · [[multi-head-attention]] · [[masked-self-attention]] · [[cross-attention]]

### Transformer components & training
[[position-wise-feed-forward]] · [[encoder-layer-parameter-count]] · [[sinusoidal-positional-encoding]] · [[residual-connections]] · [[layer-normalization]] · [[label-smoothing]] · [[learning-rate-warmup]] · [[teacher-forcing]]

### Language modelling & GPT
[[language-modelling-objective]] · [[perplexity]] · [[causal-language-modelling]] · [[gpt-1-architecture]] · [[pretrain-finetune-paradigm]] · [[task-specific-input-transformations]]

### Decoding
[[greedy-and-beam-search]] · [[top-k-top-p-sampling]] · [[temperature-scaling]]

### BERT
[[bert-masked-language-modelling]] · [[bert-input-embeddings]] · [[next-sentence-prediction]] · [[bert-parameter-count]]

## Central Governing Paradigm

```text
seq2seq → attention → self-attention → full Transformer
                                          ├→ decoder-only (GPT) → decoding
                                          └→ encoder-only (BERT) → task predictions
```

## Exam Resources and Sources

- [[exam-question-index]] and [[exam-study-guide]] are planned resources; weekly question banks are also planned.
- `LLM/sources/` contains the ebook, transcripts, exam papers, topic inventory and supporting records. Consult [[ebook-discrepancies]] for flagged claims and [[decisions]] for the decision log.

**Answer provenance:** All answers are derived; none are official answer keys.
