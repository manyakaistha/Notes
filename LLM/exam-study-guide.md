---
title: "LLM Exam Study Guide — Weeks 1–4"
status: complete
date: 2026-11-13
map: "[[Knowledge-Map]]"
tags: [llm/exam-study, llm/moc]
---

# LLM Exam Study Guide — Weeks 1–4

**Navigation:** [[Knowledge-Map\|Knowledge Map]] · [[exam-question-index\|Question index]]

## Frequency and marks ranking

For each topic, **score = occurrences × mean marks per occurrence = total routed marks** across its directly assigned questions. Occurrences are repeated paper records, so an item repeated across papers contributes repeatedly. This is a descriptive review-priority measure for the supplied corpus, not a prediction of future exam marks; questions may carry multiple topic IDs.

| Rank | Week | Topic ID and name | Occurrences | Total marks | Score | Bank checklist | Study notes |
| ---: | ---: | :--- | ---: | ---: | ---: | :--- | :--- |
| 1 | 1 | `w1-self-attention-qkv` — Self-attention and Q/K/V projections | 30 | 81 | 81 | [[week-1-exam-questions#w1-self-attention-qkv\|Checklist]] | [[week-1-notes\|Week 1 notes]] |
| 2 | 4 | `w4-topk-temperature` — Top-k sampling and temperature | 22 | 61 | 61 | [[week-4-exam-questions#w4-topk-temperature\|Checklist]] | [[week-4-notes\|Week 4 notes]] |
| 3 | 1 | `w1-multi-head-attention` — Multi-head attention and W^O | 17 | 45 | 45 | [[week-1-exam-questions#w1-multi-head-attention\|Checklist]] | [[week-1-notes\|Week 1 notes]] |
| 4 | 2 | `w2-residual-layernorm` — Residual connections, LayerNorm vs BatchNorm | 21 | 42 | 42 | [[week-2-exam-questions#w2-residual-layernorm\|Checklist]] | [[week-2-notes\|Week 2 notes]] |
| 5 | 4 | `w4-exhaustive-greedy` — Exhaustive and greedy search | 15 | 42 | 42 | [[week-4-exam-questions#w4-exhaustive-greedy\|Checklist]] | [[week-4-notes\|Week 4 notes]] |
| 6 | 4 | `w4-topp-nucleus` — Top-p (nucleus) sampling and repetition issues | 15 | 40 | 40 | [[week-4-exam-questions#w4-topp-nucleus\|Checklist]] | [[week-4-notes\|Week 4 notes]] |
| 7 | 2 | `w2-masked-self-attention` — Masked (causal) self-attention | 15 | 39 | 39 | [[week-2-exam-questions#w2-masked-self-attention\|Checklist]] | [[week-2-notes\|Week 2 notes]] |
| 8 | 1 | `w1-encoder-parameter-count` — Parameter counting for the encoder | 14 | 37 | 37 | [[week-1-exam-questions#w1-encoder-parameter-count\|Checklist]] | [[week-1-notes\|Week 1 notes]] |
| 9 | 4 | `w4-beam-search` — Beam search and length normalisation | 13 | 36 | 36 | [[week-4-exam-questions#w4-beam-search\|Checklist]] | [[week-4-notes\|Week 4 notes]] |
| 10 | 1 | `w1-scaled-dot-product` — Scaled dot-product attention and the sqrt(d_k) factor | 12 | 31 | 31 | [[week-1-exam-questions#w1-scaled-dot-product\|Checklist]] | [[week-1-notes\|Week 1 notes]] |
| 11 | 2 | `w2-positional-encoding` — Sinusoidal positional encoding | 11 | 28 | 28 | [[week-2-exam-questions#w2-positional-encoding\|Checklist]] | [[week-2-notes\|Week 2 notes]] |
| 12 | 4 | `w4-bert-finetuning` — BERT fine-tuning heads and GPT vs BERT | 11 | 26 | 26 | [[week-4-exam-questions#w4-bert-finetuning\|Checklist]] | [[week-4-notes\|Week 4 notes]] |
| 13 | 3 | `w3-language-modelling` — Language-modelling objective, chain rule and perplexity | 10 | 22 | 22 | [[week-3-exam-questions#w3-language-modelling\|Checklist]] | [[week-3-notes\|Week 3 notes]] |
| 14 | 3 | `w3-causal-lm` — Causal language modelling with a decoder-only stack | 8 | 22 | 22 | [[week-3-exam-questions#w3-causal-lm\|Checklist]] | [[week-3-notes\|Week 3 notes]] |
| 15 | 2 | `w2-teacher-forcing` — Teacher forcing | 8 | 18 | 18 | [[week-2-exam-questions#w2-teacher-forcing\|Checklist]] | [[week-2-notes\|Week 2 notes]] |
| 16 | 1 | `w1-ffn-encoder-stack` — Position-wise FFN and the encoder layer/stack | 6 | 17 | 17 | [[week-1-exam-questions#w1-ffn-encoder-stack\|Checklist]] | [[week-1-notes\|Week 1 notes]] |
| 17 | 3 | `w3-gpt1-parameter-count` — GPT-1 parameter count (117M) | 7 | 16 | 16 | [[week-3-exam-questions#w3-gpt1-parameter-count\|Checklist]] | [[week-3-notes\|Week 3 notes]] |
| 18 | 4 | `w4-bert-nsp-inputs` — NSP, [CLS]/[SEP] and the three input embeddings | 6 | 14 | 14 | [[week-4-exam-questions#w4-bert-nsp-inputs\|Checklist]] | [[week-4-notes\|Week 4 notes]] |
| 19 | 3 | `w3-pretraining-objective` — Pre-training objective and teacher forcing | 5 | 14 | 14 | [[week-3-exam-questions#w3-pretraining-objective\|Checklist]] | [[week-3-notes\|Week 3 notes]] |
| 20 | 2 | `w2-cross-attention` — Cross-attention (encoder-decoder attention) | 5 | 12 | 12 | [[week-2-exam-questions#w2-cross-attention\|Checklist]] | [[week-2-notes\|Week 2 notes]] |
| 21 | 3 | `w3-finetuning` — Fine-tuning GPT and the fine-tuning loss | 4 | 12 | 12 | [[week-3-exam-questions#w3-finetuning\|Checklist]] | [[week-3-notes\|Week 3 notes]] |
| 22 | 1 | `w1-bahdanau-attention` — Bahdanau (additive) attention | 4 | 10 | 10 | [[week-1-exam-questions#w1-bahdanau-attention\|Checklist]] | [[week-1-notes\|Week 1 notes]] |
| 23 | 2 | `w2-full-transformer` — The full Transformer data flow | 4 | 10 | 10 | [[week-2-exam-questions#w2-full-transformer\|Checklist]] | [[week-2-notes\|Week 2 notes]] |
| 24 | 3 | `w3-gpt1-architecture` — GPT-1 architecture and pre-train/fine-tune paradigm | 4 | 9 | 9 | [[week-3-exam-questions#w3-gpt1-architecture\|Checklist]] | [[week-3-notes\|Week 3 notes]] |
| 25 | 2 | `w2-decoder-output-and-params` — Decoder output layer and decoder parameter counts | 4 | 8 | 8 | [[week-2-exam-questions#w2-decoder-output-and-params\|Checklist]] | [[week-2-notes\|Week 2 notes]] |
| 26 | 4 | `w4-bert-mlm` — BERT masked language modelling (80/10/10) | 3 | 6 | 6 | [[week-4-exam-questions#w4-bert-mlm\|Checklist]] | [[week-4-notes\|Week 4 notes]] |
| 27 | 1 | `w1-seq2seq-bottleneck` — Seq2seq and the RNN bottleneck | 2 | 5 | 5 | [[week-1-exam-questions#w1-seq2seq-bottleneck\|Checklist]] | [[week-1-notes\|Week 1 notes]] |
| 28 | 4 | `w4-bert-architecture-params` — BERT sizes and parameter counts | 2 | 5 | 5 | [[week-4-exam-questions#w4-bert-architecture-params\|Checklist]] | [[week-4-notes\|Week 4 notes]] |
| 29 | 2 | `w2-training-tricks` — Training tricks: label smoothing, warm-up schedule, dropout | 1 | 3 | 3 | [[week-2-exam-questions#w2-training-tricks\|Checklist]] | [[week-2-notes\|Week 2 notes]] |
| 30 | 3 | `w3-task-input-formats` — Task-specific input transformations | 1 | 3 | 3 | [[week-3-exam-questions#w3-task-input-formats\|Checklist]] | [[week-3-notes\|Week 3 notes]] |
| 31 | 3 | `w3-model-parameterisations` — Ways to parameterise a language model | 0 | 0 | 0 | [[week-3-exam-questions#w3-model-parameterisations\|Checklist]] | [[week-3-notes\|Week 3 notes]] |

## Reading order by week

**Week 1.** Read the linked notes in their syllabus order: Seq2seq and the RNN bottleneck → Bahdanau (additive) attention → Self-attention and Q/K/V projections → Scaled dot-product attention and the sqrt(d_k) factor → Multi-head attention and W^O → Position-wise FFN and the encoder layer/stack → Parameter counting for the encoder. Then use the linked question bank checklist to identify direct evidence and work the routed occurrences in paper order. Topics without direct evidence remain part of the curriculum and should still be revised.

**Week 2.** Read the linked notes in their syllabus order: Teacher forcing → Masked (causal) self-attention → Cross-attention (encoder-decoder attention) → Decoder output layer and decoder parameter counts → Sinusoidal positional encoding → Residual connections, LayerNorm vs BatchNorm → Training tricks: label smoothing, warm-up schedule, dropout → The full Transformer data flow. Then use the linked question bank checklist to identify direct evidence and work the routed occurrences in paper order. Topics without direct evidence remain part of the curriculum and should still be revised.

**Week 3.** Read the linked notes in their syllabus order: Language-modelling objective, chain rule and perplexity → Ways to parameterise a language model → Causal language modelling with a decoder-only stack → GPT-1 architecture and pre-train/fine-tune paradigm → GPT-1 parameter count (117M) → Pre-training objective and teacher forcing → Fine-tuning GPT and the fine-tuning loss → Task-specific input transformations. Then use the linked question bank checklist to identify direct evidence and work the routed occurrences in paper order. Topics without direct evidence remain part of the curriculum and should still be revised.

**Week 4.** Read the linked notes in their syllabus order: Exhaustive and greedy search → Beam search and length normalisation → Top-k sampling and temperature → Top-p (nucleus) sampling and repetition issues → BERT masked language modelling (80/10/10) → NSP, [CLS]/[SEP] and the three input embeddings → BERT sizes and parameter counts → BERT fine-tuning heads and GPT vs BERT. Then use the linked question bank checklist to identify direct evidence and work the routed occurrences in paper order. Topics without direct evidence remain part of the curriculum and should still be revised.
