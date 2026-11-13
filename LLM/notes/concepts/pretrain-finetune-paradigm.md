---
title: Pretrain–Fine-tune Paradigm
tags: [llm/concept, transfer-learning, gpt]
---

# Pretrain–Fine-tune Paradigm

**Formal definition.** The pretrain–fine-tune paradigm first estimates general model parameters from a large source corpus, then initializes a task model with those parameters and optimizes a target-task objective. In GPT-1, pretraining maximizes the log likelihood of tokens conditioned on preceding tokens in a finite window. Supervised fine-tuning transforms labeled examples into token sequences, adds a linear task output layer, and updates the pretrained Transformer as well as the new parameters. For classification, the final position's representation supplies the classifier: causal masking permits it to incorporate the preceding input, although it does not guarantee a perfect summary. The paper also optimizes an auxiliary language-model objective during fine-tuning, combining supervised and language-model log likelihoods as $L_3=L_2+\lambda L_1$. With losses defined as negatives, the equivalent minimization adds the corresponding weighted losses. Transfer can improve data efficiency but its benefit depends on source data, task, and optimization; it does not guarantee better performance or prevent forgetting.

**Why it matters here.** This two-stage mechanism explains GPT-1's transfer from unlabeled BookCorpus to diverse labeled language-understanding tasks. The paper motivates its auxiliary objective by improved generalization and faster convergence; the ebook's claim that it slows catastrophic forgetting is not established by that statement.

**Appears in:** [[week-3-notes]]; [[week-3-exam-focused-notes]]

## Related
- [[language-modelling-objective]]
- [[gpt-1-architecture]]
- [[task-specific-input-transformations]]
- [[teacher-forcing]]
