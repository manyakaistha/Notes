---
title: Task-Specific Input Transformations
tags: [llm/concept, transfer-learning, input-format]
---

# Task-Specific Input Transformations

**Formal definition.** Task-specific input transformations serialize structured examples into token sequences that a pretrained language model can process. GPT-1 adds learned start and end tokens and, where needed, delimiter tokens, preserving the structure through token order. For textual entailment, a premise and hypothesis are concatenated in that order with a delimiter. For question answering and commonsense reasoning, each candidate is combined with its context and question, processed independently, and assigned a scalar score; a softmax over candidate scores gives a choice distribution. The final-position hidden representation supplies the task head, with the same Transformer parameters reused across candidates. For symmetric semantic similarity, the paper processes both sentence orderings and combines their representations, rather than assuming an arbitrary ordering is inherently meaningful. The transformation supplies a consistent interface, but supervised task training and task-specific output scoring remain necessary in this setting; delimiters alone do not make the model solve a task.

**Why it matters here.** These transformations connect structured downstream inputs to GPT-1's sequential pretraining without redesigning its core architecture. The ebook's question–choice shorthand omits the document or context included in the paper's question-answering transformation.

**Appears in:** [[week-3-notes]]; [[week-3-exam-focused-notes]]

## Related
- [[pretrain-finetune-paradigm]]
- [[gpt-1-architecture]]
- [[causal-language-modelling]]
