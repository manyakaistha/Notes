---
title: Bahdanau Attention
tags: [llm/concept, attention, sequence-modeling]
---

# Bahdanau Attention

**Formal definition.** Bahdanau attention computes a distinct source context for each recurrent decoder step. Given encoder annotations $h_i$ and previous decoder state $s_{t-1}$, an additive alignment network produces $e_{ti}=v_a^\top\tanh(W_a s_{t-1}+U_a h_i)$. The normalized weights and context are

$$\alpha_{ti}=\frac{\exp(e_{ti})}{\sum_j\exp(e_{tj})},\qquad c_t=\sum_i\alpha_{ti}h_i.$$

For $n$ source positions and $m$ decoder steps, the alignment matrix has shape `[m, n]`; each row sums to one over source positions. The original model uses bidirectional encoder annotations and learns the alignment jointly with translation, without supervised alignments. Scores across source positions can be evaluated in parallel at a given step, but the recurrent dependency on $s_{t-1}$ prevents parallel computation of all decoder steps. Attention weights are model-dependent soft alignments, not guaranteed explanations or a guarantee of correct translation.

**Why it matters here.** Replacing a single fixed source summary with a query-dependent context relieves the encoder bottleneck and provides the conceptual bridge to encoder–decoder cross-attention. Source: Bahdanau, Cho and Bengio (2015), “Neural Machine Translation by Jointly Learning to Align and Translate,” International Conference on Learning Representations.

**Appears in:** [[week-1-notes]]; [[week-1-exam-focused-notes]]

## Related
- [[seq2seq-bottleneck]]
- [[cross-attention]]
- [[self-attention]]
