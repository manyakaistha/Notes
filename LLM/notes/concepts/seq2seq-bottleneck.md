---
title: Sequence-to-Sequence Bottleneck
tags: [llm/concept, sequence-modeling, encoder-decoder]
---

# Sequence-to-Sequence Bottleneck

**Formal definition.** In a fixed-context recurrent encoder–decoder, an encoder maps a variable-length source sequence to a fixed-width context $c$, and a decoder models $p(y\mid x)=\prod_t p(y_t\mid y_{<t},c)$. The context is the only source-dependent channel available to the decoder, so its finite representation must preserve information relevant to every output step. This architectural constraint can make long or information-rich inputs difficult to represent; it does not imply that every longer sentence must yield a worse translation. Recurrent updates also impose sequential dependencies over time and may suffer vanishing or exploding gradients. Attention instead supplies a target-dependent weighted combination of encoder states at each decoding step, although the recurrent decoder remains sequential.

**Why it matters here.** The ebook defines seq2seq as necessarily consisting of two RNNs with a single fixed-length context. That describes an early architecture, not seq2seq in general: the Transformer is also an encoder–decoder sequence transducer and uses attention rather than recurrence (Vaswani et al., 2017, “Attention Is All You Need,” Advances in Neural Information Processing Systems).

**Appears in:** [[week-1-notes]]; [[week-1-exam-focused-notes]]

## Related
- [[bahdanau-attention]]
- [[cross-attention]]
- [[self-attention]]
