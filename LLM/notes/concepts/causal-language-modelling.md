---
title: Causal Language Modelling
tags: [llm/concept, language-modeling, causal-attention]
---

# Causal Language Modelling

**Formal definition.** Causal language modelling predicts each token from an allowed preceding context, defining $p_\theta(w_{1:T})=\prod_i p_\theta(w_i\mid w_{<i})$. In a decoder-only Transformer, a causal attention mask prevents a position from attending to later input positions; the stack has masked self-attention and position-wise feed-forward sublayers, without encoder cross-attention. With input tokens $w_1,\ldots,w_{T-1}$, the output at position $j$ predicts $w_{j+1}$. A start token supplies a prediction context for $w_1$ if that token is scored. Teacher forcing supplies observed prefixes, so all shifted predictions can be computed in parallel during training; generation instead feeds back selected tokens sequentially. A mask alone does not fix incorrect target alignment: predicting $w_i$ from a state that already receives $w_i$ leaks the answer. Likewise, padding and document boundaries require their own appropriate treatment.

**Why it matters here.** The ebook's output equation uses position $i$ to predict $w_i$, although its prose uses position $i-1$. The shifted alignment above resolves that inconsistency and matches the paper's prediction from preceding context.

**Appears in:** [[week-3-notes]]; [[week-3-exam-focused-notes]]

## Related
- [[language-modelling-objective]]
- [[masked-self-attention]]
- [[teacher-forcing]]
- [[gpt-1-architecture]]
