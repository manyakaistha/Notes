---
title: Long Short-Term Memory
slug: long-short-term-memory
tags: ['ns/concept', 'ns/deep-learning', 'ns/sequential', 'ns/rnn']
introduced: "[[week-4-report|Week 4]]"
---

# Long Short-Term Memory

Long Short-Term Memory (LSTM) is a specialized recurrent neural network architecture designed to resolve the vanishing gradient problem over extended sequence lengths. An LSTM cell augments the hidden state $h_t$ with an explicit memory cell state $c_t \in \mathbb{R}^H$ that acts as a linear "constant error carousel," allowing gradients to flow uninterrupted over hundreds of time steps.

Information flow is governed by three multiplicative sigmoid gating mechanisms:
1. **Forget Gate ($f_t$):** Controls what portion of the historical cell state to discard ($f_t pprox 0$) or retain ($f_t pprox 1$).
2. **Input Gate ($i_t$):** Regulates how much new candidate information $	ilde{c}_t$ to inject into the persistent cell memory.
3. **Output Gate ($o_t$):** Determines what proportion of the updated cell state $c_t$ is exposed to the observable hidden state $h_t = o_t \odot 	anh(c_t)$.

In electrophysiology, LSTMs track persistent contextual states across multi-second epochs, such as transitional dynamics during polysomnography or evolving seizure discharge patterns.

**Why it matters here:** LSTMs overcome the memory horizon collapse of standard RNNs, providing robust temporal tracking across multi-second electrophysiological epochs.

**Appears in:** [[week-4-report|Week 4]] · [[w4d3|W4D3]]

## Related
- [[recurrent-neural-network]]
- [[self-attention]]
- [[transformer]]
