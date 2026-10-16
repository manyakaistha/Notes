---
title: Self-Attention
slug: self-attention
tags: ['ns/concept', 'ns/deep-learning', 'ns/attention']
introduced: "[[week-4-report|Week 4]]"
---

# Self-Attention

Self-attention is an architectural mechanism that computes direct pairwise associative relationships between all positions within an input sequence, regardless of their temporal distance. For an input representation matrix $X \in \mathbb{R}^{N 	imes d}$, self-attention linearly projects elements into Query ($Q$), Key ($K$), and Value ($V$) matrices:
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V \quad (W \in \mathbb{R}^{d 	imes d_k})$$
The associative affinity between positions $i$ and $j$ is computed via scaled dot-product similarity, normalized across keys using the softmax function:
$$A = 	ext{Attention}(Q, K, V) = 	ext{softmax}\left(rac{Q K^T}{\sqrt{d_k}}ight) V$$
Each element of the attention weight matrix $A_{i,j} \in [0, 1]$ dictates how much contextual information from token $j$ is aggregated into the updated representation of token $i$. Self-attention exhibits an $\mathcal{O}(N^2)$ computational complexity, requiring continuous EEG signals to be divided into temporal tokens (time patches) before attention processing. Importantly, attention weights reflect computational query-key routing, not physiological functional connectivity.

**Why it matters here:** Self-attention enables neural decoders to model long-range temporal co-dependencies (e.g., pre-cue readiness potentials to post-cue motor execution) without sequential recurrence bottlenecks.

**Appears in:** [[week-4-report|Week 4]] · [[w4d4|W4D4]]

## Related
- [[transformer]]
- [[long-short-term-memory]]
- [[recurrent-neural-network]]
