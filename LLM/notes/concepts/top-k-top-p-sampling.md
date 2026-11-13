---
title: Top-k and Top-p Sampling
tags: [llm/concept, decoding, sampling]
---

# Top-k and Top-p Sampling

**Formal definition.** Given a next-token probability distribution, top-$k$ sampling retains the $k$ highest-probability tokens, renormalizes their probabilities, and draws randomly from that restricted distribution. Nucleus or top-$p$ sampling sorts tokens by descending probability and retains the smallest prefix whose cumulative probability is at least $p$, where $0<p\leq1$; the token crossing the threshold is included. It then renormalizes and samples from that set. Thus the nucleus size adapts to the distribution: concentrated distributions can yield small sets, while flatter distributions can require more tokens. Ties require a specified ordering or tie policy. At $p=1$, the nucleus includes the entire positive-probability support, so this is ordinary sampling from that distribution. The ebook incorrectly labels this statement a mistake; its own explanation agrees with it. Temperature or other transformations applied before truncation change the distribution and can change the selected nucleus. Holtzman et al. (2020), *The Curious Case of Neural Text Degeneration*, motivate nucleus sampling by identifying degeneration under likelihood-maximizing decoding and an unreliable probability tail.

**Why it matters here.** These methods distinguish a fixed candidate count from an adaptive probability mass, without implying that either guarantees coherence or diversity.

**Appears in:** [[week-4-notes]]; [[week-4-exam-focused-notes]]

## Related
- [[greedy-and-beam-search]]
- [[temperature-scaling]]
- [[causal-language-modelling]]
