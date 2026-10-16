---
title: Autoregressive Next-Token Prediction & Decoding
slug: autoregressive-decoding
tags: ['vt/concept', 'vt/nlp', 'vt/llm', 'vt/decoding']
introduced: "[[w1d2|W1D2]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Autoregressive Next-Token Prediction & Decoding

Autoregressive next-token prediction is the generative modeling paradigm where a sequence model predicts the conditional probability distribution over a discrete vocabulary $\mathcal{V}$ for the next token given all preceding tokens:
$$P(\mathbf{w}_{1:T}) = \prod_{t=1}^T P(w_t \mid w_{<t})$$

### 1. The Output Projection & Softmax
After the final Transformer block, the output hidden vector $\mathbf{z}_T \in \mathbb{R}^D$ is layer-normalized and projected into unnormalized log-probability scores (logits) across the vocabulary via a linear projection head:
$$\mathbf{s}_t = \mathbf{z}_t \mathbf{W}_{out} + \mathbf{b}_{out} \in \mathbb{R}^{|\mathcal{V}|}$$
where $\mathbf{W}_{out} \in \mathbb{R}^{D \times |\mathcal{V}|}$ (e.g., $768 \times 50,257$ in GPT-2). Softmax normalizes logits into class probabilities:
$$P(w_t = k \mid w_{<t}) = \frac{\exp(s_{t, k} / T)}{\sum_{j=1}^{|\mathcal{V}|} \exp(s_{t, j} / T)}$$
where $T > 0$ denotes sampling **temperature**.

### 2. Decoding Strategies & Temperature Dynamics
- **Greedy Decoding ($T \to 0$ / Argmax):** Deterministically selects the token with the highest predicted probability:
  $$\hat{w}_t = \arg\max_k P(w_t = k \mid w_{<t})$$
  While stable, pure greedy search tends to produce repetitive, generic, or looping text.
- **Stochastic Sampling & Temperature ($T$):** Samples from the probability distribution:
  - Low Temperature ($T < 1.0$): Sharpens the distribution, emphasizing top choices and reducing variance.
  - High Temperature ($T > 1.0$): Flattens the distribution, granting lower-probability tokens higher selection odds, increasing creative diversity at the risk of semantic hallucinations.

### 3. Self-Supervised Training via Teacher Forcing
During training, raw text corpus data provides both the inputs and the supervisory ground-truth targets without human annotation:
$$\text{Input: } [w_1, w_2, \dots, w_{t-1}] \implies \text{Target: } w_t$$
The cross-entropy loss across all sequence positions is computed in parallel:
$$\mathcal{L} = -\sum_{t=1}^T \log P(w_t^* \mid w_{<t})$$

**Why it matters here:** While Vision Transformers (ViT) perform image-level category prediction via a classification MLP head, generative vision-language models (e.g. multimodal LLMs) utilize autoregressive next-token prediction to generate natural language descriptions conditioned on visual token embeddings.

**Appears in:** [[w1d2|W1D2]] · [[week-1-report|Week 1]]

## Related
- [[token-embedding]]
- [[byte-pair-encoding]]
- [[vision-transformer]]
- [[self-attention]]
