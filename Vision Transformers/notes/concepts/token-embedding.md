---
title: Token Embedding
slug: token-embedding
tags: ['vt/concept', 'vt/nlp', 'vt/representation', 'vt/transformer']
introduced: "[[w1d2|W1D2]]"
reinforced: ["[[week-1-report|Week 1]]"]
---

# Token Embedding

A token embedding is a continuous, dense vector representation $\mathbf{e} \in \mathbb{R}^D$ that maps a discrete categorical token identifier $i \in \{0, 1, \dots, |\mathcal{V}| - 1\}$ from a fixed vocabulary dictionary $\mathcal{V}$ into a high-dimensional continuous latent space.

### 1. Vector Lookup Mechanics
In deep sequence models, token embedding is parameterized by an embedding matrix $\mathbf{W}_e \in \mathbb{R}^{|\mathcal{V}| \times D}$, where $|\mathcal{V}|$ is the vocabulary size (e.g., $50,257$ in GPT-2) and $D$ is the model's hidden dimension (e.g., $D = 768$ in GPT-2 Base, $D = 12,288$ in GPT-3 175B). For a one-hot encoded token $\mathbf{t}_i \in \{0, 1\}^{|\mathcal{V}|}$, the embedding lookup corresponds to:
$$\mathbf{e}_i = \mathbf{t}_i \mathbf{W}_e = \mathbf{W}_e[i, :]$$

### 2. Semantic Clustering in High-Dimensional Space
Each orthogonal dimension of the $D$-dimensional vector acts as an abstract coordinate capturing distributed semantic properties:
- **Semantic Subspaces:** Tokens sharing contextual or functional roles naturally cluster into shared subspaces. For example, animal vectors (`cat`, `dog`) point in aligned directions within the animal subspace, whereas fruit vectors (`apple`, `orange`) cluster in a distinct subspace.
- **Directional Alignment:** Cosine similarity measures semantic alignment:
  $$\cos(\theta) = \frac{\mathbf{e}_1 \cdot \mathbf{e}_2}{\|\mathbf{e}_1\| \|\mathbf{e}_2\|}$$

### 3. The Vision Analogy
While language models look up static vectors in $\mathbf{W}_e$ using discrete token IDs, Vision Transformers construct token embeddings via linear projection of flattened continuous image patches:
$$\mathbf{e}_{patch} = \mathbf{x}_p \mathbf{E} \in \mathbb{R}^D$$
In both domains, the resulting sequence enters the Transformer as a matrix of shape $[N, D]$.

**Why it matters here:** Token embeddings translate discrete symbols into the language of linear algebra, providing the raw semantic substrate to which positional embeddings are added.

**Appears in:** [[w1d2|W1D2]] · [[week-1-report|Week 1]]

## Related
- [[byte-pair-encoding]]
- [[positional-embedding]]
- [[patch-projection]]
- [[vision-transformer]]
