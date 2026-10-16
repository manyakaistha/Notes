---
title: Transformer
slug: transformer
tags: ['ns/concept', 'ns/deep-learning', 'ns/transformers', 'ns/architecture']
introduced: "[[week-4-report|Week 4]]"
---

# Transformer

A Transformer is a deep learning architecture based entirely on stacked multi-head self-attention mechanisms, feed-forward sub-layers, residual skip connections, and layer normalization, eschewing sequential recurrence and temporal convolutions entirely.

Because transformers process all sequence elements in parallel without intrinsic sequential bias, explicit **positional encodings** $P \in \mathbb{R}^{N 	imes d}$ (sinusoidal or learned vectors) must be added to the input token embeddings to provide temporal order information. 

In EEG applications, continuous multi-channel signals are tokenized into temporal patches (e.g., 500 ms windows) or spatial sensor tokens. Multi-head self-attention projects the tokens into multiple representation subspaces simultaneously, capturing distinct electrophysiological rhythms (e.g., simultaneous theta-gamma phase-amplitude coupling) across disparate cortical areas. Transformers require extensive pre-training corpora (such as foundation models) to avoid severe overfitting on small clinical cohorts.

**Why it matters here:** Transformers represent the modern paradigm for sequence modeling and neuroimaging foundation models, enabling non-local temporal and cross-channel context aggregation.

**Appears in:** [[week-4-report|Week 4]] · [[w4d4|W4D4]]

## Related
- [[self-attention]]
- [[recurrent-neural-network]]
- [[representation-learning]]
